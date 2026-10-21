# -*- coding: utf-8 -*-
"""Unified evaluation protocol (stage 0, frozen for this round).

Rules (per review/BASELINE_B_REVIEW_AND_NEXT_STEPS.txt):
- One inference per image; score EVERY distinct GT annotation variant of that
  image; average within image, then within near-dup group, then group-equal.
- Primary metric: original-coords group-equal-mean Dice. IoU also reported.
- Empty masks are NOT skipped: both empty -> Dice/IoU = 1; exactly one empty
  -> 0. Empty-GT subset reported separately.
- Probability maps stay float32 end to end (no uint8); restore uses NEAREST
  this round (bilinear/threshold search explicitly out of scope); threshold
  fixed at 0.5.
- Canvas-space metrics are training diagnostics only; they are NOT comparable
  across input resolutions.
- Inference long_side is read from the checkpoint config unless explicitly
  overridden (override recorded in output metadata).
"""
import os, json, hashlib
import numpy as np
import torch
from PIL import Image

from .data import (UmudTrainData, UmudSegDataset, ROOT, read_mask,
                   align_mask_to_image, read_image_gray3,
                   restore_original_coords, load_manifest_index)

PROTOCOL_VERSION = 'orig-group-v1'
THRESHOLD = 0.5
RESTORE_METHOD = 'nearest'


def dice_iou(pred, gt):
    """pred/gt: binary uint8 arrays, same shape. Empty rules per protocol."""
    p, g = int(pred.sum()), int(gt.sum())
    if g == 0 and p == 0:
        return 1.0, 1.0
    if g == 0 or p == 0:
        return 0.0, 0.0
    inter = int((pred.astype(bool) & gt.astype(bool)).sum())
    return (2 * inter / (p + g), inter / (p + g - inter))


def build_dev_cache(task, manifest_index=None, include_filenames=None):
    """Precompute aligned GT variants per dev image (original coords)."""
    m = manifest_index or load_manifest_index()
    data = UmudTrainData(task, 'dev', manifest_index=m,
                         include_filenames=include_filenames)
    cache = []
    for s in data.samples:
        W, H = int(s['image']['width']), int(s['image']['height'])
        variants = []
        for mrec in s['masks']:
            g = align_mask_to_image(read_mask(os.path.join(ROOT, mrec['path'])), (W, H))
            variants.append(g)
        cache.append({'sample': s, 'orig_wh': (W, H), 'variants': variants,
                      'n_variants': len(variants)})
    return cache


@torch.no_grad()
def predict_image_probs(model, item, device, amp=True):
    x = item['image'].unsqueeze(0).to(device)
    with torch.autocast('cuda', enabled=amp and device == 'cuda'):
        pb = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return restore_original_coords(pb.astype(np.float32),
                                   tuple(item['orig_wh'].tolist()),
                                   float(item['scale']),
                                   tuple(item['pad'].tolist()),
                                   method=RESTORE_METHOD)


@torch.no_grad()
def evaluate_model(model, task, device, long_side, amp=True,
                   cache=None, save_dir=None, max_overlays=8,
                   model_tag='', dev_data=None):
    """Run the frozen protocol. Returns summary dict; optionally writes
    masks/probs/overlays/per-image CSV/breakdowns under save_dir."""
    model.eval()
    cache = cache or build_dev_cache(task)
    data = dev_data or UmudTrainData(task, 'dev')
    ds = UmudSegDataset(data, long_side=long_side, augment=False)
    assert len(ds) == len(cache)

    rows, group_rows = [], {}
    n_empty_gt_imgs, n_empty_pred_imgs = 0, 0
    for i in range(len(ds)):
        item = ds[i]
        prob = predict_image_probs(model, item, device, amp)
        pred = (prob > THRESHOLD).astype(np.uint8)
        c = cache[i]
        s = c['sample']
        vd, vi = [], []
        for g in c['variants']:
            d, u = dice_iou(pred, g)
            vd.append(d); vi.append(u)
        gt_empty = all(g.sum() == 0 for g in c['variants'])
        pred_empty = pred.sum() == 0
        n_empty_gt_imgs += int(gt_empty)
        n_empty_pred_imgs += int(pred_empty)
        fg_frac = float(np.mean([g.mean() for g in c['variants']]))
        row = {'filename': s['image']['filename'], 'group_id': s['group_id'],
               'width': c['orig_wh'][0], 'height': c['orig_wh'][1],
               'n_variants': c['n_variants'], 'gt_fg_frac': round(fg_frac, 6),
               'gt_empty': int(gt_empty), 'pred_empty': int(pred_empty),
               'pred_px': int(pred.sum()),
               'dice': float(np.mean(vd)), 'iou': float(np.mean(vi)),
               'dice_min_variant': float(np.min(vd))}
        rows.append(row)
        group_rows.setdefault(s['group_id'], []).append(row)
        if save_dir and i < max_overlays:
            _save_overlay(save_dir, s, c['variants'][0], pred)

    groups = []
    for gid, rs in group_rows.items():
        groups.append({'group_id': gid, 'n_images': len(rs),
                       'dice': float(np.mean([r['dice'] for r in rs])),
                       'iou': float(np.mean([r['iou'] for r in rs]))})
    dice_g = float(np.mean([g['dice'] for g in groups]))
    iou_g = float(np.mean([g['iou'] for g in groups]))
    dice_i = float(np.mean([r['dice'] for r in rows]))
    empty_gt_rows = [r for r in rows if r['gt_empty']]
    summary = {
        'protocol': PROTOCOL_VERSION, 'threshold': THRESHOLD,
        'restore': RESTORE_METHOD, 'prob_dtype': 'float32',
        'long_side': long_side, 'task': task, 'model_tag': model_tag,
        'n_images': len(rows), 'n_groups': len(groups),
        'dice_group': dice_g, 'iou_group': iou_g,
        'dice_image': dice_i, 'iou_image': float(np.mean([r['iou'] for r in rows])),
        'empty_pred_rate': n_empty_pred_imgs / max(len(rows), 1),
        'n_empty_gt_images': n_empty_gt_imgs,
        'empty_gt_dice_group': (float(np.mean([r['dice'] for r in empty_gt_rows]))
                                if empty_gt_rows else None),
        'breakdowns': _breakdowns(rows, groups),
    }
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import csv
        with open(os.path.join(save_dir, 'per_image.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with open(os.path.join(save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
    return summary, rows, groups


def _breakdowns(rows, groups):
    def bucket_fg(f):
        if f == 0: return '0'
        if f < 0.001: return '<0.1%'
        if f < 0.005: return '0.1-0.5%'
        return '>=0.5%'
    out = {}
    for name, keyfn in [('by_size', lambda r: f"{r['width']}x{r['height']}"),
                        ('by_fg', lambda r: bucket_fg(r['gt_fg_frac'])),
                        ('by_multi_mask', lambda r: 'multi' if r['n_variants'] > 1 else 'single')]:
        agg = {}
        for r in rows:
            agg.setdefault(keyfn(r), []).append(r['dice'])
        out[name] = {k: {'n': len(v), 'dice_mean': float(np.mean(v))}
                     for k, v in sorted(agg.items())}
    return out


def _save_overlay(save_dir, sample, gt, pred):
    od = os.path.join(save_dir, 'overlays'); os.makedirs(od, exist_ok=True)
    rgb = read_image_gray3(os.path.join(ROOT, sample['image']['path']))
    ov = rgb.copy()
    ov[gt == 1] = (0.6 * ov[gt == 1] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    ov[pred == 1] = (0.6 * ov[pred == 1] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(ov).save(os.path.join(
        od, sample['image']['filename'].replace('.tif', '.png')))


def read_ckpt_long_side(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck.get('config') or {}
    return int(cfg.get('long_side', 512)), ck
