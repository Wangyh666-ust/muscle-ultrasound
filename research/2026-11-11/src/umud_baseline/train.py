# -*- coding: utf-8 -*-
"""Model, losses, training loop for UMUD baseline B (stage A/E).

Run:
  python -m src.umud_baseline.train --task apo --config configs/baseline_apo.json \
      --mode full --out runs/baseline_b/apo
Modes: tiny (8-16 groups, no aug), pilot (3 epochs), full (60 epochs).
Resume: --resume runs/baseline_b/apo/checkpoints/last.pt
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from .data import UmudTrainData, UmudSegDataset, ROOT


def build_model(pretrained=True, architecture='resnet34_unet', encoder_name=None,
                encoder_weights=None):
    """Build the configured segmentation model.

    The default remains the historical ResNet34-UNet so old checkpoints and
    callers keep working. New runs record architecture/encoder in the config.
    """
    import segmentation_models_pytorch as smp
    if architecture in ('resnet34_unet', 'unet'):
        encoder_name = encoder_name or 'resnet34'
        cls = smp.Unet
    elif architecture in ('segformer_b0', 'segformer'):
        encoder_name = encoder_name or 'mit_b0'
        cls = smp.Segformer
    else:
        raise ValueError(f'Unknown architecture: {architecture}')
    weights = None if not pretrained else (encoder_weights or 'imagenet')
    return cls(encoder_name=encoder_name, encoder_weights=weights,
               in_channels=3, classes=1, activation=None)


def encoder_source_info(encoder_name, weights_name='imagenet'):
    """Record the SMP/HF pretrained source and local weight hash."""
    import segmentation_models_pytorch as smp
    settings = smp.encoders.encoders[encoder_name]['pretrained_settings'][weights_name]
    info = {'encoder_name': encoder_name, 'weights': weights_name,
            'repo_id': settings.get('repo_id'),
            'revision': settings.get('revision')}
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(settings['repo_id'], 'model.safetensors',
                                        revision=settings['revision'])
        if isinstance(cached, str):
            info['cache_file'] = cached
            info['sha256'] = sha256_file(cached)
    except Exception as exc:
        info['source_note'] = f'cache lookup failed: {exc}'
    return info


class FixedWeightedSampler(Sampler):
    """Weighted sampler with a per-epoch, arm-independent RNG sequence.

    The sampler consumes no global torch/numpy RNG state. Given identical
    weights, seed and epoch, the returned index sequence is identical for both
    model arms. The sequence is retained for the sampling-consistency record.
    """

    def __init__(self, weights, num_samples, seed):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self.last_indices = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.last_indices = None

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        idx = rng.choice(len(self.weights), size=self.num_samples,
                         replace=True, p=self.weights)
        self.last_indices = [int(v) for v in idx.tolist()]
        return iter(self.last_indices)

    def __len__(self):
        return self.num_samples


def bce_dice_loss(logits, target, valid):
    """0.5 BCEWithLogits + 0.5 soft Dice; padding excluded via valid mask."""
    v = valid
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce = (bce * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logits) * v
    t = target * v
    inter = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return 0.5 * bce + 0.5 * dice, {'bce': float(bce.detach()), 'dice': float(dice.detach())}


@torch.no_grad()
def eval_dev(model, loader, device, amp=True):
    model.eval()
    tot_loss, nb = 0.0, 0
    dices, ious, empty_pred = [], [], 0
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=amp):
            logits = model(img)
            loss, _ = bce_dice_loss(logits, mask, valid)
        tot_loss += float(loss); nb += 1
        p = (torch.sigmoid(logits) > 0.5).float()
        for i in range(p.shape[0]):
            pv, tv = p[i] * valid[i], mask[i] * valid[i]
            inter = (pv * tv).sum().item()
            ps, ts = pv.sum().item(), tv.sum().item()
            if ps == 0:
                empty_pred += 1
            dices.append((2 * inter + 1e-6) / (ps + ts + 1e-6) if ts > 0 else float('nan'))
            ious.append(inter / (ps + ts - inter + 1e-6) if (ps + ts - inter) > 0 else 0.0)
    model.train()
    return {'dev_loss': tot_loss / max(nb, 1),
            'dice': float(np.nanmean(dices)), 'iou': float(np.mean(ious)),
            'empty_pred_rate': empty_pred / max(len(dices), 1)}


def sha256_file(path, n=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(n), b''):
            h.update(chunk)
    return h.hexdigest()


def restore_training_state(ck, model, opt, scaler, sched, device):
    """Restore epoch-boundary state after model/loader construction."""
    import random
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
    scaler.load_state_dict(ck['scaler']); sched.load_state_dict(ck['scheduler'])
    torch.set_rng_state(ck['cpu_rng'].cpu())
    np.random.set_state(ck['np_rng'])
    if 'python_rng' in ck:
        random.setstate(ck['python_rng'])
    if device == 'cuda' and ck.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(ck['cuda_rng'].cpu())


def save_checkpoint_atomic(ck, path):
    tmp = path + '.tmp'
    torch.save(ck, tmp)
    os.replace(tmp, path)


def _code_hashes():
    import hashlib
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = {}
    for rel in ['src/umud_baseline/data.py', 'src/umud_baseline/train.py',
                'src/umud_baseline/evaluate.py', 'src/umud_baseline/predict.py',
                'src/umud_baseline/audit.py']:
        p = os.path.join(base, rel)
        out[rel] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return out


def main():
    import uuid
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['apo', 'fasc'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', default='full', choices=['tiny', 'pilot', 'full'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--init_state', default=None,
                    help='load this shared initial state_dict (same-init arms)')
    ap.add_argument('--tiny_groups', type=int, default=12)
    ap.add_argument('--dev-include', default=None,
                    help='optional isolation JSON with included_filenames')
    args = ap.parse_args()

    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    if args.resume and args.init_state:
        raise ValueError('Choose resume or init_state, not both')

    cfg = json.load(open(args.config, encoding='utf-8'))
    seed = cfg.get('seed', 20261017)
    dev_include_ref = args.dev_include or cfg.get('dev_include_view')
    dev_include_path = None
    include_filenames = None
    dev_include_sha = None
    if dev_include_ref:
        dev_include_path = dev_include_ref if os.path.isabs(dev_include_ref) else os.path.join(ROOT, dev_include_ref)
        include_doc = json.load(open(dev_include_path, encoding='utf-8'))
        include_filenames = list(include_doc['included_filenames'])
        dev_include_sha = sha256_file(dev_include_path)
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    log_path = os.path.join(args.out, 'logs', f'train_log_{run_id}.jsonl')

    data = UmudTrainData(args.task, 'train')
    n_groups = len(set(s['group_id'] for s in data.samples))
    print(f'[{args.task}] train unique images: {len(data.samples)} from {n_groups} groups')

    if args.mode == 'tiny':
        seen, keep = set(), []
        for i, s in enumerate(data.samples):
            if s['group_id'] not in seen:
                seen.add(s['group_id']); keep.append(i)
            if len(keep) >= args.tiny_groups:
                break
        data.samples = [data.samples[i] for i in keep]
        cfg['epochs'] = cfg.get('tiny_epochs', 60)
        cfg['augment'] = False
        print(f'tiny-overfit: {len(data.samples)} images / {len(seen)} groups')

    ds = UmudSegDataset(data, long_side=cfg.get('long_side', 512),
                        augment=cfg.get('augment', False) and args.mode != 'tiny',
                        base_seed=seed)
    sampler = FixedWeightedSampler(data.sampler_weights(), num_samples=len(ds), seed=seed)
    loader = DataLoader(ds, batch_size=cfg.get('batch_size', 4), sampler=sampler,
                        num_workers=cfg.get('num_workers', 2), drop_last=True,
                        pin_memory=True, persistent_workers=False)

    dev_data = UmudTrainData(args.task, 'dev', include_filenames=include_filenames)
    if include_filenames is not None:
        print(f'[{args.task}] isolated dev: {len(dev_data.samples)} images / '
              f'{len(set(s["group_id"] for s in dev_data.samples))} groups')
    dev_ds = UmudSegDataset(dev_data, long_side=cfg.get('long_side', 512), augment=False,
                            base_seed=seed)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.get('batch_size', 4), shuffle=False,
                            num_workers=1, pin_memory=True)

    from .evaluate import build_dev_cache, evaluate_model, PROTOCOL_VERSION
    dev_cache = (build_dev_cache(args.task, include_filenames=include_filenames)
                 if args.mode != 'tiny' else None)
    if include_filenames is not None and args.mode != 'tiny':
        if len(dev_data.samples) != 282:
            raise ValueError(f'isolated dev must have 282 images, got {len(dev_data.samples)}')
        if len(set(s['group_id'] for s in dev_data.samples)) != 279:
            raise ValueError('isolated dev must have 279 groups')
        if len(dev_cache) != len(dev_data.samples):
            raise ValueError('dev cache and isolated dev dataset lengths differ')

    architecture = cfg.get('architecture', 'resnet34_unet')
    encoder_name = cfg.get('encoder_name')
    encoder_weights = cfg.get('encoder_weights', 'imagenet')
    model_kwargs = {'pretrained': not (args.resume or args.init_state),
                    'architecture': architecture,
                    'encoder_name': encoder_name,
                    'encoder_weights': encoder_weights}
    model = build_model(**model_kwargs).to(device)
    if args.init_state:
        model.load_state_dict(torch.load(args.init_state, map_location=device,
                                         weights_only=False))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 3e-4),
                            weight_decay=cfg.get('weight_decay', 1e-4))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', True))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                       patience=cfg.get('lr_patience', 4))
    start_epoch, best_dice = 0, float('-inf')
    patience = cfg.get('patience', 10)   # = consecutive epoch-evaluations w/o improvement
    patience_left = patience
    resume_notes = []

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key, value in ck['config'].items():
            if key != 'epochs' and cfg.get(key) != value:
                raise ValueError(f'Resume config changed: {key}')
        split_hash = sha256_file(os.path.join(ROOT, 'manifest/splits_v2.csv'))
        if ck.get('splits_sha256') != split_hash:
            raise ValueError('Checkpoint split hash does not match frozen input')
        if ck.get('protocol') != PROTOCOL_VERSION:
            raise ValueError('Checkpoint evaluation protocol differs')
        restore_training_state(ck, model, opt, scaler, sched, device)
        start_epoch = ck['epoch'] + 1
        if 'best_dice' in ck:
            best_dice = ck['best_dice']; patience_left = ck.get('patience_left', patience)
        else:
            # OLD-format checkpoint: 'best' was a canvas dev LOSS and patience/
            # CUDA-RNG states were never saved -> exact resume NOT possible.
            resume_notes.append(
                'old-format ckpt: best was dev-loss (not comparable to group Dice); '
                'patience_left/CUDA RNG missing; best_dice reset to -inf')
        if device == 'cuda':
            if 'cuda_rng' in ck:
                torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
            else:
                resume_notes.append('cuda_rng missing in ckpt; CUDA RNG reseeded by torch.manual_seed')
        print(f'resumed from {args.resume} at epoch {start_epoch}')
        for n in resume_notes:
            print('WARNING:', n)
        # A continuation may never improve: preserve the parent's best artifact.
        import shutil
        parent_best = os.path.join(os.path.dirname(args.resume), 'best.pt')
        dest_best = os.path.join(args.out, 'checkpoints', 'best.pt')
        if not os.path.isfile(parent_best):
            raise FileNotFoundError('Resume requires the companion best.pt')
        if os.path.abspath(parent_best) != os.path.abspath(dest_best):
            if os.path.exists(dest_best):
                raise FileExistsError('Use a fresh output directory for continuation')
            parent = torch.load(parent_best, map_location='cpu', weights_only=False)
            if parent.get('best_dice') != best_dice:
                raise ValueError('Parent best.pt is inconsistent with last.pt')
            del parent
            shutil.copy2(parent_best, dest_best)

    resolved_encoder_name = encoder_name or ('mit_b0' if architecture == 'segformer_b0' else 'resnet34')
    try:
        encoder_info = encoder_source_info(resolved_encoder_name, encoder_weights or 'imagenet')
    except Exception as exc:
        encoder_info = {'encoder_name': resolved_encoder_name,
                        'weights': encoder_weights, 'source_error': str(exc)}
    run_meta = {'run_id': run_id, 'task': args.task, 'mode': args.mode, 'config': cfg,
                'architecture': architecture, 'encoder_name': resolved_encoder_name,
                'encoder_weights': encoder_weights, 'encoder_source': encoder_info,
                'dev_include': ({'path': os.path.relpath(dev_include_path, ROOT),
                                 'sha256': dev_include_sha,
                                 'n_images': len(dev_data.samples),
                                 'n_groups': len(set(s['group_id'] for s in dev_data.samples))}
                                if include_filenames is not None else None),
                'sampler': {'type': 'FixedWeightedSampler', 'seed': seed,
                            'n_samples_per_epoch': len(ds)},
                'config_sha256': sha256_file(args.config),
                'splits_sha256': sha256_file(os.path.join(ROOT, 'manifest/splits_v2.csv')),
                'manifest_sha256': sha256_file(os.path.join(ROOT, 'manifest/manifest.csv')),
                'code_hashes': _code_hashes(),
                'protocol': PROTOCOL_VERSION,
                'selection_metric': 'orig_group_dice',
                'seed': seed, 'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'init_state': args.init_state,
                'resume_from': args.resume, 'resume_notes': resume_notes,
                'resume_sha256': sha256_file(args.resume) if args.resume else None,
                'init_sha256': sha256_file(args.init_state) if args.init_state else None}
    with open(os.path.join(args.out, 'run_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, f'run_meta_{run_id}.json'), 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=1, ensure_ascii=False)

    eval_every = cfg.get('eval_every', 1)
    accum = cfg.get('grad_accum', 1)   # optimizer steps every `accum` micro-batches
    for epoch in range(start_epoch, cfg['epochs']):
        if patience_left <= 0:
            print('Restored checkpoint has already early-stopped'); break
        ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            img = batch['image'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            valid = batch['valid'].to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=cfg.get('amp', True)):
                loss, parts = bce_dice_loss(model(img), mask, valid)
            (scaler.scale(loss) / accum).backward()
            if (nb + 1) % accum == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += float(loss); nb += 1
        if nb % accum != 0:   # trailing partial accumulation block
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        do_eval = (epoch % eval_every == 0) or (epoch == cfg['epochs'] - 1)
        if do_eval:
            dev = eval_dev(model, dev_loader, device, cfg.get('amp', True))
            sched.step(dev['dev_loss'])
            if dev_cache is not None:
                proto, _, _ = evaluate_model(model, args.task, device,
                                             cfg.get('long_side', 512),
                                             cache=dev_cache, amp=cfg.get('amp', True),
                                             dev_data=dev_data)
                gd = proto['dice_group']
            else:
                gd = float('nan')
        else:
            dev = {'dev_loss': float('nan'), 'dice': float('nan'), 'iou': float('nan'),
                   'empty_pred_rate': float('nan')}
            gd = float('nan')
        # selection + early stop on ORIG-group Dice (maximize); patience counts
        # consecutive evaluated epochs without improvement
        improved = do_eval and (gd > best_dice)
        if improved:
            best_dice = gd; patience_left = patience
        elif do_eval:
            patience_left -= 1
        import hashlib
        seq = sampler.last_indices
        seq_sha = hashlib.sha256(json.dumps(seq, separators=(',', ':')).encode('utf-8')).hexdigest() if seq is not None else None
        sampling_dir = os.path.join(args.out, 'sampling')
        os.makedirs(sampling_dir, exist_ok=True)
        with open(os.path.join(sampling_dir, f'epoch_{epoch:03d}.json'), 'w', encoding='utf-8') as f:
            json.dump({'epoch': epoch, 'seed': seed, 'n': len(seq) if seq is not None else 0,
                       'indices': seq, 'sha256': seq_sha}, f, separators=(',', ':'))
        rec = {'run_id': run_id, 'epoch': epoch, 'train_loss': tot / max(nb, 1),
               'sampler_sha256': seq_sha, 'sampler_n': len(seq) if seq is not None else 0,
               **dev, 'orig_group_dice': gd,
               'patience_left': patience_left if do_eval else None,
               'lr': opt.param_groups[0]['lr'], 'sec': round(time.time() - t0, 1)}
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)
        # ckpt built AFTER selection update -> 'best_dice' never lags
        import random
        ckpt = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'scaler': scaler.state_dict(), 'scheduler': sched.state_dict(),
                'epoch': epoch, 'best_dice': best_dice, 'patience_left': patience_left,
                'cpu_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state() if device == 'cuda' else None,
                'np_rng': np.random.get_state(), 'config': cfg,
                'python_rng': random.getstate(),
                'run_id': run_id, 'protocol': PROTOCOL_VERSION,
                'config_sha256': run_meta['config_sha256'],
                'splits_sha256': run_meta['splits_sha256'],
                'architecture': architecture, 'encoder_name': encoder_name,
                'encoder_weights': encoder_weights}
        save_checkpoint_atomic(ckpt, os.path.join(args.out, 'checkpoints', 'last.pt'))
        if improved:
            save_checkpoint_atomic(ckpt, os.path.join(args.out, 'checkpoints', 'best.pt'))
        if do_eval and patience_left <= 0:
            print(f'early stop at epoch {epoch}')
            break

    if args.mode == 'tiny':
        # stage-B evidence: fit quality on the tiny TRAIN images + checkpoint
        # save/load prediction consistency
        model.eval()
        from .data import restore_original_coords, read_image_gray3
        od = os.path.join(args.out, 'tiny_overfit')
        os.makedirs(os.path.join(od, 'overlays'), exist_ok=True)
        rows = []
        state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for i in range(len(ds)):
            item = ds[i]
            x = item['image'].unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast('cuda'):
                pb = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            # PRIMARY stage-B metric: canvas-space dice vs the resized training
            # mask (what the model is actually trained on). Original-coords dice
            # is a diagnostic only: 1px fascicle lines lose pixels to nearest
            # downscale, capping it near ~0.5 even for a perfect canvas fit.
            cpred = (pb > 0.5)
            cgt = (item['mask'][0].numpy() > 0.5)
            cval = (item['valid'][0].numpy() > 0.5)
            pv, tv = cpred & cval, cgt & cval
            cinter = (pv & tv).sum()
            canvas_dice = float(2 * cinter / (pv.sum() + tv.sum() + 1e-9))
            prob = restore_original_coords(pb.astype(np.float32),
                                           tuple(item['orig_wh'].tolist()),
                                           float(item['scale']), tuple(item['pad'].tolist()))
            pred = (prob > 0.5).astype(np.uint8)
            s = data.samples[i]
            from .data import read_mask, align_mask_to_image
            gt = align_mask_to_image(read_mask(os.path.join(ROOT, s['masks'][0]['path'])),
                                     tuple(item['orig_wh'].tolist()))
            inter = ((pred == 1) & (gt == 1)).sum()
            dice = float(2 * inter / (pred.sum() + gt.sum() + 1e-9))
            rows.append({'filename': s['image']['filename'],
                         'canvas_dice': round(canvas_dice, 4),
                         'orig_dice': round(dice, 4)})
            if i < 6:
                rgb = read_image_gray3(os.path.join(ROOT, s['image']['path']))
                ov = rgb.copy()
                ov[gt == 1] = (0.6 * ov[gt == 1] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
                ov[pred == 1] = (0.6 * ov[pred == 1] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
                import PIL.Image as I
                I.fromarray(ov).save(os.path.join(od, 'overlays', s['image']['filename'].replace('.tif', '.png')))
        # save/load consistency
        torch.save(model.state_dict(), os.path.join(od, '_tmp_state.pt'))
        model2 = build_model().to(device)
        model2.load_state_dict(torch.load(os.path.join(od, '_tmp_state.pt'), map_location=device))
        model2.eval()
        import PIL.Image as I
        consistent = True
        for i in range(min(3, len(ds))):
            item = ds[i]
            x = item['image'].unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast('cuda'):
                p1 = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                p2 = torch.sigmoid(model2(x))[0, 0].cpu().numpy()
            consistent = consistent and bool(np.allclose(p1, p2, atol=1e-5))
        os.remove(os.path.join(od, '_tmp_state.pt'))
        with open(os.path.join(od, 'tiny_overfit_report.json'), 'w', encoding='utf-8') as f:
            json.dump({'n_train_images': len(rows),
                       'canvas_dice_mean': float(np.mean([r['canvas_dice'] for r in rows])),
                       'canvas_dice_min': float(np.min([r['canvas_dice'] for r in rows])),
                       'orig_dice_mean_diagnostic': float(np.mean([r['orig_dice'] for r in rows])),
                       'checkpoint_reload_consistent': consistent,
                       'per_image': rows}, f, indent=1)
        print('tiny_overfit:', json.dumps({'canvas_dice_mean': float(np.mean([r['canvas_dice'] for r in rows])),
                                           'orig_dice_mean_diagnostic': float(np.mean([r['orig_dice'] for r in rows])),
                                           'reload_consistent': consistent}))
    print('done. best orig_group_dice:', best_dice)
    with open(os.path.join(args.out, 'completion.json'), 'w', encoding='utf-8') as f:
        json.dump({'run_id': run_id, 'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'best_dice': best_dice, 'patience_left': patience_left,
                   'stop_reason': 'early_stop' if patience_left <= 0 else 'epoch_limit'}, f, indent=2)


if __name__ == '__main__':
    main()
