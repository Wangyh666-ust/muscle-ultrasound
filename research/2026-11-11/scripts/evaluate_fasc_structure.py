"""Fascicle structure evaluation against human-drawn line masks.

Primary reference: raw evaluable components (no merging).
Secondary reference: collinear fragments grouped at 10 deg / 10 px / 50 px.
Aggregation: annotation variant -> image -> near-duplicate group.
Human masks are reference/diagnostic only; prediction extraction never sees them.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.umud_baseline.data import (  # noqa: E402
    UmudTrainData, read_mask, align_mask_to_image, read_image_gray3, ROOT as DATA_ROOT,
)
from scripts.fascicle_geometry import (  # noqa: E402
    extract_segments, build_tracks, match_segments, summarize_match, aggregate_variants,
)

PRED_DIR = ROOT / 'runs/stage2_fasc/ls1024/dev_pred/masks'
PRED_META = ROOT / 'runs/stage2_fasc/ls1024/dev_pred/eval_meta.json'
CKPT = ROOT / 'runs/stage2_fasc/ls1024/checkpoints/best.pt'
PRIMARY = {'perp_tol': 10.0, 'angle_tol': 10.0, 'overlap_min': 0.2}
TRACK_GROUP = {'angle_tol': 10.0, 'perp_tol': 10.0, 'gap_tol': 50.0}
SENSITIVITY = [
    ('strict', {'perp_tol': 5.0, 'angle_tol': 5.0, 'overlap_min': 0.2}),
    ('primary', PRIMARY),
    ('loose', {'perp_tol': 20.0, 'angle_tol': 15.0, 'overlap_min': 0.2}),
]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_coverage(pred, human, tol=2):
    """Empty prediction has coverage 0, not an undefined value.

    Precision is undefined when there are no predicted pixels, so it stays
    None and is reported with an explicit valid count; coverage remains in the
    fixed denominator as 0.
    """
    pred = np.asarray(pred, dtype=bool)
    human = np.asarray(human, dtype=bool)
    if human.sum() == 0:
        return None, None
    if pred.sum() == 0:
        return 0.0, None
    d_pred = ndimage.distance_transform_edt(~pred)
    d_human = ndimage.distance_transform_edt(~human)
    human_covered = float((d_pred[human] <= tol).mean())
    pred_precision = float((d_human[pred] <= tol).mean())
    return human_covered, pred_precision


def variant_metrics(pred_mask, human_mask):
    p_ext = extract_segments(pred_mask)
    h_ext = extract_segments(human_mask)
    out = {
        'n_human_segments_evaluable': len(h_ext['segments']),
        'n_pred_segments_evaluable': len(p_ext['segments']),
        'n_human_short': sum(r['category'] == 'short_segment' for r in h_ext['rejected']),
        'n_human_crossing_or_branched': sum(r['category'] == 'crossing_or_branched' for r in h_ext['rejected']),
        'n_pred_short': sum(r['category'] == 'short_segment' for r in p_ext['rejected']),
        'n_pred_crossing_or_branched': sum(r['category'] == 'crossing_or_branched' for r in p_ext['rejected']),
        'n_human_fragmented_segments': sum(s.get('fragmented_neighbor', False) for s in h_ext['segments']),
        'n_pred_fragmented_segments': sum(s.get('fragmented_neighbor', False) for s in p_ext['segments']),
        'n_components_human': h_ext['n_components'],
        'n_components_pred': p_ext['n_components'],
        'pred_pixel_count': int(np.asarray(pred_mask, dtype=bool).sum()),
        'human_pixel_count': int(np.asarray(human_mask, dtype=bool).sum()),
        'pred_empty': bool(np.asarray(pred_mask, dtype=bool).sum() == 0),
        'human_empty': bool(np.asarray(human_mask, dtype=bool).sum() == 0),
    }
    for name, tol in SENSITIVITY:
        m = match_segments(p_ext['segments'], h_ext['segments'], **tol)
        s = summarize_match(p_ext['segments'], h_ext['segments'], m)
        out[f'raw_{name}'] = s
    # Moderate collinear grouping, reported separately.
    ht = build_tracks(h_ext['segments'], **TRACK_GROUP)
    pt = build_tracks(p_ext['segments'], **TRACK_GROUP)
    out['n_human_tracks'] = len(ht)
    out['n_human_fragmented_tracks'] = sum(t['fragmented'] for t in ht)
    out['n_pred_tracks'] = len(pt)
    out['n_pred_fragmented_tracks'] = sum(t['fragmented'] for t in pt)
    m = match_segments(pt, ht, **PRIMARY)
    out['track_primary'] = summarize_match(pt, ht, m)
    hc, pp = pixel_coverage(pred_mask, human_mask, 2)
    out['human_pixel_coverage_2px'] = hc
    out['pred_pixel_precision_2px'] = pp
    return out


def safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/geometry_eval/fasc_structure_eval_v1')
    parser.add_argument('--pred-dir', default=str(PRED_DIR))
    parser.add_argument('--pred-meta', default=str(PRED_META))
    parser.add_argument('--checkpoint', default=str(CKPT))
    parser.add_argument('--expected-long-side', type=int, default=1024)
    args = parser.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    pred_dir = resolve(args.pred_dir)
    pred_meta = resolve(args.pred_meta)
    ckpt = resolve(args.checkpoint)
    out = resolve(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(pred_meta.read_text(encoding='utf-8'))
    expected_ckpt = str(ckpt.relative_to(ROOT)).replace('\\', '/')
    if str(meta.get('checkpoint', '')).replace('\\', '/') != expected_ckpt:
        raise ValueError(f'Unexpected fasc prediction checkpoint: {meta.get("checkpoint")}')
    if (meta.get('threshold') != 0.5 or meta.get('restore') != 'nearest'
            or meta.get('long_side') != args.expected_long_side):
        raise ValueError('Unexpected fasc prediction protocol')
    data = UmudTrainData('fasc', 'dev')

    per_variant = []
    per_image = []
    for sample in data.samples:
        ident = sample['image']['filename']
        image = read_image_gray3(Path(DATA_ROOT) / sample['image']['path'])
        h, w = image.shape[:2]
        pred = (np.asarray(Image.open(pred_dir / (Path(ident).stem + '.png')).convert('L')) > 127)
        if pred.shape != (h, w):
            raise ValueError(f'{ident}: prediction shape mismatch')
        variants = []
        for vi, mrec in enumerate(sample['masks']):
            hm = align_mask_to_image(read_mask(Path(DATA_ROOT) / mrec['path']), (w, h))
            vm = variant_metrics(pred, hm)
            vm.update({'filename': ident, 'variant_index': vi,
                       'mask_filename': mrec['filename'],
                       'group_id': sample['group_id'], 'width': w, 'height': h})
            variants.append(vm)
            per_variant.append(vm)
        image_row = {
            'filename': ident, 'group_id': sample['group_id'],
            'width': w, 'height': h, 'n_variants': len(variants),
        }
        for key in ('n_human_segments_evaluable', 'n_pred_segments_evaluable',
                    'n_human_short', 'n_human_crossing_or_branched',
                    'n_pred_short', 'n_pred_crossing_or_branched',
                    'n_human_fragmented_segments', 'n_pred_fragmented_segments',
                    'n_components_human', 'n_components_pred',
                    'n_human_tracks', 'n_human_fragmented_tracks',
                    'n_pred_tracks', 'n_pred_fragmented_tracks',
                    'human_pixel_coverage_2px', 'pred_pixel_precision_2px',
                    'pred_pixel_count', 'human_pixel_count',
                    'pred_empty', 'human_empty'):
            image_row[key] = safe_mean([v[key] for v in variants])
        for name, _ in SENSITIVITY:
            for k in ('coverage', 'failure_rate_human', 'position_perp_median_px',
                      'position_mid_median_px', 'direction_median_deg',
                      'extra_candidate_rate', 'n_matched', 'n_unmatched_human',
                      'n_unmatched_pred_extra_candidates'):
                image_row[f'raw_{name}_{k}'] = safe_mean(
                    [v[f'raw_{name}'][k] for v in variants])
            image_row[f'raw_{name}_n_matched_sum'] = int(sum(
                v[f'raw_{name}']['n_matched'] for v in variants))
            image_row[f'raw_{name}_n_direction_pairs_sum'] = int(sum(
                v[f'raw_{name}']['n_direction_pairs'] for v in variants))
            image_row[f'raw_{name}_n_variants_direction_defined'] = int(sum(
                bool(v[f'raw_{name}']['direction_defined']) for v in variants))
            image_row[f'raw_{name}_direction_defined'] = bool(any(
                v[f'raw_{name}']['direction_defined'] for v in variants))
        for k in ('coverage', 'failure_rate_human', 'position_perp_median_px',
                  'position_mid_median_px', 'direction_median_deg',
                  'extra_candidate_rate', 'n_matched'):
            image_row[f'track_primary_{k}'] = safe_mean(
                [v['track_primary'][k] for v in variants])
        image_row['track_primary_n_matched_sum'] = int(sum(
            v['track_primary']['n_matched'] for v in variants))
        image_row['track_primary_n_direction_pairs_sum'] = int(sum(
            v['track_primary']['n_direction_pairs'] for v in variants))
        image_row['track_primary_n_variants_direction_defined'] = int(sum(
            bool(v['track_primary']['direction_defined']) for v in variants))
        image_row['track_primary_direction_defined'] = bool(any(
            v['track_primary']['direction_defined'] for v in variants))
        per_image.append(image_row)

    # Group aggregation: group-equal mean of image metrics.
    groups = defaultdict(list)
    for r in per_image:
        groups[r['group_id']].append(r)
    group_rows = []
    for gid, rows in sorted(groups.items()):
        g = {'group_id': gid, 'n_images': len(rows)}
        for key in per_image[0]:
            if key in ('filename', 'group_id', 'width', 'height', 'n_variants'):
                continue
            g[key] = safe_mean([r[key] for r in rows])
        # A group has a defined direction metric if any constituent image has
        # at least one matched predicted->human pair.
        g['raw_primary_direction_defined'] = bool(any(
            r['raw_primary_direction_defined'] for r in rows))
        g['track_primary_direction_defined'] = bool(any(
            r['track_primary_direction_defined'] for r in rows))
        group_rows.append(g)
    metric_keys = [k for k in group_rows[0] if k not in ('group_id', 'n_images')]
    summary = {
        'n_images': len(per_image),
        'n_groups': len(group_rows),
        'n_mask_variants': len(per_variant),
        'n_images_multi_variant': sum(r['n_variants'] > 1 for r in per_image),
        'aggregation': 'variant mean -> image; image mean -> group; group-equal mean reported',
        'primary_matching': PRIMARY,
        'track_grouping': TRACK_GROUP,
        'sensitivity_matching': {name: tol for name, tol in SENSITIVITY},
        'direction_metric': {
            'matched_pair_condition': PRIMARY,
            'matched_pair_definition': ('predicted midpoint to human infinite line <= perp_tol, '
                                        'acute direction difference <= angle_tol, overlap along '
                                        'human direction >= overlap_min'),
            'per_variant_aggregation': 'median direction difference over matched pairs; undefined if no pair',
            'image_aggregation': 'mean of variant medians over variants with >=1 matched pair',
            'group_aggregation': 'mean of image medians over images with a defined direction',
            'n_variants_total': len(per_variant),
            'n_variants_with_direction': int(sum(
                v['raw_primary']['direction_defined'] for v in per_variant)),
            'n_matched_pairs_total': int(sum(
                v['raw_primary']['n_direction_pairs'] for v in per_variant)),
            'n_images_total': len(per_image),
            'n_images_with_direction': int(sum(
                bool(r['raw_primary_direction_defined']) for r in per_image)),
            'n_groups_total': len(group_rows),
            'n_groups_with_direction': int(sum(
                bool(g['raw_primary_direction_defined']) for g in group_rows)),
        },
        'pixel_coverage_summary': {
            'empty_prediction_rule': 'if prediction mask has no pixels, human coverage is 0.0 and prediction precision is undefined (None)',
            'n_variants_pred_empty': int(sum(v['pred_empty'] for v in per_variant)),
            'n_variants_human_empty': int(sum(v['human_empty'] for v in per_variant)),
            'n_variants_human_coverage_defined': int(sum(
                not v['human_empty'] for v in per_variant)),
            'n_variants_pred_precision_defined': int(sum(
                (not v['pred_empty']) and (not v['human_empty']) for v in per_variant)),
        },
        'image_level': {k: safe_mean([r[k] for r in per_image]) for k in metric_keys},
        'group_equal': {k: safe_mean([r[k] for r in group_rows]) for k in metric_keys},
        'meaning': ('Engineering/structure agreement with human line masks, not PA/FL parameter accuracy. '
                    'Human masks are reference only; prediction extraction never uses them. '
                    'Unmatched predicted segments are extra_candidates, not automatically false positives. '
                    'Direction metrics are conditional on matched predicted->human pairs; valid counts '
                    'are reported explicitly, and empty predictions use coverage 0 with undefined precision.'),
    }
    (out / 'per_variant.json').write_text(json.dumps(per_variant, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'per_image.json').write_text(json.dumps(per_image, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'per_group.json').write_text(json.dumps(group_rows, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    with open(out / 'per_image.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader(); w.writerows(per_image)
    with open(out / 'per_group.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        w.writeheader(); w.writerows(group_rows)

    # Representative overlays: lowest/highest coverage, highest extra, most fragmented.
    candidates = sorted([r for r in per_image if (r['raw_primary_coverage'] or 0) > 0],
                        key=lambda r: r['raw_primary_coverage'])
    picks = []
    picks += candidates[:3]
    picks += candidates[-3:]
    picks += sorted(per_image, key=lambda r: -(r['raw_primary_extra_candidate_rate'] or 0))[:3]
    picks += sorted(per_image, key=lambda r: -(r['n_human_fragmented_segments'] or 0))[:3]
    seen = set(); picks = [r for r in picks if not (r['filename'] in seen or seen.add(r['filename']))]
    rep = out / 'representatives'; rep.mkdir(exist_ok=True)
    font = ImageFont.truetype('arial.ttf', 14)
    paths = []
    for r in picks[:12]:
        sample = next(s for s in data.samples if s['image']['filename'] == r['filename'])
        image = read_image_gray3(Path(DATA_ROOT) / sample['image']['path'])
        h, w = image.shape[:2]
        pred = (np.asarray(Image.open(pred_dir / (Path(r['filename']).stem + '.png')).convert('L')) > 127)
        hm = align_mask_to_image(read_mask(Path(DATA_ROOT) / sample['masks'][0]['path']), (w, h))
        pe = extract_segments(pred); he = extract_segments(hm)
        # Match primary on raw segments for colouring.
        pm = match_segments(pe['segments'], he['segments'], **PRIMARY)
        matched_h = {m['human_index']: m['pred_index'] for m in pm['matches']}
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        for i, s in enumerate(he['segments']):
            col = (0, 255, 0) if i not in matched_h else (255, 255, 0)
            draw.line([tuple(s['p0']), tuple(s['p1'])], fill=col, width=2)
        for i, s in enumerate(pe['segments']):
            col = (255, 0, 0)
            draw.line([tuple(s['p0']), tuple(s['p1'])], fill=col, width=2)
        header = 60
        canvas = Image.new('RGB', (w, h + header), 'white'); canvas.paste(pil, (0, header))
        d = ImageDraw.Draw(canvas)
        lines = [
            f"{r['filename']} | variants={r['n_variants']} | human segments={r['n_human_segments_evaluable']:.1f} pred={r['n_pred_segments_evaluable']:.1f}",
            f"coverage={r['raw_primary_coverage']:.3f} extra_rate={r['raw_primary_extra_candidate_rate']:.3f} dir_med={r['raw_primary_direction_median_deg']:.2f} deg perp_med={r['raw_primary_position_perp_median_px']:.2f} px",
            "green=human, red=pred, yellow=matched human segment",
        ]
        for i, line in enumerate(lines):
            d.text((6, 4 + i * 18), line, fill='black', font=font)
        p = rep / f"{Path(r['filename']).stem}.png"; canvas.save(p); paths.append(p)
    if paths:
        ims = [Image.open(p).convert('RGB') for p in paths]
        cw = 600; ims = [im.resize((cw, int(im.height * cw / im.width)), Image.LANCZOS) for im in ims]
        ch = max(im.height for im in ims); cols = 2; rows = (len(ims) + 1) // 2
        sheet = Image.new('RGB', (cols * cw, rows * ch), 'white')
        for i, im in enumerate(ims): sheet.paste(im, ((i % 2) * cw, (i // 2) * ch))
        sheet.save(out / 'fasc_structure_representatives.png')

    verification = {
        'status': 'passed',
        'model_rerun': False,
        'threshold_changed': False,
        'human_masks_used_in_prediction_extraction': False,
        'holdout_read': False,
        'n_images': len(per_image),
        'n_groups': len(group_rows),
        'n_variants': len(per_variant),
        'prediction_dir': str(pred_dir.relative_to(ROOT)),
        'n_prediction_masks': len(list(pred_dir.glob('*.png'))),
        'fascicle_geometry_sha256': sha256_file(ROOT / 'scripts/fascicle_geometry.py'),
        'checkpoint': meta['checkpoint'],
        'checkpoint_sha256': sha256_file(ckpt),
        'checkpoint_epoch': meta.get('ckpt_epoch'),
        'prediction_protocol': {k: meta.get(k) for k in
                                ('protocol', 'threshold', 'restore', 'prob_dtype', 'long_side')},
        'splits_sha256': sha256_file(ROOT / 'manifest/splits_v2.csv'),
        'manifest_sha256': sha256_file(ROOT / 'manifest/manifest.csv'),
        'script_sha256': sha256_file(Path(__file__)),
    }
    (out / 'verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False),
                                           encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
