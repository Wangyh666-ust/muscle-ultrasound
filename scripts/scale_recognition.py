"""Independent ultrasound image scale recognition for Round16.

The reader accepts only:
- the original image;
- optional explicit spatial calibration metadata bound to the image identity;
- the image identity (id, size, grayscale pixel hash).

It never receives expert PA/FL/MT values, model measurements, reference scales,
sample-submission numeric columns or physiological ranges.

Derived rules (all recorded in the output evidence):
1. Explicit spatial calibration metadata with a unit and matching image identity.
2. Numeric labels attached to a vertical image ruler plus a unit source.
3. An explicit depth label with a unit plus the vertical scan-region span.

If the evidence is insufficient or conflicts, the status is unknown or
ambiguous and px_per_mm is null. "ok" means traceable unit + pixel distance,
not expert-confirmed accuracy.
"""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODULE_VERSION = 'round16_2_scale_reader_v1'
CONFIG = {
    'version': MODULE_VERSION,
    'ocr_engine': 'rapidocr_onnxruntime',
    'ocr_min_score': 0.45,
    'ocr_orientations_fast': ['orig', 'mirror'],
    'ocr_orientations_fallback': ['rot180', 'rot180_mirror'],
    'ocr_coordinate_transform_version': 'round16_1_coord_fix_v1',
    'ruler_line_min_fraction': 0.30,
    'ruler_tick_threshold': 80,
    'ruler_tick_min_score': 2,
    'ruler_min_ticks': 3,
    'ruler_label_max_distance_fraction': 0.06,
    'ruler_label_max_distance_px': 60,
    'ruler_label_min_score': 0.80,
    'ruler_reread_upscale': 3,
    'ruler_reread_half_width_px': 180,
    'zero_reread_upscale': 4,
    'zero_reread_half_width_px': 100,
    'zero_reread_height_fraction': 0.20,
    'bottom_reread_upscale': 4,
    'bottom_reread_height_fraction': 0.25,
    'scan_threshold': 8,
    'scan_column_min_fraction': 0.30,
    'scan_row_min_fraction': 0.12,
    'scan_core_fraction': 0.50,
    'scan_expand_gap_fraction': 0.015,
    'scan_min_span_fraction': 0.15,
    'unit_consistency_max_relative_diff': 0.20,
    'mode_consistency_max_relative_diff': 0.05,
    'endpoint_bottom_fraction': 0.15,
    'endpoint_top_fraction': 0.15,
    'endpoint_margin_fraction': 0.10,
    'depth_span_endpoint_max_offset_fraction': 0.10,
    'metadata_required_identity_fields': [
        'image_id', 'pixel_sha256', 'width', 'height'],
    'evidence_max_width': 1100,
    'scale_formula': {
        'numeric_ticks': 'abs(y2-y1) / abs(value2_mm-value1_mm)',
        'depth_span': ('span_height_px / '
                       'abs(bottom_endpoint_mm - top_endpoint_mm)'),
    },
    'pa_unit': 'degree',
}

CONFIG_SHA256 = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True, separators=(',', ':'),
               ensure_ascii=False).encode('utf-8')).hexdigest()
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_OCR = None


def ocr_engine_version():
    try:
        from importlib.metadata import version
        return version('rapidocr_onnxruntime')
    except Exception:
        return None


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_identity(path):
    """Same identity definition as the measurement entry: L-converted bytes."""
    with Image.open(path) as im:
        w, h = im.size
        gray = im.convert('L')
        digest = hashlib.sha256(gray.tobytes()).hexdigest()
    return {'width': int(w), 'height': int(h), 'pixel_sha256': digest}


def validate_identity(actual, expected):
    if expected is None:
        return True, []
    bad = []
    if actual.get('pixel_sha256') != expected.get('pixel_sha256'):
        bad.append('pixel_sha256')
    if int(actual.get('width')) != int(expected.get('width')):
        bad.append('width')
    if int(actual.get('height')) != int(expected.get('height')):
        bad.append('height')
    return not bad, bad


def validate_scale(value):
    if value is None:
        return False, 'missing'
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, 'not_numeric'
    if not math.isfinite(v):
        return False, 'non_finite'
    if v <= 0:
        return False, 'not_positive'
    return True, ''


def _get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def _map_box(box, width, height, orientation):
    pts = np.asarray(box, dtype=float)
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    # Forward image transforms:
    # mirror: x -> W-1-x
    # rot180: x -> W-1-x, y -> H-1-y
    # rot180_mirror: y -> H-1-y only
    if orientation in ('mirror', 'rot180'):
        x = (width - 1) - x
    if orientation in ('rot180', 'rot180_mirror'):
        y = (height - 1) - y
    return [[float(xi), float(yi)] for xi, yi in zip(x, y)]


def run_ocr_region(image, region, upscale=3):
    """OCR an image crop and map boxes back to original coordinates."""
    x0, y0, x1, y1 = region
    crop = image.crop((x0, y0, x1, y1))
    if crop.width <= 0 or crop.height <= 0:
        return []
    cw, ch = crop.size
    crop = crop.resize((cw * upscale, ch * upscale), Image.LANCZOS)
    out = []
    engine = _get_ocr()
    for orientation in ('orig', 'mirror'):
        img = crop if orientation == 'orig' else ImageOps.mirror(crop)
        try:
            result, _ = engine(np.asarray(img))
        except Exception:
            continue
        for box, text, score in (result or []):
            score = float(score)
            if score < CONFIG['ocr_min_score']:
                continue
            pts = np.asarray(box, dtype=float)
            xs = pts[:, 0].copy()
            ys = pts[:, 1].copy()
            if orientation == 'mirror':
                xs = (cw * upscale - 1) - xs
            xs = x0 + xs / upscale
            ys = y0 + ys / upscale
            out.append({
                'orientation': f'crop_{orientation}',
                'text': str(text), 'score': score,
                'box': [[float(x), float(y)] for x, y in zip(xs, ys)],
                'xc': float(np.mean(xs)), 'yc': float(np.mean(ys)),
            })
    return out


def supplemental_ocr(image, existing):
    """Try bottom/top/right crops when whole-image OCR found no unit text."""
    w, h = image.size
    regions = [
        ('bottom', (0, int(0.68 * h), w, h)),
        ('top_left', (0, 0, int(0.40 * w), int(0.40 * h))),
        ('top_right', (int(0.60 * w), 0, w, int(0.40 * h))),
        ('right', (int(0.72 * w), 0, w, h)),
    ]
    added = []
    for _, region in regions:
        items = run_ocr_region(image, region)
        added.extend(items)
        combined = existing + added
        if any(parse_number_unit(i['text']) for i in combined):
            break
    return added


def run_ocr(image, orientations):
    engine = _get_ocr()
    out = []
    w, h = image.size
    for orientation in orientations:
        if orientation == 'orig':
            img = image
        elif orientation == 'mirror':
            img = ImageOps.mirror(image)
        elif orientation == 'rot180':
            img = image.transpose(Image.ROTATE_180)
        elif orientation == 'rot180_mirror':
            img = ImageOps.mirror(image.transpose(Image.ROTATE_180))
        else:
            raise ValueError(orientation)
        try:
            result, _ = engine(np.asarray(img))
        except Exception:
            continue
        for box, text, score in (result or []):
            score = float(score)
            if score < CONFIG['ocr_min_score']:
                continue
            pts = _map_box(box, w, h, orientation)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            out.append({
                'orientation': orientation,
                'text': str(text),
                'score': score,
                'box': pts,
                'xc': float(np.mean(xs)),
                'yc': float(np.mean(ys)),
            })
    # Deduplicate the same text/box seen in two orientations.
    seen = set()
    deduped = []
    for item in sorted(out, key=lambda r: -r['score']):
        key = (item['text'], round(item['xc'] / 3), round(item['yc'] / 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


_NUM_UNIT = re.compile(
    r'(?<![\w.])(\d+(?:[.,]\d+)?)\s*(mm|cm)\b', re.IGNORECASE)
_NUM_ONLY = re.compile(r'^\s*(\d+(?:[.,]\d+)?)\s*$')
_RULER_UNIT_LABEL = re.compile(
    r'^\s*(\d+(?:[.,]\d+)?)\s*(mm|cm)\s*$', re.IGNORECASE)
_BAD_UNIT_CONTEXT = ('mhz', 'khz', 'hz', 'db', 'fps', '%')


def parse_number_unit(text):
    found = []
    # OCR sometimes prefixes a clipped tick with "_" or another punctuation
    # character; remove leading non-digit characters before matching.
    text = re.sub(r'^[^0-9]+', '', str(text))
    for m in _NUM_UNIT.finditer(text):
        low = text.lower()
        if any(bad in low for bad in _BAD_UNIT_CONTEXT):
            continue
        value = float(m.group(1).replace(',', '.'))
        unit = m.group(2).lower()
        found.append({'value': value, 'unit': unit,
                      'value_mm': value * (10.0 if unit == 'cm' else 1.0),
                      'matched': m.group(0)})
    return found


def parse_number_only(text):
    m = _NUM_ONLY.match(text)
    if not m:
        return None
    return float(m.group(1).replace(',', '.'))


def _longest_run(mask):
    best_len = 0
    best_start = 0
    cur = 0
    start = 0
    for i, value in enumerate(mask):
        if value:
            if cur == 0:
                start = i
            cur += 1
            if cur > best_len:
                best_len = cur
                best_start = start
        else:
            cur = 0
    return int(best_len), int(best_start)


def _merged_runs(mask, max_gap):
    runs = []
    start = None
    prev = None
    for i, value in enumerate(mask):
        if value:
            if start is None:
                start = i
            prev = i
        else:
            if start is not None:
                runs.append([start, prev])
                start = None
    if start is not None:
        runs.append([start, prev])
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return merged


def _fit_regular_ticks(ticks, min_spacing=8):
    ticks = sorted(set(int(t) for t in ticks))
    if len(ticks) < CONFIG['ruler_min_ticks']:
        return None
    best = None
    for a in ticks:
        for b in ticks:
            if b <= a:
                continue
            for k in range(1, 11):
                spacing = (b - a) / k
                if spacing < min_spacing:
                    continue
                tol = max(3.0, 0.12 * spacing)
                inliers = []
                for t in ticks:
                    n = round((t - a) / spacing)
                    if n >= 0 and abs(t - (a + n * spacing)) <= tol:
                        inliers.append((n, t))
                by_n = {}
                for n, t in inliers:
                    by_n.setdefault(n, t)
                inliers = [(n, by_n[n]) for n in sorted(by_n)]
                if len(inliers) < CONFIG['ruler_min_ticks']:
                    continue
                span = inliers[-1][1] - inliers[0][1]
                grid_points = inliers[-1][0] - inliers[0][0] + 1
                coverage = len(inliers) / float(grid_points)
                if coverage < 0.65:
                    continue
                score = (len(inliers), span, coverage)
                if best is None or score > best['score']:
                    best = {
                        'score': score,
                        'ticks': [t for _, t in inliers],
                        'spacing_px': float(spacing),
                        'coverage': float(coverage),
                        'start_y': int(inliers[0][1]),
                        'end_y': int(inliers[-1][1]),
                    }
    if best is None:
        return None
    best.pop('score', None)
    return best


def detect_ruler(gray):
    """Find a vertical ruler line with a regular tick series."""
    h, w = gray.shape
    candidates = []
    for x in range(w):
        run, start = _longest_run(gray[:, x] > CONFIG['ruler_tick_threshold'])
        if run >= CONFIG['ruler_line_min_fraction'] * h:
            candidates.append((run, x, start))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    # Group adjacent columns, keep the longest representative.
    compact = []
    for run, x, start in candidates:
        if all(abs(x - cx) > 3 for _, cx, _ in compact):
            compact.append((run, x, start))
    best = None
    for run, x, start in compact[:8]:
        for side in (-1, 1):
            x0 = max(0, x + (2 if side > 0 else -18))
            x1 = min(w, x + (18 if side > 0 else -1))
            if x1 - x0 < 3:
                continue
            band = gray[:, x0:x1]
            score = (band > CONFIG['ruler_tick_threshold']).sum(axis=1)
            peaks = []
            for y in range(1, len(score) - 1):
                if (score[y] >= CONFIG['ruler_tick_min_score']
                        and score[y] >= score[y - 1]
                        and score[y] >= score[y + 1]):
                    peaks.append((y, int(score[y])))
            groups = []
            if peaks:
                s = prev = peaks[0][0]
                best_peak = peaks[0]
                for y, v in peaks[1:]:
                    if y - prev > 4:
                        groups.append(best_peak[0])
                        s = y
                        best_peak = (y, v)
                    elif v > best_peak[1]:
                        best_peak = (y, v)
                    prev = y
                groups.append(best_peak[0])
            series = _fit_regular_ticks(groups)
            if series is None:
                continue
            # Prefer more ticks, then larger span.
            rank = (len(series['ticks']), series['end_y'] - series['start_y'])
            if best is None or rank > best['rank']:
                best = {
                    'rank': rank,
                    'line_x': int(x),
                    'side': side,
                    'ticks': series['ticks'],
                    'spacing_px': series['spacing_px'],
                    'coverage': series['coverage'],
                    'start_y': series['start_y'],
                    'end_y': series['end_y'],
                    'line_run_px': int(run),
                }
    if best is not None:
        best.pop('rank', None)
    return best


def detect_scan_span(gray):
    """Bounding vertical scan region from background darkness."""
    h, w = gray.shape
    step = 2 if max(h, w) > 700 else 1
    small = gray[::step, ::step]
    sh, sw = small.shape
    th = CONFIG['scan_threshold']
    col_frac = (small > th).mean(axis=0)
    col_len, col_start = _longest_run(
        col_frac > CONFIG['scan_column_min_fraction'])
    if col_len < CONFIG['scan_min_span_fraction'] * sw:
        return None
    x0, x1 = col_start, col_start + col_len - 1
    row_frac = (small[:, x0:x1 + 1] > th).mean(axis=1)
    row_mask = row_frac > CONFIG['scan_row_min_fraction']
    core_mask = row_frac > CONFIG['scan_core_fraction']
    gap = max(4, int(CONFIG['scan_expand_gap_fraction'] * sh))
    core_merged = _merged_runs(core_mask, max_gap=gap)
    if core_merged:
        y0, y1 = max(core_merged, key=lambda r: r[1] - r[0])
    else:
        merged = _merged_runs(row_mask, max_gap=gap)
        if not merged:
            return None
        y0, y1 = max(merged, key=lambda r: r[1] - r[0])

    def expand(y, direction):
        while True:
            found = None
            for k in range(1, gap + 2):
                j = y + direction * k
                if j < 0 or j >= sh:
                    break
                if row_mask[j]:
                    found = j
                    break
            if found is None:
                return y
            y = found

    y0 = expand(y0, -1)
    y1 = expand(y1, 1)
    coverage = float(row_mask[y0:y1 + 1].mean()) if y1 >= y0 else 0.0
    if y1 - y0 + 1 < CONFIG['scan_min_span_fraction'] * sh:
        return None
    ox0, ox1 = int(x0 * step), int(x1 * step)
    oy0, oy1 = int(y0 * step), int(y1 * step)
    full_frame = (ox0 <= 1 and ox1 >= w - 2) or (oy0 <= 1 and oy1 >= h - 2)
    return {
        'x0': ox0, 'x1': ox1, 'y0': oy0, 'y1': oy1,
        'width_px': int((x1 - x0 + 1) * step),
        'height_px': int((y1 - y0 + 1) * step),
        'step': step,
        'row_coverage': float(coverage),
        'full_frame': bool(full_frame),
    }


def _depth_candidates(ocr_items, width, height):
    out = []
    for item in ocr_items:
        for parsed in parse_number_unit(item['text']):
            low = item['text'].lower()
            context = ''
            for key in ('tiefe', 'depth', 'de '):
                if key in low:
                    context = key.strip()
                    break
            # Standalone values are accepted only near an image edge.
            nx = item['xc'] / max(1.0, width)
            ny = item['yc'] / max(1.0, height)
            near_edge = nx < 0.28 or nx > 0.72 or ny > 0.72 or ny < 0.28
            if not context and not near_edge:
                continue
            out.append({
                'text': item['text'],
                'score': item['score'],
                'orientation': item['orientation'],
                'box': item['box'],
                'xc': item['xc'], 'yc': item['yc'],
                'value': parsed['value'], 'unit': parsed['unit'],
                'value_mm': parsed['value_mm'],
                'matched': parsed['matched'],
                'context': context or 'edge',
            })
    # Keep one observation per value/unit/location. Multiple locations with
    # the same value are valuable endpoint evidence (e.g. a depth label at the
    # bottom edge), so they must not be collapsed by value alone.
    best = {}
    for c in out:
        key = (round(c['value_mm'], 6), c['unit'],
               round(c['xc'] / 8), round(c['yc'] / 8))
        if key not in best or c['score'] > best[key]['score']:
            best[key] = c
    return sorted(best.values(), key=lambda c: -c['score'])


def _choose_depth(candidates):
    if not candidates:
        return None, 'no_depth_label_with_unit'
    values = [c['value_mm'] for c in candidates]
    if max(values) > 0 and (max(values) - min(values)) / max(values) > 0.05:
        return None, 'conflicting_depth_labels'
    high = [c for c in candidates if c['context'] in ('tiefe', 'depth', 'de')]
    pool = high or [c for c in candidates if c['context'] == 'edge']
    if not pool:
        return None, 'no_depth_label_with_unit'
    return max(pool, key=lambda c: c['score']), ''


def _choose_unit_for_labels(label_value, depth_candidates):
    """Choose a unit only when the depth label makes one interpretation consistent."""
    if not depth_candidates:
        return None, 'ruler_labels_have_no_unit'
    options = []
    for unit in ('mm', 'cm'):
        label_total_mm = label_value * (10.0 if unit == 'cm' else 1.0)
        for c in depth_candidates:
            denom = max(abs(c['value_mm']), abs(label_total_mm), 1e-9)
            diff = abs(c['value_mm'] - label_total_mm) / denom
            options.append((diff, unit, c))
    options.sort(key=lambda x: x[0])
    best_diff, best_unit, _ = options[0]
    other = [o for o in options if o[1] != best_unit]
    second_diff = min(o[0] for o in other) if other else float('inf')
    limit = CONFIG['unit_consistency_max_relative_diff']
    if best_diff <= limit and second_diff > limit:
        return best_unit, ''
    if best_diff <= limit and second_diff <= limit:
        return None, 'ambiguous_ruler_label_unit'
    return None, 'ruler_label_unit_inconsistent_with_depth'


def _labels_for_ruler(ocr_items, ruler, width, height, depth_candidates):
    line_x = ruler['line_x']
    max_dist = max(CONFIG['ruler_label_max_distance_px'],
                   CONFIG['ruler_label_max_distance_fraction'] * width)
    labels = []
    excluded = []
    for item in ocr_items:
        if item['score'] < CONFIG['ruler_label_min_score']:
            excluded.append({'text': item['text'], 'score': item['score'],
                             'orientation': item.get('orientation'),
                             'xc': item.get('xc'), 'yc': item.get('yc'),
                             'reason': 'low_ocr_score'})
            continue
        if abs(item['xc'] - line_x) > max_dist:
            excluded.append({'text': item['text'], 'score': item['score'],
                             'orientation': item.get('orientation'),
                             'xc': item.get('xc'), 'yc': item.get('yc'),
                             'reason': 'outside_ruler_label_band'})
            continue
        value = parse_number_only(item['text'])
        unit = None
        if value is None:
            # A unit-bearing ruler label must be only "value unit"; a depth
            # field such as "De 50 mm" is not a ruler tick label.
            m = _RULER_UNIT_LABEL.match(item['text'])
            if m:
                value = float(m.group(1).replace(',', '.'))
                unit = m.group(2).lower()
        if value is None:
            excluded.append({'text': item['text'], 'score': item['score'],
                             'orientation': item.get('orientation'),
                             'xc': item.get('xc'), 'yc': item.get('yc'),
                             'reason': 'not_a_pure_numeric_ruler_label'})
            continue
        nearest = min(ruler['ticks'], key=lambda t: abs(t - item['yc']))
        distance = abs(nearest - item['yc'])
        if distance > 0.45 * ruler['spacing_px']:
            excluded.append({'text': item['text'], 'score': item['score'],
                             'orientation': item.get('orientation'),
                             'xc': item.get('xc'), 'yc': item.get('yc'),
                             'reason': 'not_aligned_to_detected_tick'})
            continue
        labels.append({
            'text': item['text'], 'score': item['score'],
            'orientation': item['orientation'], 'box': item['box'],
            'xc': item['xc'], 'yc': item['yc'],
            'value': value, 'unit': unit, 'tick_y': int(nearest),
            'tick_distance_px': float(distance),
        })
    # Keep the best-scoring label per tick. Other same-tick labels are kept in
    # the exclusion audit; they are not silently used.
    by_tick = {}
    for lab in labels:
        key = lab['tick_y']
        if key not in by_tick:
            by_tick[key] = lab
        elif lab['score'] > by_tick[key]['score']:
            excluded.append({**by_tick[key], 'reason': 'same_tick_lower_score'})
            by_tick[key] = lab
        else:
            excluded.append({**lab, 'reason': 'same_tick_lower_score'})
    return sorted(by_tick.values(), key=lambda lab: lab['tick_y']), excluded


def _standard_interval_mm(value):
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0):
        if abs(value - candidate) <= 0.05 * candidate:
            return candidate
    return None


def _unit_for_ruler_labels(labels, ruler, depth_candidates, width):
    """Resolve explicit mixed units and unlabelled decimal interpretations.

    Explicit mm/cm labels are first converted label by label to millimetres.
    Compatible mixed units then remain usable; genuinely inconsistent values
    are reported as a structured ambiguity (not a string keyword check). A
    label without a unit is interpreted from ruler geometry only under the
    existing zero-anchored, standard-interval, depth-consistent rules.
    """
    labels_with_unit = [lab for lab in labels if lab.get('unit')]
    units = {lab['unit'] for lab in labels_with_unit}
    if labels_with_unit:
        unresolved = [lab for lab in labels
                      if lab.get('unit') is None
                      and abs(float(lab.get('value') or 0.0)) > 1e-9]
        if unresolved:
            return {
                'status': 'unknown',
                'reason': 'ruler_mixed_explicit_and_unproven_unit_labels',
                'unit': None, 'unit_source': None, 'unit_context': None,
                'values_mm': None,
            }
        values_mm = []
        for lab in labels:
            value = abs(float(lab.get('value') or 0.0))
            if lab.get('unit') == 'cm':
                value *= 10.0
            elif lab.get('unit') == 'mm':
                pass
            else:
                # A zero without a unit is the same physical value in mm/cm.
                value = 0.0
            values_mm.append(float(value))
        # Ruler values must not decrease as the tick moves down the image.
        ordered = sorted(zip([lab['tick_y'] for lab in labels], values_mm))
        ordered_values = [v for _, v in ordered]
        if any(b < a for a, b in zip(ordered_values, ordered_values[1:])):
            return {
                'status': 'ambiguous',
                'reason': 'ruler_label_values_not_increasing_with_depth',
                'unit': None, 'unit_source': None, 'unit_context': None,
                'values_mm': None, 'explicit_units': sorted(units),
            }
        return {
            'status': 'resolved',
            'reason': '',
            'unit': 'mm' if len(units) > 1 else next(iter(units)),
            'unit_source': 'explicit_units_normalized_to_mm',
            'unit_context': None,
            'values_mm': values_mm,
            'explicit_units': sorted(units),
        }

    if not depth_candidates:
        return {
            'status': 'unknown', 'reason': 'ruler_labels_have_no_unit',
            'unit': None, 'unit_source': None, 'unit_context': None,
            'values_mm': None,
        }
    depth, depth_reason = _choose_depth(depth_candidates)
    if depth is None:
        return {
            'status': ('ambiguous' if depth_reason == 'conflicting_depth_labels'
                       else 'unknown'),
            'reason': depth_reason,
            'unit': None, 'unit_source': None, 'unit_context': None,
            'values_mm': None,
        }
    tick_index = {t: i for i, t in enumerate(ruler['ticks'])}
    if any(lab['tick_y'] not in tick_index for lab in labels):
        return {
            'status': 'unknown', 'reason': 'ruler_label_not_on_detected_tick',
            'unit': None, 'unit_source': None, 'unit_context': None,
            'values_mm': None,
        }
    if len(labels) < 2:
        return {
            'status': 'unknown',
            'reason': 'single_label_needs_zero_or_second_endpoint',
            'unit': None, 'unit_source': None,
            'unit_context': depth['text'], 'values_mm': None,
        }
    indices = [tick_index[lab['tick_y']] for lab in labels]
    if len(set(indices)) < len(indices):
        return {
            'status': 'unknown', 'reason': 'ruler_labels_share_a_tick',
            'unit': None, 'unit_source': None,
            'unit_context': depth['text'], 'values_mm': None,
        }

    # Candidate physical values per label; explicit units never reach here.
    candidate_sets = []
    for lab in labels:
        v = abs(lab['value'])
        candidate_sets.append(sorted(set([v, v * 10.0])))
    options = []
    import itertools
    for unit in ('mm', 'cm'):
        for values in itertools.product(*candidate_sets):
            values = [float(v) for v in values]
            b, a = np.polyfit(indices, values, 1)
            if b <= 0:
                continue
            max_value = max(values)
            residual = max(abs(v - (a + b * i))
                           for v, i in zip(values, indices))
            if residual > max(0.02 * max_value, 0.5):
                continue
            if abs(a) > max(0.05 * max_value, 1.0):
                continue
            if _standard_interval_mm(b) is None:
                continue
            denom = max(abs(depth['value_mm']), abs(max_value), 1e-9)
            diff = abs(depth['value_mm'] - max_value) / denom
            if diff > CONFIG['unit_consistency_max_relative_diff']:
                continue
            options.append({
                'unit': unit,
                'values_mm': values,
                'interval_mm': float(b),
                'top_intercept_mm': float(a),
                'depth_consistency_relative_diff': float(diff),
            })
    # Deduplicate by physical values; unit is secondary when values coincide.
    by_values = {}
    for option in options:
        key = tuple(round(v, 6) for v in option['values_mm'])
        by_values.setdefault(key, []).append(option)
    unique = list(by_values.values())
    if len(unique) == 1:
        opts = unique[0]
        chosen = next((o for o in opts if o['unit'] == 'mm'), opts[0])
        return {
            'status': 'resolved', 'reason': '',
            'unit': chosen['unit'],
            'unit_source': 'ruler_geometry_and_depth_context',
            'unit_context': depth['text'],
            'values_mm': chosen['values_mm'],
        }
    if len(unique) > 1:
        return {
            'status': 'ambiguous', 'reason': 'ambiguous_ruler_label_unit',
            'unit': None, 'unit_source': None,
            'unit_context': depth['text'], 'values_mm': None,
        }
    return {
        'status': 'unknown',
        'reason': 'ruler_label_unit_not_proven_by_geometry_or_depth',
        'unit': None, 'unit_source': None,
        'unit_context': depth['text'], 'values_mm': None,
    }


def _numeric_ruler_method(labels, ruler, depth_candidates, width):
    method = {
        'method': 'ruler_numeric',
        'status': 'unknown',
        'reason': '',
        'px_per_mm': None,
        'labels': labels,
        'ticks': list(ruler['ticks']),
        'spacing_px': ruler['spacing_px'],
        'formula': None,
        'unit': None,
        'unit_source': None,
        'pairwise_scales_px_per_mm': [],
        'max_pairwise_relative_spread': None,
        'hypotheses': [],
    }
    if not labels:
        method['reason'] = 'ruler_has_no_numeric_label'
        return method
    if len(labels) == 1:
        # A single label is not a two-point measurement. Keep the old-style
        # standard-interval idea only as an explicitly unconfirmed hypothesis.
        lab = labels[0]
        resolution = _unit_for_ruler_labels(
            labels, ruler, depth_candidates, width)
        unit = (resolution.get('unit')
                if resolution.get('status') == 'resolved' else None)
        if unit is not None:
            value_mm = abs(lab['value']) * (10.0 if unit == 'cm' else 1.0)
            span_px = lab['tick_y'] - ruler['ticks'][0]
            if value_mm > 0:
                method['hypotheses'].append({
                    'type': 'single_label_if_first_tick_is_zero',
                    'label': lab,
                    'unit': unit,
                    'value_mm': value_mm,
                    'span_px': int(span_px),
                    'px_per_mm': float(span_px / value_mm),
                    'reliable': False,
                    'reason': 'first_detected_tick_is_not_proven_zero',
                })
        method['reason'] = 'single_label_without_proven_zero_or_second_endpoint'
        return method

    resolution = _unit_for_ruler_labels(
        labels, ruler, depth_candidates, width)
    method['unit_resolution'] = resolution
    method['unit'] = resolution.get('unit')
    method['unit_source'] = resolution.get('unit_source')
    method['unit_context'] = resolution.get('unit_context')
    values_mm = resolution.get('values_mm')
    method['interpreted_values_mm'] = values_mm
    if resolution.get('status') == 'ambiguous':
        method['status'] = 'ambiguous'
        method['reason'] = resolution.get('reason')
        return method
    if resolution.get('status') != 'resolved' or values_mm is None:
        method['status'] = 'unknown'
        method['reason'] = resolution.get('reason')
        return method
    scales = []
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            dy = abs(labels[j]['tick_y'] - labels[i]['tick_y'])
            dv = abs(values_mm[j] - values_mm[i])
            if dv <= 0:
                continue
            scale = dy / dv
            scales.append(scale)
            pairs.append({
                'i': i, 'j': j,
                'y_i': labels[i]['tick_y'], 'y_j': labels[j]['tick_y'],
                'value_i_mm': values_mm[i], 'value_j_mm': values_mm[j],
                'dy_px': dy, 'dvalue_mm': dv,
                'px_per_mm': float(scale),
            })
    method['pairwise_scales_px_per_mm'] = [float(s) for s in scales]
    method['pairs'] = pairs
    if not scales:
        method['reason'] = 'ruler_labels_have_equal_values'
        return method
    median = float(np.median(scales))
    spread = max(abs(s - median) / median for s in scales)
    method['max_pairwise_relative_spread'] = float(spread)
    method['px_per_mm'] = median
    method['formula'] = 'abs(y2-y1) / abs(value2_mm-value1_mm)'
    if spread > CONFIG['mode_consistency_max_relative_diff']:
        method['status'] = 'ambiguous'
        method['reason'] = 'conflicting_ruler_label_scales'
        method['px_per_mm'] = None
        return method
    method['status'] = 'ok'
    method['evidence_level'] = 'direct_numeric_ticks'
    method['reason'] = ''
    return method


def _unlabelled_tick_hypotheses(ruler, depth_candidates):
    depth, reason = _choose_depth(depth_candidates)
    out = []
    if depth is None:
        return out
    n_intervals = len(ruler['ticks']) - 1
    if n_intervals < 1:
        return out
    for interval_mm in (5.0, 10.0, 20.0, 2.0, 1.0, 25.0, 50.0, 100.0):
        total_mm = n_intervals * interval_mm
        denom = max(total_mm, depth['value_mm'], 1e-9)
        diff = abs(depth['value_mm'] - total_mm) / denom
        if diff <= CONFIG['unit_consistency_max_relative_diff']:
            out.append({
                'type': 'unlabelled_ticks_assumed_interval',
                'interval_mm': interval_mm,
                'total_mm': total_mm,
                'depth_value_mm': depth['value_mm'],
                'depth_consistency_relative_diff': float(diff),
                'span_px': int(ruler['ticks'][-1] - ruler['ticks'][0]),
                'px_per_mm': float((ruler['ticks'][-1] - ruler['ticks'][0])
                                   / total_mm),
                'reliable': False,
                'reason': 'no_numeric_label_and_no_proven_endpoint',
            })
    return out


def _depth_endpoint_qualified(candidate, span):
    """A depth readout must be an explicit depth field or outside the scan.

    UI depth text is normally placed in the side margin; text inside the scan
    region is not accepted as a scan-depth endpoint merely because it contains
    a unit. Explicit Tiefe/Depth/De fields remain usable.
    """
    if candidate.get('context') in ('tiefe', 'depth', 'de'):
        return True
    x = candidate.get('xc')
    if x is None:
        return False
    return x < span['x0'] or x > span['x1']


def _group_depth_values(candidates):
    """Group candidates by physical value, keeping the best-scored first."""
    groups = []
    for candidate in sorted(candidates, key=lambda c: -c['score']):
        for group in groups:
            base = group[0]['value_mm']
            if abs(base - candidate['value_mm']) <= 0.01 * max(
                    abs(base), abs(candidate['value_mm']), 1e-9):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    return groups


def _depth_span_method(depth_candidates, span, ruler, labels):
    """Depth-span scale only when BOTH physical endpoints are proven.

    A bottom depth readout, the first detected tick, the bright-region bounding
    box, or evenly spaced ticks are each insufficient on their own. Accepted
    evidence is either a zero ruler label at the span top plus a depth readout
    at the span bottom, or two depth readouts with known values at the two span
    endpoints. The scale is span-height-pixel difference divided by the physical
    depth difference; a non-zero start is allowed.
    """
    method = {
        'method': 'depth_span',
        'status': 'unknown',
        'reason': '',
        'px_per_mm': None,
        'depth': None,
        'top_endpoint': None,
        'bottom_endpoint': None,
        'delta_value_mm': None,
        'span_height_px': None,
        'span': span,
        'endpoint_basis': None,
        'formula': ('span_height_px / '
                    'abs(bottom_endpoint_mm - top_endpoint_mm)'),
        'hypotheses': [],
    }
    if not depth_candidates:
        method['reason'] = 'no_depth_label_with_unit'
        return method
    if span is None:
        method['reason'] = 'scan_span_not_detected'
        return method
    if span.get('full_frame'):
        method['reason'] = 'scan_span_is_full_frame'
        return method
    h = max(1, span['y1'] - span['y0'])
    max_offset = CONFIG['depth_span_endpoint_max_offset_fraction'] * h
    top_limit = span['y0'] + max_offset
    bottom_limit = span['y1'] - max_offset
    positive = [c for c in depth_candidates if c['value_mm'] > 0]
    near_top_all = [c for c in positive if c['yc'] <= top_limit]
    near_bottom_all = [c for c in positive if c['yc'] >= bottom_limit]
    qualified = [c for c in positive if _depth_endpoint_qualified(c, span)]
    near_top = [c for c in qualified if c['yc'] <= top_limit]
    near_bottom = [c for c in qualified if c['yc'] >= bottom_limit]
    zero_labels = [lab for lab in labels
                   if abs(lab['value']) < 1e-9
                   and abs(lab['tick_y'] - span['y0']) <= max_offset]
    ruler_aligned = False
    if ruler is not None:
        ruler_aligned = (abs(ruler['ticks'][0] - span['y0']) <= max_offset
                         and abs(ruler['ticks'][-1] - span['y1']) <= max_offset)
    method.update({
        'near_bottom_labels': near_bottom,
        'near_bottom_labels_all': near_bottom_all,
        'near_top_labels': near_top,
        'zero_labels_at_span_top': zero_labels,
        'ruler_endpoints_align_span': bool(ruler_aligned),
        'bottom_limit_y': float(bottom_limit),
        'top_limit_y': float(top_limit),
        'endpoint_max_offset_px': float(max_offset),
        'endpoint_qualification': ('explicit_depth_field_or_outside_scan_x'),
    })

    top_groups = _group_depth_values(near_top)
    bottom_groups = _group_depth_values(near_bottom)
    bottom_all_groups = _group_depth_values(near_bottom_all)
    top_options = []
    if zero_labels:
        # A zero ruler label proves the top endpoint; a depth field near the
        # top is normally the total-depth setting, not a second top value.
        top_options.append({
            'type': 'zero_ruler_label',
            'value_mm': 0.0,
            'label': zero_labels[0],
        })
    else:
        for group in top_groups:
            top_options.append({
                'type': 'depth_label',
                'value_mm': float(group[0]['value_mm']),
                'candidate': group[0],
                'group': group,
            })
    # Merge top options that describe the same physical value.
    merged_top = []
    for option in top_options:
        for existing in merged_top:
            if abs(existing['value_mm'] - option['value_mm']) <= 0.01 * max(
                    abs(existing['value_mm']), abs(option['value_mm']), 1e-9):
                break
        else:
            merged_top.append(option)
    if len(merged_top) > 1:
        method.update({
            'status': 'ambiguous',
            'reason': 'conflicting_top_endpoint_values',
            'top_endpoint_options': merged_top,
        })
        return method
    if len(bottom_groups) > 1:
        method.update({
            'status': 'ambiguous',
            'reason': 'conflicting_depth_labels_at_span_bottom',
            'bottom_endpoint_groups': bottom_groups,
        })
        return method

    top = merged_top[0] if merged_top else None
    bottom = bottom_groups[0][0] if bottom_groups else None
    if bottom is None:
        method['reason'] = 'depth_span_endpoint_unverified'
        # Keep the old conditional zero-to-depth estimate as an explicit
        # hypothesis, including readouts overlaid inside the scan region. It is
        # never promoted to a reliable scale without a proven top endpoint.
        for group in bottom_all_groups:
            candidate = group[0]
            method['hypotheses'].append({
                'type': 'depth_span_if_span_is_full_zero_to_depth',
                'depth_value_mm': float(candidate['value_mm']),
                'span_height_px': span['height_px'],
                'px_per_mm': float(
                    span['height_px'] / candidate['value_mm']),
                'reliable': False,
                'reason': ('bottom depth readout alone does not prove the '
                           'span starts at physical zero; no qualified zero '
                           'ruler label or known top depth endpoint'),
            })
        return method
    method['bottom_endpoint'] = {
        'type': 'depth_label',
        'value_mm': float(bottom['value_mm']),
        'text': bottom['text'],
        'candidate': bottom,
    }
    method['depth'] = bottom
    if top is None:
        method['reason'] = 'depth_span_endpoint_unverified'
        method['hypotheses'].append({
            'type': 'depth_span_if_span_is_full_zero_to_depth',
            'depth_value_mm': float(bottom['value_mm']),
            'span_height_px': span['height_px'],
            'px_per_mm': float(span['height_px'] / bottom['value_mm']),
            'reliable': False,
            'reason': ('bottom depth readout alone does not prove the span '
                       'starts at physical zero; no zero ruler label or known '
                       'top depth endpoint'),
        })
        return method
    delta = float(bottom['value_mm']) - float(top['value_mm'])
    if delta <= 0:
        method.update({
            'status': 'ambiguous',
            'reason': 'depth_endpoint_values_not_increasing',
            'top_endpoint': top if top['type'] == 'zero_ruler_label' else {
                'type': 'depth_label',
                'value_mm': top['value_mm'],
                'text': top['candidate']['text'],
                'candidate': top['candidate'],
            },
        })
        return method
    method['top_endpoint'] = top if top['type'] == 'zero_ruler_label' else {
        'type': 'depth_label',
        'value_mm': top['value_mm'],
        'text': top['candidate']['text'],
        'candidate': top['candidate'],
    }
    method['delta_value_mm'] = delta
    method['span_height_px'] = span['height_px']
    method['px_per_mm'] = float(span['height_px'] / delta)
    method['status'] = 'ok'
    method['endpoint_basis'] = (
        'zero_label_at_span_top_and_depth_label_at_span_bottom'
        if top['type'] == 'zero_ruler_label'
        else 'depth_labels_at_span_top_and_bottom')
    method['evidence_level'] = 'depth_label_plus_bounded_span'
    method['reason'] = ''
    return method


def _metadata_method(metadata, image_id, actual_identity):
    method = {
        'method': 'metadata',
        'status': 'absent',
        'reason': '',
        'px_per_mm': None,
        'source': None,
        'evidence_level': 'explicit_metadata',
        'binding': {},
        'metadata': metadata,
    }
    if metadata is None:
        return method
    actual = actual_identity or {}
    required = CONFIG['metadata_required_identity_fields']
    missing = [k for k in required if k not in metadata]
    if missing:
        method.update({'status': 'invalid',
                       'reason': f'metadata_unbound_missing:{",".join(missing)}'})
        return method
    binding = {
        'image_id_match': metadata.get('image_id') == image_id,
        'pixel_sha256_match': metadata.get('pixel_sha256')
        == actual.get('pixel_sha256'),
        'width_match': int(metadata.get('width')) == int(actual.get('width')),
        'height_match': int(metadata.get('height')) == int(actual.get('height')),
    }
    method['binding'] = binding
    if not all(binding.values()):
        method.update({'status': 'invalid',
                       'reason': 'metadata_identity_mismatch'})
        return method
    ok, reason = validate_scale(metadata.get('px_per_mm'))
    if not ok:
        method.update({'status': 'invalid',
                       'reason': f'metadata_scale_{reason}'})
        return method
    method.update({
        'status': 'ok',
        'px_per_mm': float(metadata['px_per_mm']),
        'source': metadata.get('source', 'explicit_metadata'),
        'reason': '',
    })
    return method


def _method_summary(method):
    return {
        'status': method.get('status'),
        'px_per_mm': method.get('px_per_mm'),
        'reason': method.get('reason'),
        'evidence_level': method.get('evidence_level'),
        'source': method.get('source'),
    }


def infer_scale_from_observations(width, height, ocr_items, gray,
                                  metadata=None, expected_identity=None,
                                  actual_identity=None, image_id=None):
    """Infer scale with explicit method separation and conflict propagation."""
    result = {
        'status': 'unknown',
        'px_per_mm': None,
        'source': None,
        'evidence_level': None,
        'reason': '',
        'width': width, 'height': height,
        'image_id': image_id,
        'metadata_evidence': metadata or {},
        'depth_candidates': [],
        'ruler_evidence': None,
        'scan_span': None,
        'mode_scales': {},
        'method_results': {},
        'method_comparison': None,
        'conflicts': [],
        'hypotheses': [],
        'excluded_observations': [],
        'module_version': MODULE_VERSION,
        'config': CONFIG,
    }
    actual = actual_identity or {}
    expected = expected_identity
    if expected is not None and actual:
        ok, bad = validate_identity(actual, expected)
        if not ok:
            result.update({'status': 'invalid',
                           'reason': f'identity_mismatch:{",".join(bad)}'})
            return result

    methods = {}
    methods['metadata'] = _metadata_method(metadata, image_id, actual)

    ruler = detect_ruler(gray)
    labels = []
    excluded = []
    if ruler is not None:
        # Identify pure numeric/unit ruler labels first, so the same OCR
        # observations are not also treated as depth-field candidates.
        labels, excluded = _labels_for_ruler(
            ocr_items, ruler, width, height, [])
    depth_candidates = _depth_candidates(ocr_items, width, height)
    if labels:
        label_keys = {(lab['text'], round(lab['xc']), round(lab['yc']))
                      for lab in labels}
        depth_candidates = [
            c for c in depth_candidates
            if (c['text'], round(c['xc']), round(c['yc'])) not in label_keys]
    result['depth_candidates'] = depth_candidates
    if ruler is not None:
        methods['ruler_numeric'] = _numeric_ruler_method(
            labels, ruler, depth_candidates, width)
        result['ruler_evidence'] = {
            **ruler, 'labels': labels,
            'method': methods['ruler_numeric'],
        }
    else:
        methods['ruler_numeric'] = {
            'method': 'ruler_numeric', 'status': 'unknown',
            'reason': 'no_vertical_ruler_detected', 'px_per_mm': None,
            'labels': [], 'hypotheses': [],
        }
    result['excluded_observations'] = excluded

    span = detect_scan_span(gray)
    result['scan_span'] = span
    methods['depth_span'] = _depth_span_method(
        depth_candidates, span, ruler, labels)

    # Keep old-style diagnostic hypotheses explicitly out of px_per_mm.
    if ruler is not None and not labels:
        result['hypotheses'].extend(_unlabelled_tick_hypotheses(
            ruler, depth_candidates))
    if ruler is not None and len(labels) == 1:
        result['hypotheses'].extend(
            methods['ruler_numeric'].get('hypotheses', []))
    result['hypotheses'].extend(
        methods['depth_span'].get('hypotheses', []))

    result['method_results'] = methods
    for name, method in methods.items():
        if method.get('status') == 'ok' and method.get('px_per_mm') is not None:
            result['mode_scales'][name] = float(method['px_per_mm'])

    # Credible conflicts are never hidden by a fallback path.
    ambiguous = [name for name, m in methods.items()
                 if m.get('status') == 'ambiguous']
    if ambiguous:
        result.update({
            'status': 'ambiguous',
            'px_per_mm': None,
            'source': 'conflicting_evidence',
            'evidence_level': 'conflict',
            'reason': ';'.join(f'{n}:{methods[n].get("reason")}'
                               for n in ambiguous),
            'conflicts': ambiguous,
        })
        return result

    ok_methods = [name for name, m in methods.items()
                  if m.get('status') == 'ok' and m.get('px_per_mm') is not None]
    if ok_methods:
        values = {n: float(methods[n]['px_per_mm']) for n in ok_methods}
        if len(values) > 1:
            pairs = []
            names = list(values)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = values[names[i]], values[names[j]]
                    rel = abs(a - b) / max(a, b)
                    pairs.append({'a': names[i], 'b': names[j],
                                  'relative_difference': float(rel)})
            worst = max(p['relative_difference'] for p in pairs)
            result['method_comparison'] = {
                'pairs': pairs,
                'max_relative_difference': float(worst),
            }
            if worst > CONFIG['mode_consistency_max_relative_diff']:
                result.update({
                    'status': 'ambiguous',
                    'px_per_mm': None,
                    'source': 'conflicting_evidence',
                    'evidence_level': 'conflict',
                    'reason': 'credible_scale_methods_disagree',
                    'conflicts': [p for p in pairs
                                  if p['relative_difference']
                                  > CONFIG['mode_consistency_max_relative_diff']],
                })
                return result
        hierarchy = ['metadata', 'ruler_numeric', 'depth_span']
        chosen = next(n for n in hierarchy if n in values)
        result.update({
            'status': 'ok',
            'px_per_mm': values[chosen],
            'source': methods[chosen].get('source') or chosen,
            'evidence_level': methods[chosen].get('evidence_level') or chosen,
            'reason': '',
        })
        return result

    # No credible scale. Keep invalid metadata reasons visible but distinguish
    # them from image-evidence absence.
    invalid_metadata = [m.get('reason') for m in [methods['metadata']]
                        if m.get('status') == 'invalid']
    reasons = []
    for name in ('ruler_numeric', 'depth_span'):
        reason = methods[name].get('reason')
        if reason:
            reasons.append(f'{name}:{reason}')
    reasons.extend(f'metadata:{r}' for r in invalid_metadata)
    result['reason'] = ';'.join(reasons) or 'no_scale_evidence'
    return result



def _draw_evidence(image, result, out_path):
    img = image.convert('RGB').copy()
    scale = min(1.0, CONFIG['evidence_max_width'] / max(1, img.width))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    def xy(box):
        return [(x * scale, y * scale) for x, y in box]

    for c in result.get('depth_candidates', []):
        box = xy(c['box'])
        bx0 = min(p[0] for p in box); bx1 = max(p[0] for p in box)
        by0 = min(p[1] for p in box); by1 = max(p[1] for p in box)
        draw.rectangle([bx0, by0, bx1, by1],
                       outline=(255, 80, 80), width=2)
        draw.text((bx0, max(0, by0 - 16)),
                  f"{c['value']} {c['unit']} ({c['context']})",
                  fill=(255, 80, 80), font=font)
    ruler = result.get('ruler_evidence')
    if ruler:
        lx = ruler['line_x'] * scale
        draw.line([(lx, 0), (lx, img.height)], fill=(80, 200, 255), width=1)
        for ty in ruler['ticks']:
            draw.line([(lx - 6, ty * scale), (lx + 6, ty * scale)],
                      fill=(80, 200, 255), width=2)
        for lab in ruler.get('labels', []):
            bx = lab['xc'] * scale
            by = lab['tick_y'] * scale
            draw.ellipse([bx - 3, by - 3, bx + 3, by + 3],
                         outline=(255, 255, 0), width=2)
    span = result.get('scan_span')
    if span:
        draw.rectangle([span['x0'] * scale, span['y0'] * scale,
                        span['x1'] * scale, span['y1'] * scale],
                       outline=(0, 255, 120), width=2)
    text = (f"{result.get('status')} "
            f"px/mm={result.get('px_per_mm')} "
            f"source={result.get('source')} "
            f"evidence={result.get('evidence_level')} "
            f"reason={result.get('reason')}")
    draw.rectangle([0, 0, min(img.width, 900), 20], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _dedupe_ocr_items(items):
    seen = set()
    out = []
    for item in sorted(items, key=lambda r: -r['score']):
        key = (item['text'], round(item['xc'] / 3), round(item['yc'] / 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def recognize_scale(path, image_id, expected_identity=None,
                    metadata=None, evidence_path=None,
                    ocr_items=None, save_evidence=True,
                    input_manifest_sha256=None, ocr_source='fresh_ocr',
                    decision_source='recognize_scale_current_module'):
    """Recognize the scale for one image and optionally save an evidence image."""
    path = Path(path)
    record = {
        'image_id': image_id,
        'path': str(path),
        'module_version': MODULE_VERSION,
        'config': CONFIG,
        'config_sha256': CONFIG_SHA256,
        'recognition_implementation_sha256': CODE_SHA256,
        'decision_source': decision_source,
        'ocr_engine': CONFIG['ocr_engine'],
        'ocr_engine_version': ocr_engine_version(),
        'ocr_coordinate_transform_version': CONFIG[
            'ocr_coordinate_transform_version'],
        'ocr_source': ocr_source,
        'input_manifest_sha256': input_manifest_sha256,
        'status': 'unknown',
        'px_per_mm': None,
        'source': None,
        'evidence_level': None,
        'reason': '',
        'evidence_image': None,
    }
    try:
        with Image.open(path) as im:
            image = im.convert('RGB')
            size = image.size
            gray = np.asarray(image.convert('L'))
            info = dict(im.info)
            tags = dict(im.tag_v2) if hasattr(im, 'tag_v2') else {}
        actual = image_identity(path)
    except Exception as exc:
        record.update({'status': 'invalid',
                       'reason': f'input:{type(exc).__name__}: {exc}',
                       'width': None, 'height': None,
                       'pixel_sha256': None})
        return record
    imagej = tags.get(270)
    def _nums(values):
        if not values:
            return None
        try:
            return [float(v) for v in values]
        except (TypeError, ValueError):
            return [str(v) for v in values]

    embedded_metadata = {
        'source': 'embedded_image_metadata_check',
        'dpi': _nums(info.get('dpi')),
        'resolution': _nums(info.get('resolution')),
        'imagej_description': str(imagej) if imagej else None,
        'has_explicit_ultrasound_px_per_mm': False,
        'accepted_for_scale': False,
        'reason': ('embedded DPI/print resolution is not ultrasound spatial '
                   'calibration; not accepted'),
    }
    record.update({
        'width': actual['width'], 'height': actual['height'],
        'pixel_sha256': actual['pixel_sha256'],
        'embedded_metadata': embedded_metadata,
    })
    if expected_identity is not None:
        ok, bad = validate_identity(actual, expected_identity)
        if not ok:
            record.update({'status': 'invalid',
                           'reason': f'identity_mismatch:{",".join(bad)}'})
            return record
    ocr_items_provided = ocr_items is not None
    if ocr_items is None:
        ocr_items = run_ocr(image, CONFIG['ocr_orientations_fast'])
        # If the fast orientations produce no unit-bearing or numeric label
        # evidence, try the upside-down orientations once.
        has_evidence = any(parse_number_unit(i['text']) for i in ocr_items)
        if not has_evidence:
            ocr_items = ocr_items + run_ocr(
                image, CONFIG['ocr_orientations_fallback'])
        if not any(parse_number_unit(i['text']) for i in ocr_items):
            ocr_items = ocr_items + supplemental_ocr(image, ocr_items)
    record['ruler_reread'] = None
    record['bottom_reread'] = None
    ruler_probe = detect_ruler(gray) if not ocr_items_provided else None
    if not ocr_items_provided:
        if ruler_probe is not None:
            half = CONFIG['ruler_reread_half_width_px']
            x0 = max(0, ruler_probe['line_x'] - half)
            x1 = min(actual['width'], ruler_probe['line_x'] + half)
            readded = run_ocr_region(
                image, (x0, 0, x1, actual['height']),
                upscale=CONFIG['ruler_reread_upscale'])
            ocr_items = _dedupe_ocr_items(ocr_items + readded)
            record['ruler_reread'] = {
                'region': [x0, 0, x1, actual['height']],
                'upscale': CONFIG['ruler_reread_upscale'],
                'n_added_items': len(readded),
                'added_texts': [{'text': i['text'], 'score': i['score'],
                                 'box': i['box']} for i in readded],
            }
            # Fixed top-band reread for the visible zero label. The full
            # ruler crop can make small zero glyphs too small for OCR.
            zh = max(20, int(CONFIG['zero_reread_height_fraction']
                             * actual['height']))
            zx0 = max(0, ruler_probe['line_x']
                      - CONFIG['zero_reread_half_width_px'])
            zx1 = min(actual['width'], ruler_probe['line_x']
                      + CONFIG['zero_reread_half_width_px'])
            zitems = run_ocr_region(
                image, (zx0, 0, zx1, zh),
                upscale=CONFIG['zero_reread_upscale'])
            ocr_items = _dedupe_ocr_items(ocr_items + zitems)
            record['ruler_zero_reread'] = {
                'region': [zx0, 0, zx1, zh],
                'upscale': CONFIG['zero_reread_upscale'],
                'n_added_items': len(zitems),
                'added_texts': [{'text': i['text'], 'score': i['score'],
                                 'box': i['box']} for i in zitems],
            }
        else:
            # Depth-labels are often placed at the bottom edge of images
            # without a long vertical ruler. A fixed bottom-band reread
            # recovers small "4 cm" labels without per-image tuning.
            bh = max(40, int(CONFIG['bottom_reread_height_fraction']
                             * actual['height']))
            region = (0, actual['height'] - bh, actual['width'],
                      actual['height'])
            bitems = run_ocr_region(
                image, region, upscale=CONFIG['bottom_reread_upscale'])
            ocr_items = _dedupe_ocr_items(ocr_items + bitems)
            record['bottom_reread'] = {
                'region': list(region),
                'upscale': CONFIG['bottom_reread_upscale'],
                'n_added_items': len(bitems),
                'added_texts': [{'text': i['text'], 'score': i['score'],
                                 'box': i['box']} for i in bitems],
            }
    result = infer_scale_from_observations(
        actual['width'], actual['height'], ocr_items, gray,
        metadata=metadata, expected_identity=expected_identity,
        actual_identity=actual, image_id=image_id)
    record.update(result)
    record['ocr_items'] = ocr_items
    if save_evidence and evidence_path is not None:
        try:
            _draw_evidence(image, record, Path(evidence_path))
            record['evidence_image'] = str(evidence_path)
        except Exception as exc:
            record['evidence_error'] = f'{type(exc).__name__}: {exc}'
    return record
