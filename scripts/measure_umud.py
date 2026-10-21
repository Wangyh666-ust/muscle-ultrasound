"""Unified fixed-model UMUD measurement prototype.

Model combination (engineering version, not an accuracy claim):
- frozen apo model: runs/baseline_b/apo/checkpoints/best.pt
- frozen Round13 B ResNet34-UNet: runs/round13_fasc_b_unet34_v1/checkpoints/best.pt

Inputs are only the original image, an optional px_per_mm, and necessary
identity metadata. Expert PA/FL/MT values are never accepted or read.

Outputs PA, MT px/mm, FL G0/G1 px/mm, actual selected candidates, extrapolation
ratios, +/-1 degree sensitivity, validity diagnostics and independent statuses.
G1 is explicitly experimental. No training, ensemble, threshold change or
Kaggle submission occurs.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.umud_baseline.data import (  # noqa: E402
    read_image_gray3, resize_long_side, restore_original_coords,
)
from src.umud_baseline.apo_baseline import (  # noqa: E402
    load_frozen_rule, measure_conservative,
)
from src.umud_baseline.train import build_model  # noqa: E402
from scripts.fascicle_pa_fl import (  # noqa: E402
    evaluate_mask_pair, fl_for_segment,
)
from scripts.fl_acceptance_policy import (  # noqa: E402
    policy_equal, policy_fields, policy_from_candidate,
    policy_failure_conditions, select_with_policy,
)
from scripts.scale_recognition import (  # noqa: E402
    CODE_SHA256 as SCALE_IMPLEMENTATION_SHA256,
    CONFIG as SCALE_CONFIG,
    CONFIG_SHA256 as SCALE_CONFIG_SHA256,
    MODULE_VERSION as SCALE_MODULE_VERSION,
)

DEFAULT_APO = ROOT / 'runs/baseline_b/apo/checkpoints/best.pt'
DEFAULT_FASC = ROOT / 'runs/round13_fasc_b_unet34_v1/checkpoints/best.pt'
DEFAULT_ELIGIBLE = ROOT / 'external/umud_expert/eligible_dev.json'
OUT = ROOT / 'runs/round15_measurement_prototype_v2/measurement'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXCLUDED = {'im_07_arch'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck.get('config') or {}
    arch = ck.get('architecture') or cfg.get('architecture', 'resnet34_unet')
    enc = ck.get('encoder_name') or cfg.get('encoder_name')
    return ck, cfg, arch, enc


def load_checkpoint_model(path, device):
    ck, cfg, arch, enc = read_checkpoint(path)
    model = build_model(pretrained=False, architecture=arch,
                        encoder_name=enc, encoder_weights=None)
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    return model, ck, cfg


def infer_mask(image_path, model, long_side, device, amp=True):
    image = read_image_gray3(Path(image_path))
    h0, w0 = image.shape[:2]
    small, scale = resize_long_side(image, long_side, is_mask=False)
    h, w = small.shape[:2]
    canvas = int(long_side)
    if canvas % 32:
        raise ValueError('long_side must be divisible by 32')
    if h > canvas or w > canvas:
        raise ValueError('resized image exceeds canvas')
    padded = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    padded[:h, :w] = small
    pad = (canvas - w, canvas - h)
    x = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast('cuda', enabled=amp and device == 'cuda'):
        prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    prob = restore_original_coords(prob.astype(np.float32), (w0, h0),
                                   float(scale), pad, method='nearest')
    mask = (prob > 0.5).astype(np.uint8)
    return mask, prob


def validate_scale(value):
    """None means unknown; otherwise require a finite positive number."""
    if value is None:
        return True, None, ''
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, None, 'not_numeric'
    if not math.isfinite(v):
        return False, None, 'non_finite'
    if v <= 0:
        return False, None, 'not_positive'
    return True, v, ''


def scale_value(value, px_per_mm):
    ok, scale, _ = validate_scale(px_per_mm)
    if value is None or not ok:
        return None
    try:
        out = float(value) / scale
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) and out >= 0 else None


def image_identity(path):
    """Existing external identity definition: convert('L') pixel bytes."""
    with Image.open(path) as im:
        w, h = im.size
        gray = im.convert('L')
        digest = hashlib.sha256(gray.tobytes()).hexdigest()
    return {'width': int(w), 'height': int(h), 'pixel_sha256': digest}


class PolicyZeroMismatch(RuntimeError):
    pass


def policy_sensitivity(orig, fits, width, height, image_apo_ok):
    """Compare original, 0, -1 and +1 under the same policy and apo status.

    Base acceptance for G0/G1 is derived from the unmodified original candidate
    conditions; a G1 copy is never used to reconstruct G0.
    """
    seg = orig['segment']
    fl0 = fl_for_segment(seg, fits, width, height, apo_ok=image_apo_ok)
    bases = {}
    for policy in ('G0', 'G1'):
        from_fl = policy_fields(fl0, policy)
        from_orig = policy_from_candidate(orig, policy)
        if not policy_equal(from_fl, from_orig):
            raise PolicyZeroMismatch(
                f'0-degree policy mismatch {policy} for {orig.get("label_id")}')
        bases[policy] = from_fl
    changes = {'G0': [], 'G1': []}
    vals = []
    for delta in (-1.0, 1.0):
        s = dict(seg)
        s['direction_deg'] = float(s['direction_deg']) + delta
        fl = fl_for_segment(s, fits, width, height, apo_ok=image_apo_ok)
        if fl.get('fl_px_raw') is not None and orig.get('fl_px_raw') is not None:
            vals.append(abs(float(fl['fl_px_raw']) - float(orig['fl_px_raw'])))
        for policy in ('G0', 'G1'):
            if bool(policy_fields(fl, policy)['accept']) != bool(
                    bases[policy]['accept']):
                changes[policy].append(f'{delta:+.0f}')
    return {
        'max_abs_fl_change_px': max(vals) if vals else None,
        'g0_status_changes_at': changes['G0'],
        'g1_status_changes_at': changes['G1'],
        'g0_zero_degree_equal': True,
        'g1_zero_degree_equal': True,
    }


def selected_details(orig_candidates, selected_originals, fits, width, height,
                     image_apo_ok, image_id, policy):
    """Build details in selector order from stable original candidates."""
    out = []
    for order, c in enumerate(selected_originals):
        seg = c['segment']
        label = int(seg.get('label_id', -1))
        sens = policy_sensitivity(c, fits, width, height, image_apo_ok)
        pfields = policy_from_candidate(c, policy)
        out.append({
            'candidate_identity': f'{image_id}:{label}',
            'label_id': label,
            'candidate_index': label,
            'selection_order': order,
            'policy': policy,
            'centroid': seg.get('centroid'),
            'p0': seg.get('p0'), 'p1': seg.get('p1'),
            'direction_deg': seg.get('direction_deg'),
            'fl_px': c.get('fl_px_raw'),
            'fl_px_policy': pfields['fl_px_pure'],
            'extrapolation_ratio': c.get('extrapolation_ratio'),
            'intersection_sup': c.get('intersection_sup'),
            'intersection_deep': c.get('intersection_deep'),
            'outside_image': bool(c.get('fl_outside_image')),
            'outside_apo_support': bool(c.get('fl_outside_apo_support')),
            'validity_status': c.get('candidate_validity_status'),
            'validity_conditions': c.get('candidate_validity_conditions'),
            'direction_sensitivity': sens,
            f'{policy.lower()}_status_changes_at': (
                sens[f'{policy.lower()}_status_changes_at']),
        })
    return out


def pa_details(selected, image_id):
    out = []
    for order, c in enumerate(selected):
        seg = c['segment']
        label = int(seg.get('label_id', -1))
        out.append({
            'candidate_identity': f'{image_id}:{label}',
            'label_id': label,
            'candidate_index': label,
            'selection_order': order,
            'pa_deg': c.get('pa_deg'),
            'pa_deg_raw': c.get('pa_deg_raw'),
            'segment_direction_deg': c.get('segment_direction_deg',
                                           seg.get('direction_deg')),
            'deep_direction_deg': c.get('deep_direction_deg'),
            'intersection_deep': c.get('intersection_deep'),
            'p0': seg.get('p0'), 'p1': seg.get('p1'),
            'validity_status': c.get('candidate_validity_status'),
            'validity_conditions': c.get('candidate_validity_conditions'),
        })
    return out

def selected_invalid_summary(cands, pa_rows, g0_details, g1_details, image_id):
    """Count selected invalid candidates by parameter and as a unique union.

    Identities are image_id + component label_id, so G1 policy copies and
    original candidates bind to the same candidate. The union is deduplicated
    by identity, not a sum of per-parameter counts.
    """
    def identity(c):
        return f"{image_id}:{int(c['segment']['label_id'])}"
    invalid = [c for c in cands if c.get('candidate_validity_status') == 'invalid']
    invalid_ids = {identity(c) for c in invalid}
    pa_ids = {d['candidate_identity'] for d in pa_rows}
    g0_ids = {d['candidate_identity'] for d in g0_details}
    g1_ids = {d['candidate_identity'] for d in g1_details}
    counts = {
        'PA': len(pa_ids & invalid_ids),
        'FL_G0': len(g0_ids & invalid_ids),
        'FL_G1': len(g1_ids & invalid_ids),
    }
    union = sorted((pa_ids | g0_ids | g1_ids) & invalid_ids)
    return {
        'n_candidates': len(cands),
        'n_invalid': len(invalid),
        'invalid_candidate_identities': sorted(invalid_ids),
        'selected_invalid_counts': counts,
        'n_selected_invalid_parameter_count_sum': sum(counts.values()),
        'n_selected_invalid_union_unique': len(union),
        'selected_invalid_union_definition': (
            'unique candidate identities selected by PA or FL_G0 or FL_G1'),
        'selected_invalid_identities_union': union,
        'valid_is_not_expert_confirmation': True,
    }


def _failed_parameters(reason):
    def base(experimental=False):
        return {
            'status': 'failed', 'reason': reason,
            'experimental': experimental,
            'selected_count': 0,
            'selected_details': [],
        }
    return {
        'MT': {'status': 'failed', 'reason': reason, 'value_px': None,
               'value_mm': None, 'selected_positions_px': [], 'selected_values_px': [],
               'selected_count': 0, 'apo_status': 'failed', 'apo_reason': reason},
        'PA': {**base(), 'value_deg': None, 'candidate_count': 0,
               'selected_pa_deg': [], 'selected_candidate_identities': []},
        'FL_G0': {**base(False), 'selected_fl_px': [], 'selected_fl_mm': [],
                  'selected_mean_px': None, 'selected_mean_mm': None,
                  'extrapolation_ratios': [], 'outside_image_selected_count': 0,
                  'outside_support_selected_count': 0,
                  'selected_invalid_count': 0,
                  'selected_candidate_identities': []},
        'FL_G1': {**base(True), 'selected_fl_px': [], 'selected_fl_mm': [],
                  'selected_mean_px': None, 'selected_mean_mm': None,
                  'extrapolation_ratios': [], 'outside_image_selected_count': 0,
                  'outside_support_selected_count': 0,
                  'selected_invalid_count': 0,
                  'selected_candidate_identities': [],
                  'label': 'experimental extrapolation output'},
    }


def _failure_record(image_id, image_path, px_per_mm, scale_status, scale_source,
                    reason, stage='input', expected=None, actual=None,
                    scale_reason='', inference_source='none',
                    scale_reader_version=None, scale_binding_status=None):
    actual = actual or {}
    rec = {
        'image_id': image_id, 'image_path': str(Path(image_path)),
        'width': actual.get('width'), 'height': actual.get('height'),
        'px_per_mm': None if scale_status in ('invalid', 'invalid_identity') else px_per_mm,
        'input_px_per_mm': px_per_mm,
        'scale_status': scale_status, 'scale_source': scale_source,
        'scale_reason': scale_reason,
        'scale_reader_version': scale_reader_version,
        'scale_binding_status': scale_binding_status,
        'model_combination': {
            'apo': 'runs/baseline_b/apo/checkpoints/best.pt',
            'fasc': 'runs/round13_fasc_b_unet34_v1/checkpoints/best.pt',
            'fasc_arm': 'B_resnet34_unet', 'engineering_version': True,
            'accuracy_claim': False,
        },
        'parameters': _failed_parameters(reason),
        'failure_reasons': [f'{stage}:{reason}'],
        'error': reason, 'error_stage': stage,
        'experimental_policy': 'G1',
        'expert_values_used': False,
        'expected_identity': expected, 'actual_identity': actual or None,
        'inference_source': inference_source,
        'validity': {'n_candidates': 0, 'n_evaluable': 0, 'n_invalid': 0,
                     'invalid_candidate_identities': [],
                     'selected_invalid_counts': {'PA': 0, 'FL_G0': 0, 'FL_G1': 0},
                     'n_selected_invalid_parameter_count_sum': 0,
                     'n_selected_invalid_union_unique': 0,
                     'selected_invalid_union_definition': (
                         'unique candidate identities selected by PA or FL_G0 or FL_G1'),
                     'selected_invalid_identities_union': [],
                     'valid_is_not_expert_confirmation': True},
        'candidate_counts': {'n_evaluable': 0, 'n_short': 0, 'n_crossing': 0,
                             'n_pa_selected_g0': 0, 'n_fl_selected_g0': 0,
                             'n_fl_selected_g1': 0},
        'apo': {'status': 'failed', 'reason': reason, 'source': None,
                'rule_version': None, 'rule_sha256': None,
                'fits': None, 'support': None},
        'plot_status': 'pending', 'plot_info': None,
        'mask_save_status': 'no_mask',
        'structure_and_values_unchanged': False,
    }
    return rec


def measure_one(image_path, image_id, px_per_mm, apo_model, fasc_model,
                apo_side, fasc_side, rule, rule_hash, device, amp=True,
                expected_identity=None, scale_status=None, scale_source=None,
                scale_reason='', mask_paths=None, scale_reader_version=None,
                scale_binding_status=None):
    scale_status = scale_status or (
        'provided' if px_per_mm is not None else 'unknown')
    scale_source = scale_source or ('user' if px_per_mm is not None else 'unknown')
    if scale_status not in ('provided', 'ok'):
        px_per_mm = None
    record = {
        'image_id': image_id,
        'image_path': str(Path(image_path)),
        'px_per_mm': px_per_mm,
        'input_px_per_mm': px_per_mm,
        'scale_status': scale_status,
        'scale_source': scale_source,
        'scale_reason': scale_reason,
        'scale_reader_version': scale_reader_version,
        'scale_binding_status': scale_binding_status,
        'model_combination': {
            'apo': 'runs/baseline_b/apo/checkpoints/best.pt',
            'fasc': 'runs/round13_fasc_b_unet34_v1/checkpoints/best.pt',
            'fasc_arm': 'B_resnet34_unet', 'engineering_version': True,
            'accuracy_claim': False,
        },
        'parameters': {},
        'failure_reasons': [],
        'experimental_policy': 'G1',
        'expert_values_used': False,
        'error': None, 'error_stage': None,
        'expected_identity': expected_identity,
        'actual_identity': None,
        'inference_source': 'none',
        'plot_status': 'pending', 'plot_info': None,
        'mask_save_status': 'no_mask',
    }
    try:
        actual = image_identity(image_path)
    except Exception as exc:
        return _failure_record(
            image_id, image_path, px_per_mm, scale_status, scale_source,
            f'{type(exc).__name__}: {exc}', 'input', expected_identity, None,
            scale_reason=scale_reason, inference_source='none',
            scale_reader_version=scale_reader_version,
            scale_binding_status=scale_binding_status)
    record['width'] = actual['width']; record['height'] = actual['height']
    record['actual_identity'] = actual
    try:
        if expected_identity is not None:
            record['expected_identity'] = expected_identity
            bad = []
            if actual['pixel_sha256'] != expected_identity.get('pixel_sha256'):
                bad.append('pixel_sha256')
            if actual['width'] != int(expected_identity.get('width')):
                bad.append('width')
            if actual['height'] != int(expected_identity.get('height')):
                bad.append('height')
            if bad:
                return _failure_record(
                    image_id, image_path, px_per_mm, 'invalid_identity',
                    scale_source, f'identity_mismatch:{",".join(bad)}',
                    'input_identity', expected_identity, actual,
                    scale_reason=scale_reason,
                    scale_reader_version=scale_reader_version,
                    scale_binding_status=scale_binding_status)
        if scale_status not in ('provided', 'ok') and scale_reason:
            record['failure_reasons'].append(
                f'scale:{scale_status}:{scale_reason.replace(";", "|")}')
        if mask_paths is not None:
            try:
                apo_mask = np.asarray(Image.open(mask_paths['apo']).convert('L')) > 127
                fasc_mask = np.asarray(Image.open(mask_paths['fasc']).convert('L')) > 127
            except Exception as exc:
                return _failure_record(
                    image_id, image_path, px_per_mm, scale_status, scale_source,
                    f'{type(exc).__name__}: {exc}', 'input_mask',
                    expected_identity, actual, scale_reason=scale_reason,
                    inference_source='cached_masks',
                    scale_reader_version=scale_reader_version,
                    scale_binding_status=scale_binding_status)
            apo_mask = apo_mask.astype(np.uint8)
            fasc_mask = fasc_mask.astype(np.uint8)
            record['inference_source'] = 'cached_masks'
        else:
            apo_mask, apo_prob = infer_mask(image_path, apo_model, apo_side, device, amp)
            fasc_mask, fasc_prob = infer_mask(image_path, fasc_model, fasc_side, device, amp)
            record['inference_source'] = 'model'
        if apo_mask.shape != (actual['height'], actual['width']) or \
                fasc_mask.shape != (actual['height'], actual['width']):
            raise ValueError('inference mask shape mismatch')
        w0, h0 = actual['width'], actual['height']
        ao = measure_conservative(apo_mask, rule)
        ev = evaluate_mask_pair(fasc_mask, apo_mask, w0, h0, rule, rule_hash,
                                'prediction', apo_out=ao)
        fits = ev.get('apo_fits')
        cands = ev['candidates']
        by_label = {int(c['segment']['label_id']): c for c in cands}
        g0_sel = ev['selection']
        g1_sel, g1_adj = select_with_policy(cands, 'G1')
        g0_orig = [by_label[int(c['segment']['label_id'])] for c in g0_sel['fl']]
        g1_orig = [by_label[int(c['segment']['label_id'])] for c in g1_sel['fl']]
        image_apo_ok = ao.get('status') == 'candidate'
        mt_res = ao.get('result') or {}
        mt_px = mt_res.get('mt_mean_px')
        mt_values = mt_res.get('mt_px') or []
        mt_positions = mt_res.get('x_positions') or []
        mt_status = 'ok' if ao.get('status') == 'candidate' and mt_px is not None else 'failed'
        mt_reason = '' if mt_status == 'ok' else (ao.get('reason') or 'no_mt')
        record['parameters']['MT'] = {
            'status': mt_status, 'reason': mt_reason,
            'value_px': mt_px, 'value_mm': scale_value(mt_px, px_per_mm),
            'selected_positions_px': mt_positions,
            'selected_values_px': mt_values, 'selected_count': len(mt_values),
            'apo_status': ao.get('status'), 'apo_reason': ao.get('reason'),
        }
        pa_candidates = g0_sel['pa']
        pa_deg = g0_sel['pa_mean_deg']
        pa_rows = pa_details(pa_candidates, image_id)
        record['parameters']['PA'] = {
            'status': 'ok' if pa_deg is not None else 'failed',
            'reason': '' if pa_deg is not None else (g0_sel.get('pa_reason') or 'no_pa_candidate'),
            'value_deg': pa_deg, 'selected_count': len(pa_candidates),
            'candidate_count': sum(c.get('pa_status') == 'ok' for c in cands),
            'selected_pa_deg': [d['pa_deg'] for d in pa_rows],
            'selected_candidate_identities': [d['candidate_identity'] for d in pa_rows],
            'selected_details': pa_rows,
        }

        def fl_section(selected_orig, policy, experimental):
            details = selected_details(cands, selected_orig, fits, w0, h0,
                                       image_apo_ok, image_id, policy)
            values = [d.get('fl_px') for d in details if d.get('fl_px') is not None]
            mean = float(np.mean(values)) if values else None
            mm = [scale_value(v, px_per_mm) for v in values]
            return {
                'status': 'ok' if values else 'failed',
                'experimental': experimental,
                'reason': '' if values else f'no_valid_fl_candidate_{policy}',
                'selected_count': len(values),
                'selected_fl_px': values,
                'selected_fl_mm': mm,
                'selected_mean_px': mean,
                'selected_mean_mm': scale_value(mean, px_per_mm),
                'extrapolation_ratios': [d.get('extrapolation_ratio') for d in details],
                'outside_image_selected_count': sum(
                    bool(d.get('outside_image')) for d in details),
                'outside_support_selected_count': sum(
                    bool(d.get('outside_apo_support')) for d in details),
                'selected_invalid_count': sum(
                    d.get('validity_status') == 'invalid' for d in details),
                'selected_candidate_identities': [
                    d['candidate_identity'] for d in details],
                'selected_details': details,
            }

        record['parameters']['FL_G0'] = fl_section(g0_orig, 'G0', False)
        record['parameters']['FL_G1'] = fl_section(g1_orig, 'G1', True)
        record['parameters']['FL_G1']['label'] = 'experimental extrapolation output'

        record['validity'] = selected_invalid_summary(
            cands, pa_rows,
            record['parameters']['FL_G0']['selected_details'],
            record['parameters']['FL_G1']['selected_details'], image_id)
        record['validity']['n_evaluable'] = ev['n_fascicle_evaluable']
        record['candidate_counts'] = {
            'n_evaluable': ev['n_fascicle_evaluable'],
            'n_short': ev['n_fascicle_short'], 'n_crossing': ev['n_fascicle_crossing'],
            'n_pa_selected_g0': len(pa_candidates),
            'n_fl_selected_g0': len(g0_orig), 'n_fl_selected_g1': len(g1_orig),
        }
        fits_out = None
        if isinstance(fits, dict):
            fits_out = {'sup': [float(v) for v in fits['sup']],
                        'deep': [float(v) for v in fits['deep']],
                        'support': [float(v) for v in fits['support']]}
        record['apo'] = {
            'status': ao.get('status'), 'reason': ao.get('reason'),
            'source': ao.get('source'), 'rule_version': rule.get('version'),
            'rule_sha256': rule_hash,
            'support': fits_out['support'] if fits_out else None,
            'fits': fits_out,
        }
        record['_apo_mask'] = apo_mask
        record['_fasc_mask'] = fasc_mask
        for key, p in record['parameters'].items():
            if p['status'] != 'ok':
                record['failure_reasons'].append(f'{key}:{p["reason"]}')
        record['structure_and_values_unchanged'] = True
    except PolicyZeroMismatch as exc:
        return _failure_record(
            image_id, image_path, px_per_mm, scale_status, scale_source,
            f'policy_zero_mismatch:{exc}', 'policy_zero', expected_identity,
            record.get('actual_identity'), scale_reason=scale_reason,
            inference_source=record.get('inference_source', 'none'),
            scale_reader_version=scale_reader_version,
            scale_binding_status=scale_binding_status)
    except Exception as exc:
        return _failure_record(image_id, image_path, px_per_mm, scale_status,
                               scale_source, f'{type(exc).__name__}: {exc}',
                               'measurement', expected_identity,
                               record.get('actual_identity'),
                               scale_reason=scale_reason,
                               inference_source=record.get('inference_source', 'none'),
                               scale_reader_version=scale_reader_version,
                               scale_binding_status=scale_binding_status)
    return record

def load_eligible_records(path):
    """Load only identity and scale fields; expert values are never passed on."""
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    out = []
    for r in doc['records']:
        if r['image_id'] in EXCLUDED:
            raise ValueError('excluded image present in eligible list')
        has_scale = r.get('scale_pixel_per_cm') is not None
        px = None
        scale_ok = True
        scale_reason = ''
        if has_scale:
            try:
                raw = float(r['scale_pixel_per_cm'])
            except (TypeError, ValueError):
                scale_ok = False
                scale_reason = 'not_numeric'
            else:
                ok, scale, reason = validate_scale(raw / 10.0)
                scale_ok = ok
                scale_reason = reason
                px = scale if ok else None
        out.append({
            'image_id': r['image_id'],
            'path': r['path'],
            'width': int(r['width']), 'height': int(r['height']),
            'pixel_sha256': r['pixel_sha256'],
            'scale_pixel_per_cm': r.get('scale_pixel_per_cm'),
            'px_per_mm': px,
            'scale_status': ('provided' if has_scale and scale_ok
                             else ('invalid' if has_scale else 'unknown')),
            'scale_reason': scale_reason,
            'scale_source': 'eligible_dev.Scale_pixel_per_cm/10',
        })
    if len(out) != 34:
        raise ValueError(f'expected 34 eligible external images, got {len(out)}')
    if len({r['image_id'] for r in out}) != 34:
        raise ValueError('duplicate external image IDs')
    return out

def _unique_by_image_id(records, label):
    ids = [r['image_id'] for r in records]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f'duplicate {label} image_id: {duplicates}')
    return {r['image_id']: r for r in records}


def _is_nonempty_str(value):
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value):
    return (_is_nonempty_str(value) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _canonical_sha256(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(',', ':'),
                   ensure_ascii=False).encode('utf-8')).hexdigest()


SCALE_RESULTS_VERSION = 'round16_2_scan_results_v3'


def load_input_manifest(path, scale_results_path=None):
    """Load an identity manifest and strictly bound automatic scale results.

    Every top-level and record-level binding field must exist, be non-empty,
    have the correct type, and match the current frozen input and scale-reader
    implementation. Failures become per-image invalid scale records with null
    px/mm; a bad width/height or null field never aborts the batch.
    """
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    input_records = doc.get('records', [])
    for r in input_records:
        for key in ('image_id', 'path', 'width', 'height', 'pixel_sha256'):
            if key not in r:
                raise ValueError(f'input manifest record missing {key}')
        if not _is_nonempty_str(r.get('image_id')):
            raise ValueError('input manifest has empty image_id')
        if _positive_int(r.get('width')) is None:
            raise ValueError('input manifest has invalid width')
        if _positive_int(r.get('height')) is None:
            raise ValueError('input manifest has invalid height')
        if not _is_sha256(r.get('pixel_sha256')):
            raise ValueError('input manifest has invalid pixel_sha256')
    input_map = _unique_by_image_id(input_records, 'input manifest')
    scale_map = {}
    scale_meta = None
    if scale_results_path is not None:
        scale_doc = json.loads(
            Path(scale_results_path).read_text(encoding='utf-8'))
        scale_records = scale_doc.get('records', [])
        if not isinstance(scale_records, list):
            raise ValueError('scale results records must be a list')
        scale_map = _unique_by_image_id(scale_records, 'scale results')
        input_hash = sha(path)
        top_errors = []
        version = scale_doc.get('version')
        module_version = scale_doc.get('module_version')
        config = scale_doc.get('config')
        config_sha = scale_doc.get('config_sha256')
        top_input_hash = scale_doc.get('input_manifest_sha256')
        ocr_engine = scale_doc.get('ocr_engine')
        ocr_engine_version = scale_doc.get('ocr_engine_version')
        coord_version = scale_doc.get('ocr_coordinate_transform_version')
        if version != SCALE_RESULTS_VERSION:
            top_errors.append('version')
        if module_version != SCALE_MODULE_VERSION:
            top_errors.append('module_version')
        if not _is_sha256(config_sha):
            top_errors.append('config_sha256')
        if config_sha != SCALE_CONFIG_SHA256:
            top_errors.append('config_sha256_current')
        if not isinstance(config, dict):
            top_errors.append('config')
        else:
            if _canonical_sha256(config) != config_sha:
                top_errors.append('config_content_hash')
            if config.get('version') != module_version:
                top_errors.append('config_version')
        if not _is_sha256(top_input_hash):
            top_errors.append('input_manifest_sha256')
        elif top_input_hash != input_hash:
            top_errors.append('input_manifest_hash_mismatch')
        if ocr_engine != SCALE_CONFIG.get('ocr_engine'):
            top_errors.append('ocr_engine')
        if not _is_nonempty_str(ocr_engine_version):
            top_errors.append('ocr_engine_version')
        if coord_version != SCALE_CONFIG.get(
                'ocr_coordinate_transform_version'):
            top_errors.append('ocr_coordinate_transform_version')
        top_binding_ok = not top_errors
        scale_meta = {
            'path': str(Path(scale_results_path)),
            'sha256': sha(scale_results_path),
            'version': version,
            'module_version': module_version,
            'config_sha256': config_sha,
            'input_manifest_sha256': top_input_hash,
            'input_manifest_hash_match': bool(
                top_input_hash is not None and top_input_hash == input_hash),
            'top_binding_ok': bool(top_binding_ok),
            'top_binding_errors': top_errors,
            'duplicate_ids': [],
            'extra_ids': sorted(set(scale_map) - set(input_map)),
            'missing_ids': sorted(set(input_map) - set(scale_map)),
            'binding_rejections': [],
        }
    out = []
    for r in input_records:
        image_id = r['image_id']
        input_width = _positive_int(r['width'])
        input_height = _positive_int(r['height'])
        sr = scale_map.get(image_id)
        binding = {}
        binding_status = 'missing'
        if sr is None:
            status = 'unknown'
            px = None
            source = 'scale_result_missing'
            reason = 'scale_result_missing'
            reader_version = None
            binding_status = 'missing'
        else:
            required = ('image_id', 'pixel_sha256', 'width', 'height',
                        'module_version', 'config_sha256',
                        'recognition_implementation_sha256', 'ocr_source',
                        'input_manifest_sha256')
            missing = [k for k in required
                       if k not in sr or sr.get(k) is None]
            record_width = _positive_int(sr.get('width'))
            record_height = _positive_int(sr.get('height'))
            type_errors = []
            if record_width is None:
                type_errors.append('width')
            if record_height is None:
                type_errors.append('height')
            if not _is_nonempty_str(sr.get('image_id')):
                type_errors.append('image_id')
            if not _is_sha256(sr.get('pixel_sha256')):
                type_errors.append('pixel_sha256')
            if not _is_nonempty_str(sr.get('module_version')):
                type_errors.append('module_version')
            if not _is_sha256(sr.get('config_sha256')):
                type_errors.append('config_sha256')
            if not _is_sha256(sr.get('recognition_implementation_sha256')):
                type_errors.append('recognition_implementation_sha256')
            if not _is_nonempty_str(sr.get('ocr_source')):
                type_errors.append('ocr_source')
            if not _is_sha256(sr.get('input_manifest_sha256')):
                type_errors.append('input_manifest_sha256')
            if not _is_nonempty_str(sr.get('status')):
                type_errors.append('status')
            binding = {
                'missing_fields': missing,
                'type_errors': type_errors,
                'image_id_match': sr.get('image_id') == image_id,
                'pixel_sha256_match': sr.get('pixel_sha256')
                == r['pixel_sha256'],
                'width_match': (record_width is not None
                                and input_width is not None
                                and record_width == input_width),
                'height_match': (record_height is not None
                                 and input_height is not None
                                 and record_height == input_height),
                'module_version_match': (
                    _is_nonempty_str(sr.get('module_version'))
                    and sr.get('module_version') == SCALE_MODULE_VERSION),
                'config_sha256_match': (
                    _is_sha256(sr.get('config_sha256'))
                    and sr.get('config_sha256') == SCALE_CONFIG_SHA256),
                'implementation_sha256_match': (
                    sr.get('recognition_implementation_sha256')
                    == SCALE_IMPLEMENTATION_SHA256),
                'input_manifest_hash_match': bool(
                    scale_meta is not None
                    and scale_meta.get('top_binding_ok')
                    and sr.get('input_manifest_sha256') == scale_meta.get(
                        'input_manifest_sha256')),
                'top_binding_ok': bool(
                    scale_meta is not None
                    and scale_meta.get('top_binding_ok')),
            }
            binding_ok = (not missing and not type_errors
                          and all(v for k, v in binding.items()
                                  if k not in ('missing_fields',
                                               'type_errors')))
            status = sr.get('status')
            source = sr.get('source') or 'scale_reader'
            reason = sr.get('reason') or ''
            reader_version = sr.get('module_version')
            if not binding_ok:
                status = 'invalid'
                px = None
                binding_status = 'unbound' if (missing or type_errors) else 'mismatch'
                if scale_meta is not None and not scale_meta.get('top_binding_ok'):
                    reason = ('scale_top_binding_invalid:'
                              + ','.join(scale_meta.get('top_binding_errors') or []))
                elif missing or type_errors:
                    reason = 'scale_binding_missing_or_invalid:' + ','.join(
                        missing + type_errors)
                else:
                    reason = 'scale_binding_mismatch:' + ','.join(
                        k for k, v in binding.items()
                        if k not in ('missing_fields', 'type_errors') and not v)
                if scale_meta is not None:
                    scale_meta['binding_rejections'].append({
                        'image_id': image_id, 'reason': reason,
                        'binding': binding})
            elif status == 'ok':
                ok, value, why = validate_scale(sr.get('px_per_mm'))
                if ok and value is not None:
                    px = value
                    binding_status = 'verified'
                else:
                    status = 'invalid'
                    px = None
                    binding_status = 'verified'
                    reason = f'reader_ok_invalid_scale:{why or "missing"}'
            elif status in ('unknown', 'ambiguous', 'invalid'):
                px = None
                binding_status = 'verified'
            else:
                status = 'invalid'
                px = None
                binding_status = 'verified'
                reason = f'unknown_scale_status:{status}'
        out.append({
            'image_id': image_id,
            'path': r['path'],
            'width': input_width,
            'height': input_height,
            'pixel_sha256': r['pixel_sha256'],
            'px_per_mm': px,
            'scale_status': status,
            'scale_source': source,
            'scale_reason': reason,
            'scale_reader_version': reader_version,
            'scale_binding_status': binding_status,
            'scale_binding': binding,
        })
    return out, scale_meta


def _draw_extended_impl(record, pred_paths, apo_mask, fasc_mask, out_path):
    img = read_image_gray3(record['image_path'])
    h, w = img.shape[:2]
    sel = []
    for key in ('FL_G0', 'FL_G1'):
        for d in record['parameters'].get(key, {}).get('selected_details', []):
            for ptkey in ('intersection_sup', 'intersection_deep'):
                pt = d.get(ptkey)
                if pt is not None:
                    sel.append(pt)
    xs = [0, w - 1] + [float(p[0]) for p in sel]
    ys = [0, h - 1] + [float(p[1]) for p in sel]
    margin = 30
    minx = math.floor(min(xs)) - margin; maxx = math.ceil(max(xs)) + margin
    miny = math.floor(min(ys)) - margin; maxy = math.ceil(max(ys)) + margin
    cw = max(1, maxx - minx + 1); ch = max(1, maxy - miny + 1)
    canvas = Image.new('RGB', (cw, ch), (20, 20, 20))
    xoff, yoff = -minx, -miny
    canvas.paste(Image.fromarray(img), (xoff, yoff))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([xoff, yoff, xoff + w - 1, yoff + h - 1],
                   outline=(255, 255, 255), width=2)
    apo = record.get('apo') or {}
    def xy(p):
        return (float(p[0]) + xoff, float(p[1]) + yoff)
    # Selected apo fits and MT positions. Fits are stored in original coords.
    fits = apo.get('fits')
    if fits:
        sup = np.asarray(fits['sup'], dtype=float)
        deep = np.asarray(fits['deep'], dtype=float)
        xs_img = np.arange(w)
        ys_sup = np.polyval(sup, xs_img)
        ys_deep = np.polyval(deep, xs_img)
        for ys, color in ((ys_sup, (0, 220, 220)), (ys_deep, (255, 210, 0))):
            pts = [xy((x, y)) for x, y in zip(xs_img, ys) if np.isfinite(y)]
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
        for sx in fits.get('support') or []:
            draw.line([(xoff + sx, yoff), (xoff + sx, yoff + h - 1)],
                      fill=(130, 130, 130), width=1)
        mt = record['parameters'].get('MT', {})
        for x, ysup, ydeep in zip(mt.get('selected_positions_px') or [],
                                  np.interp(mt.get('selected_positions_px') or [],
                                            xs_img, ys_sup, left=np.nan, right=np.nan),
                                  np.interp(mt.get('selected_positions_px') or [],
                                            xs_img, ys_deep, left=np.nan, right=np.nan)):
            if np.isfinite(ysup) and np.isfinite(ydeep):
                draw.line([xy((x, ysup)), xy((x, ydeep))],
                          fill=(180, 180, 255), width=2)
    # PA selected lines and deep-boundary markers.
    for d in record['parameters'].get('PA', {}).get('selected_details', []):
        p0, p1 = d.get('p0'), d.get('p1')
        if p0 and p1:
            draw.line([xy(p0), xy(p1)], fill=(80, 160, 255), width=1)
        ideep = d.get('intersection_deep')
        if ideep:
            px, py = xy(ideep)
            draw.ellipse([px - 3, py - 3, px + 3, py + 3],
                         fill=(80, 160, 255))
        ctr = d.get('p0')
        if ctr:
            draw.text((xy(ctr)[0] + 2, xy(ctr)[1] - 10), 'PA', fill=(80, 160, 255))
    # Apo and selected lines are read from record metadata where available.
    for key, color in (('FL_G0', (0, 200, 0)), ('FL_G1', (255, 140, 0))):
        for d in record['parameters'].get(key, {}).get('selected_details', []):
            p0, p1 = d.get('p0'), d.get('p1')
            if p0 and p1:
                draw.line([xy(p0), xy(p1)], fill=color, width=2)
            isup, ideep = d.get('intersection_sup'), d.get('intersection_deep')
            if isup and ideep:
                draw.line([xy(isup), xy(ideep)], fill=color, width=1)
            for ptkey, cc in (('intersection_sup', (255, 255, 0)),
                              ('intersection_deep', (255, 0, 255))):
                pt = d.get(ptkey)
                if pt:
                    draw.ellipse([*[xy(pt)[0] - 4, xy(pt)[1] - 4],
                                  *[xy(pt)[0] + 4, xy(pt)[1] + 4]],
                                 outline=cc, width=2)
    # MT positions if present.
    mt = record['parameters'].get('MT', {})
    for x in mt.get('selected_positions_px') or []:
        draw.line([(xoff + x, yoff), (xoff + x, yoff + h - 1)],
                  fill=(180, 180, 255), width=1)
    label = (f"{record['image_id']} apo={record.get('apo', {}).get('status', '?')} "
             f"MT={mt.get('status', '?')} PA={record['parameters'].get('PA', {}).get('status', '?')} "
             f"G0={record['parameters'].get('FL_G0', {}).get('status', '?')} "
             f"G1={record['parameters'].get('FL_G1', {}).get('status', '?')} "
             f"fail={';'.join(record.get('failure_reasons', []))[:180]}")
    draw.text((6, 6), label, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {'overlay': str(out_path), 'extended_canvas': [cw, ch],
            'image_offset': [xoff, yoff],
            'selected_intersections': sel}


def _draw_extended(record, pred_paths, apo_mask, fasc_mask, out_path):
    try:
        info = _draw_extended_impl(record, pred_paths, apo_mask, fasc_mask, out_path)
        record['plot_status'] = 'ok'
        record['plot_info'] = info
        return info
    except Exception as exc:
        # Plot failure must not destroy already computed measurements.
        record['plot_status'] = 'failed'
        record['plot_error'] = f'{type(exc).__name__}: {exc}'
        placeholder = Image.new('RGB', (800, 200), (30, 30, 30))
        d = ImageDraw.Draw(placeholder)
        d.text((10, 10), f"plot failed for {record.get('image_id')}: "
                          f"{record['plot_error']}", fill=(255, 255, 255))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        placeholder.save(out_path)
        record['plot_info'] = {'overlay': str(out_path), 'placeholder': True}
        return record['plot_info']


def _resolve_dir(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _mask_paths(apo_mask_dir, fasc_mask_dir, image_id):
    return {
        'apo': _resolve_dir(apo_mask_dir) / f'{image_id}_apo.png',
        'fasc': _resolve_dir(fasc_mask_dir) / f'{image_id}_fasc.png',
    }


def _identity_status(record):
    actual = record.get('actual_identity') or {}
    expected = record.get('expected_identity')
    if not actual:
        return 'unreadable'
    if expected is None:
        return 'self_recorded'
    same = (actual.get('pixel_sha256') == expected.get('pixel_sha256')
            and actual.get('width') == expected.get('width')
            and actual.get('height') == expected.get('height'))
    return 'verified' if same else 'mismatch'


def _identity_manifest(records, mode, inference_mode):
    keep = ('image_id', 'image_path', 'width', 'height',
            'px_per_mm', 'input_px_per_mm', 'scale_status',
            'scale_source', 'scale_reason')
    rows = []
    for r in records:
        item = {k: r.get(k) for k in keep}
        item['pixel_sha256'] = (r.get('actual_identity') or {}).get('pixel_sha256')
        item['identity_status'] = _identity_status(r)
        rows.append(item)
    return {
        'version': 'round15_inference_manifest_v2',
        'mode': mode,
        'inference_mode': inference_mode,
        'n_records': len(rows),
        'records': rows,
        'hash_definition': "sha256(PIL Image.open(path).convert('L').tobytes())",
        'scale_formula': 'px_per_mm = Scale_pixel_per_cm / 10',
        'length_formula': 'length_mm = original_coordinate_length_px / px_per_mm',
        'pa_unit': 'degree',
        'expert_fields_passed_to_inference_selection_calibration': False,
        'note': ('This manifest contains identity and scale fields only. The '
                 'eligible JSON contains expert fields, but they are not passed '
                 'to inference, selection, tuning or calibration.'),
    }


def _walk_numbers(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numbers(v, k)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_numbers(v, key)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield key, obj


def validate_output_numbers(records):
    """Reject NaN/Infinity anywhere and negative *_mm values."""
    for rec in records:
        for key, value in _walk_numbers(rec):
            if not math.isfinite(float(value)):
                raise ValueError(
                    f'non-finite output value in {rec.get("image_id")} {key}')
            if key and key.endswith('_mm') and float(value) < 0:
                raise ValueError(
                    f'negative mm output in {rec.get("image_id")} {key}')


def sensitivity_summary(records):
    checks = {'G0': 0, 'G1': 0}
    mismatches = {'G0': 0, 'G1': 0}
    changed = {'G0': 0, 'G1': 0}
    sections = {'FL_G0': 0, 'FL_G1': 0}
    for rec in records:
        details = []
        for key in ('FL_G0', 'FL_G1'):
            part = (rec.get('parameters', {}).get(key, {})
                    .get('selected_details') or [])
            sections[key] += len(part)
            details.extend(part)
        for d in details:
            sens = d.get('direction_sensitivity') or {}
            for policy in ('G0', 'G1'):
                checks[policy] += 1
                if not sens.get(f'{policy.lower()}_zero_degree_equal', False):
                    mismatches[policy] += 1
                if sens.get(f'{policy.lower()}_status_changes_at'):
                    changed[policy] += 1
    return {'version': 'round15_unified_sensitivity_summary_v2',
            'selected_candidate_counts_by_section': sections,
            'policy_checks_on_all_selected_details': checks,
            'zero_degree_mismatches': mismatches,
            'selected_candidates_with_policy_status_change_at_pm1': changed,
            'note': ('Each check compares the original, recomputed 0 degree, '
                     'and both +/-1 degree perturbations under the same policy '
                     'and image-level apo status.')}


def _mm_values(parameter):
    for key in ('value_mm', 'selected_mean_mm'):
        yield parameter.get(key)
    for value in parameter.get('selected_fl_mm') or []:
        yield value


def build_verification(records, meta, sens):
    ids = [r.get('image_id') for r in records]
    binding_errors = []
    duplicate_g1 = []
    for rec in records:
        p = rec.get('parameters', {})
        px_per_mm = rec.get('px_per_mm')
        pa = p.get('PA') or {}
        if len(pa.get('selected_details') or []) != pa.get('selected_count'):
            binding_errors.append(f'{rec.get("image_id")}:PA detail count')
        for key in ('FL_G0', 'FL_G1'):
            fl = p.get(key) or {}
            dets = fl.get('selected_details') or []
            values = fl.get('selected_fl_px') or []
            mms = fl.get('selected_fl_mm') or []
            ratios = fl.get('extrapolation_ratios') or []
            if not (len(dets) == len(values) == len(mms) == len(ratios)):
                binding_errors.append(f'{rec.get("image_id")}:{key} array length')
                continue
            for d, v, mm, ratio in zip(dets, values, mms, ratios):
                if d.get('fl_px') != v:
                    binding_errors.append(
                        f'{rec.get("image_id")}:{key}:{d.get("candidate_identity")}:px')
                if d.get('extrapolation_ratio') != ratio:
                    binding_errors.append(
                        f'{rec.get("image_id")}:{key}:{d.get("candidate_identity")}:ratio')
                expected_mm = scale_value(v, px_per_mm)
                if mm != expected_mm:
                    binding_errors.append(
                        f'{rec.get("image_id")}:{key}:{d.get("candidate_identity")}:mm')
            fl_ids = [d.get('candidate_identity') for d in dets]
            if len(fl_ids) != len(set(fl_ids)):
                duplicate_g1.append(f'{rec.get("image_id")}:{key}')
    def count(key, status='ok'):
        return sum((r.get('parameters', {}).get(key) or {}).get('status') == status
                   for r in records)
    no_false_mm = all(
        value is None
        for r in records if r.get('px_per_mm') is None
        for key in ('MT', 'PA', 'FL_G0', 'FL_G1')
        for value in _mm_values((r.get('parameters') or {}).get(key) or {}))
    return {
        'version': 'round15_measurement_verification_v2',
        'n_records': len(records),
        'n_unique_image_ids': len(set(ids)),
        'excluded_ids_absent': sorted(EXCLUDED - set(ids)),
        'inference_mode': meta['inference_mode'],
        'identity_status_counts': dict(Counter(
            _identity_status(r) for r in records)),
        'scale_status_counts': dict(Counter(
            r.get('scale_status') for r in records)),
        'output_counts': {
            'apo_candidate': sum(
                (r.get('apo') or {}).get('status') == 'candidate' for r in records),
            'MT_ok': count('MT'), 'PA_ok': count('PA'),
            'FL_G0_ok': count('FL_G0'), 'FL_G1_ok': count('FL_G1')},
        'selected_counts': {
            key: sum((r.get('parameters', {}).get(key) or {}).get(
                'selected_count', 0) for r in records)
            for key in ('PA', 'FL_G0', 'FL_G1')},
        'selected_invalid_counts': {
            key: sum((r.get('validity', {}).get(
                'selected_invalid_counts') or {}).get(key, 0) for r in records)
            for key in ('PA', 'FL_G0', 'FL_G1')},
        'selected_invalid_union_unique': sum(
            r.get('validity', {}).get('n_selected_invalid_union_unique', 0)
            for r in records),
        'array_binding_errors': binding_errors,
        'duplicate_selected_identities': duplicate_g1,
        'sensitivity': sens,
        'plot_status_counts': dict(Counter(
            r.get('plot_status') for r in records)),
        'mask_save_status_counts': dict(Counter(
            r.get('mask_save_status') for r in records)),
        'no_false_mm': bool(no_false_mm),
        'status': ('passed' if not binding_errors and not duplicate_g1
                   and no_false_mm
                   and all(v == 0 for v in sens['zero_degree_mismatches'].values())
                   else 'failed'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=None, help='single image path (original image only)')
    ap.add_argument('--image-id', default=None)
    ap.add_argument('--px-per-mm', type=float, default=None,
                    help='optional scale; without it mm outputs are null')
    ap.add_argument('--eligible-dev', default=None,
                    help='batch eligible_dev.json; only identity/image/scale fields read')
    ap.add_argument('--input-manifest', default=None,
                    help='generic competition input identity manifest')
    ap.add_argument('--scale-results', default=None,
                    help='independent Round16 scale reader results for --input-manifest')
    ap.add_argument('--apo-checkpoint', default=str(DEFAULT_APO))
    ap.add_argument('--fasc-checkpoint', default=str(DEFAULT_FASC))
    ap.add_argument('--apo-mask-dir', default=None,
                    help='replay cached apo masks named {image_id}_apo.png')
    ap.add_argument('--fasc-mask-dir', default=None,
                    help='replay cached fascicle masks named {image_id}_fasc.png')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--device', default=None)
    ap.add_argument('--no-amp', action='store_true')
    args = ap.parse_args()
    selected_inputs = [x for x in (args.image, args.eligible_dev,
                                   args.input_manifest) if x is not None]
    if len(selected_inputs) != 1:
        raise ValueError('choose exactly one of --image, --eligible-dev or '
                         '--input-manifest')
    if args.scale_results is not None and args.input_manifest is None:
        raise ValueError('--scale-results requires --input-manifest')
    if (args.apo_mask_dir is None) != (args.fasc_mask_dir is None):
        raise ValueError('provide both --apo-mask-dir and --fasc-mask-dir, or neither')
    replay = args.apo_mask_dir is not None
    single_scale = None
    single_scale_status = 'unknown'
    single_scale_source = 'unknown'
    if args.image is not None:
        ok, single_scale, reason = validate_scale(args.px_per_mm)
        if not ok:
            raise ValueError(
                f'px_per_mm must be omitted or a finite positive number: {reason}')
        if single_scale is not None:
            single_scale_status = 'provided'
            single_scale_source = 'user'
    if args.image is None and args.px_per_mm is not None:
        raise ValueError('--px-per-mm is only for single-image mode')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    amp = not args.no_amp
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    rule, rule_hash = load_frozen_rule()
    if replay:
        apo_ck, apo_cfg, apo_arch, apo_enc = read_checkpoint(args.apo_checkpoint)
        fasc_ck, fasc_cfg, fasc_arch, fasc_enc = read_checkpoint(args.fasc_checkpoint)
        apo_model = fasc_model = None
    else:
        apo_model, apo_ck, apo_cfg = load_checkpoint_model(args.apo_checkpoint, device)
        fasc_model, fasc_ck, fasc_cfg = load_checkpoint_model(args.fasc_checkpoint, device)
        apo_arch = apo_ck.get('architecture') or apo_cfg.get('architecture')
        apo_enc = apo_ck.get('encoder_name') or apo_cfg.get('encoder_name')
        fasc_arch = fasc_ck.get('architecture') or fasc_cfg.get('architecture')
        fasc_enc = fasc_ck.get('encoder_name') or fasc_cfg.get('encoder_name')
    apo_side = int(apo_cfg.get('long_side', 512))
    fasc_side = int(fasc_cfg.get('long_side', 1024))
    records = []
    scale_meta = None
    if args.image is not None:
        image_id = args.image_id or Path(args.image).stem
        mask_paths = (_mask_paths(args.apo_mask_dir, args.fasc_mask_dir, image_id)
                      if replay else None)
        if mask_paths is not None:
            mask_paths = {'apo': str(mask_paths['apo']), 'fasc': str(mask_paths['fasc'])}
        records.append(measure_one(
            args.image, image_id, single_scale,
            apo_model, fasc_model, apo_side, fasc_side,
            rule, rule_hash, device, amp,
            scale_status=single_scale_status, scale_source=single_scale_source,
            mask_paths=mask_paths))
    else:
        scale_meta = None
        if args.eligible_dev is not None:
            batch_records = load_eligible_records(args.eligible_dev)
        else:
            batch_records, scale_meta = load_input_manifest(
                args.input_manifest, args.scale_results)
        for r in batch_records:
            mask_paths = None
            if replay:
                mp = _mask_paths(args.apo_mask_dir, args.fasc_mask_dir,
                                 r['image_id'])
                mask_paths = {'apo': str(mp['apo']), 'fasc': str(mp['fasc'])}
            records.append(measure_one(
                r['path'], r['image_id'], r['px_per_mm'],
                apo_model, fasc_model, apo_side, fasc_side,
                rule, rule_hash, device, amp,
                expected_identity={
                    'pixel_sha256': r['pixel_sha256'],
                    'width': r['width'], 'height': r['height']},
                scale_status=r['scale_status'],
                scale_source=r['scale_source'],
                scale_reason=r['scale_reason'],
                mask_paths=mask_paths,
                scale_reader_version=r.get('scale_reader_version'),
                scale_binding_status=r.get('scale_binding_status')))
    # Persist masks and overlays for every record, including failures where
    # inference may not have produced masks (then no mask is claimed).
    predictions = out / 'predictions'; predictions.mkdir(exist_ok=True)
    overlays = out / 'overlays'; overlays.mkdir(exist_ok=True)
    for rec in records:
        apo_mask = rec.get('_apo_mask')
        fasc_mask = rec.get('_fasc_mask')
        try:
            if apo_mask is not None:
                Image.fromarray((apo_mask * 255).astype(np.uint8)).save(
                    predictions / f'{rec["image_id"]}_apo.png')
            if fasc_mask is not None:
                Image.fromarray((fasc_mask * 255).astype(np.uint8)).save(
                    predictions / f'{rec["image_id"]}_fasc.png')
            if apo_mask is not None or fasc_mask is not None:
                rec['mask_save_status'] = 'ok'
        except Exception as exc:
            rec['mask_save_status'] = 'failed'
            rec['mask_save_error'] = f'{type(exc).__name__}: {exc}'
        rec.pop('_apo_mask', None); rec.pop('_fasc_mask', None)
    # Write measurements before drawing so a plotting failure cannot drop them.
    validate_output_numbers(records)
    (out / 'results.json').write_text(
        json.dumps(records, indent=2, allow_nan=False,
                   ensure_ascii=False), encoding='utf-8')
    for rec in records:
        try:
            _draw_extended(rec, {}, None, None, overlays / f'{rec["image_id"]}.png')
        except Exception as exc:
            rec['plot_status'] = 'failed'
            rec['plot_error'] = f'{type(exc).__name__}: {exc}'
            rec['plot_info'] = {'overlay': str(
                overlays / f'{rec["image_id"]}.png'), 'placeholder': True}
    validate_output_numbers(records)
    (out / 'results.json').write_text(
        json.dumps(records, indent=2, allow_nan=False,
                   ensure_ascii=False), encoding='utf-8')
    fields = ['image_id', 'image_path', 'width', 'height', 'identity_status',
              'px_per_mm', 'scale_status', 'scale_source', 'scale_reason',
              'scale_binding_status', 'scale_reader_version',
              'inference_source', 'apo_status',
              'mt_status', 'mt_px', 'mt_mm',
              'pa_status', 'pa_deg', 'pa_selected_count',
              'pa_selected_invalid_count',
              'fl_g0_status', 'fl_g0_count', 'fl_g0_mean_px',
              'fl_g0_mean_mm', 'fl_g0_selected_invalid_count',
              'fl_g1_status', 'fl_g1_count', 'fl_g1_mean_px',
              'fl_g1_mean_mm', 'fl_g1_experimental',
              'fl_g1_selected_invalid_count',
              'selected_invalid_union_unique',
              'plot_status', 'failure_reasons']
    with open(out / 'results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            p = r.get('parameters', {})
            v = r.get('validity') or {}
            w.writerow({
                'image_id': r['image_id'], 'image_path': r['image_path'],
                'width': r.get('width'), 'height': r.get('height'),
                'identity_status': _identity_status(r),
                'px_per_mm': r.get('px_per_mm'),
                'scale_status': r.get('scale_status'),
                'scale_source': r.get('scale_source'),
                'scale_reason': r.get('scale_reason'),
                'scale_binding_status': r.get('scale_binding_status'),
                'scale_reader_version': r.get('scale_reader_version'),
                'inference_source': r.get('inference_source'),
                'apo_status': (r.get('apo') or {}).get('status'),
                'mt_status': p.get('MT', {}).get('status'),
                'mt_px': p.get('MT', {}).get('value_px'),
                'mt_mm': p.get('MT', {}).get('value_mm'),
                'pa_status': p.get('PA', {}).get('status'),
                'pa_deg': p.get('PA', {}).get('value_deg'),
                'pa_selected_count': p.get('PA', {}).get('selected_count'),
                'pa_selected_invalid_count': (
                    v.get('selected_invalid_counts') or {}).get('PA'),
                'fl_g0_status': p.get('FL_G0', {}).get('status'),
                'fl_g0_count': p.get('FL_G0', {}).get('selected_count'),
                'fl_g0_mean_px': p.get('FL_G0', {}).get('selected_mean_px'),
                'fl_g0_mean_mm': p.get('FL_G0', {}).get('selected_mean_mm'),
                'fl_g0_selected_invalid_count': p.get(
                    'FL_G0', {}).get('selected_invalid_count'),
                'fl_g1_status': p.get('FL_G1', {}).get('status'),
                'fl_g1_count': p.get('FL_G1', {}).get('selected_count'),
                'fl_g1_mean_px': p.get('FL_G1', {}).get('selected_mean_px'),
                'fl_g1_mean_mm': p.get('FL_G1', {}).get('selected_mean_mm'),
                'fl_g1_experimental': p.get('FL_G1', {}).get('experimental'),
                'fl_g1_selected_invalid_count': p.get(
                    'FL_G1', {}).get('selected_invalid_count'),
                'selected_invalid_union_unique': v.get(
                    'n_selected_invalid_union_unique'),
                'plot_status': r.get('plot_status'),
                'failure_reasons': ';'.join(r.get('failure_reasons', [])),
            })
    # HTML gallery
    rows = []
    for r in records:
        p = r.get('parameters', {})
        rows.append(
            "<section><h2>{id}</h2><img src='overlays/{id}.png' style='max-width:100%;border:1px solid #444'>"
            "<pre>{summary}</pre></section>".format(
                id=html.escape(r['image_id']),
                summary=html.escape(json.dumps({
                    'image': r['image_path'],
                    'size': [r.get('width'), r.get('height')],
                    'identity_status': _identity_status(r),
                    'inference_source': r.get('inference_source'),
                    'scale': [r.get('px_per_mm'), r.get('scale_status'),
                              r.get('scale_source'), r.get('scale_reason')],
                    'apo': r.get('apo'),
                    'MT': p.get('MT'), 'PA': p.get('PA'),
                    'FL_G0': p.get('FL_G0'), 'FL_G1': p.get('FL_G1'),
                    'candidate_counts': r.get('candidate_counts'),
                    'validity': r.get('validity'),
                    'plot_status': r.get('plot_status'),
                    'failure_reasons': r.get('failure_reasons'),
                }, indent=2, ensure_ascii=False))))
    doc = ("<!doctype html><meta charset='utf-8'><title>UMUD fixed-model measurement</title>"
           "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}"
           "section{margin:24px 0;padding:12px;border:1px solid #333}"
           "pre{white-space:pre-wrap;font-size:12px;background:#191919;padding:8px}</style>"
           "<h1>UMUD fixed-model measurement prototype</h1>"
           "<p>B ResNet34-UNet + frozen apo. G1 is experimental. "
           "Every input record is included; failures are not dropped.</p>"
           + "".join(rows) + "</body>")
    (out / 'gallery.html').write_text(doc, encoding='utf-8')
    inference_mode = 'cached_masks' if replay else 'model'
    meta = {
        'version': 'round15_fixed_model_measurement_v2',
        'mode': ('single' if args.image is not None else
                 ('eligible_dev_batch' if args.eligible_dev is not None
                  else 'input_manifest_batch')),
        'inference_mode': inference_mode,
        'n_records': len(records),
        'eligible_dev': args.eligible_dev,
        'eligible_dev_sha256': sha(args.eligible_dev) if args.eligible_dev else None,
        'input_manifest': args.input_manifest,
        'input_manifest_sha256': sha(args.input_manifest) if args.input_manifest else None,
        'scale_results': args.scale_results,
        'scale_results_sha256': scale_meta['sha256'] if scale_meta else None,
        'scale_reader_version': (records[0].get('scale_reader_version')
                                 if records else None),
        'scale_binding_status_counts': dict(Counter(
            r.get('scale_binding_status') for r in records)),
        'scale_results_extra_ids': (scale_meta.get('extra_ids')
                                    if scale_meta else []),
        'scale_results_missing_ids': (scale_meta.get('missing_ids')
                                      if scale_meta else []),
        'scale_binding_rejections': (scale_meta.get('binding_rejections')
                                     if scale_meta else []),
        'scale_input_manifest_hash_match': (
            scale_meta.get('input_manifest_hash_match')
            if scale_meta else None),
        'scale_results_version': (scale_meta.get('version')
                                  if scale_meta else None),
        'scale_top_binding_ok': (scale_meta.get('top_binding_ok')
                                 if scale_meta else None),
        'scale_top_binding_errors': (
            scale_meta.get('top_binding_errors') if scale_meta else []),
        'excluded_ids': sorted(EXCLUDED),
        'model_combination': records[0].get('model_combination') if records else None,
        'apo_checkpoint': args.apo_checkpoint,
        'apo_checkpoint_sha256': sha(args.apo_checkpoint),
        'apo_architecture': apo_arch, 'apo_encoder': apo_enc,
        'fasc_checkpoint': args.fasc_checkpoint,
        'fasc_checkpoint_sha256': sha(args.fasc_checkpoint),
        'fasc_architecture': fasc_arch, 'fasc_encoder': fasc_enc,
        'apo_mask_dir': args.apo_mask_dir,
        'fasc_mask_dir': args.fasc_mask_dir,
        'apo_long_side': apo_side, 'fasc_long_side': fasc_side,
        'threshold': 0.5, 'restore': 'nearest',
        'device': device,
        'device_used_for_inference': not replay,
        'amp': amp, 'amp_used_for_inference': (not replay and amp and device == 'cuda'),
        'hash_definition': "sha256(PIL Image.open(path).convert('L').tobytes())",
        'scale_formula': 'px_per_mm = Scale_pixel_per_cm / 10',
        'length_formula': 'length_mm = original_coordinate_length_px / px_per_mm',
        'pa_unit': 'degree',
        'expert_values_passed_to_inference_selection_calibration': False,
        'eligible_json_contains_expert_fields': bool(args.eligible_dev),
        'expert_values_used': False,
        'no_training': True, 'no_ensemble': True,
        'no_threshold_change': True,
        'no_pa_fl_rule_change': True,
        'g1_default': False, 'g1_status': 'experimental',
    }
    (out / 'run_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    manifest = _identity_manifest(
        records,
        'single' if args.image is not None else 'eligible_dev_batch',
        inference_mode)
    (out / 'inference_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    sens = sensitivity_summary(records)
    (out / 'sensitivity_summary.json').write_text(
        json.dumps(sens, indent=2, ensure_ascii=False), encoding='utf-8')
    verification = build_verification(records, meta, sens)
    (out / 'verification.json').write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'n_records': len(records),
                      'inference_mode': inference_mode,
                      'status_counts': {
                          'apo_ok': sum((r.get('apo') or {}).get('status') == 'candidate'
                                        for r in records),
                          'mt_ok': sum((r.get('parameters') or {}).get('MT', {}).get('status') == 'ok'
                                       for r in records),
                          'pa_ok': sum((r.get('parameters') or {}).get('PA', {}).get('status') == 'ok'
                                       for r in records),
                          'fl_g0_ok': sum((r.get('parameters') or {}).get('FL_G0', {}).get('status') == 'ok'
                                          for r in records),
                          'fl_g1_ok': sum((r.get('parameters') or {}).get('FL_G1', {}).get('status') == 'ok'
                                          for r in records)}},
                     indent=2))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
