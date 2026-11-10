"""Shared source, identity and isolation gates for Round26.1."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/round26_direction_aux_v1'
NEW = ROOT / 'runs/round26_1_direction_aux_corrected_v1'
CONFIG_OLD = ROOT / 'configs/round26_direction_aux.json'
CONFIG_NEW = ROOT / 'configs/round26_1_direction_aux_corrected.json'
ISOLATION = ROOT / 'runs/geometry_eval/isolation_audit_v1/isolation_view.json'
OLD_C0_CKPT = OLD / 'training/C0/checkpoints/epoch_017.pt'
OLD_C0_SHA = 'a27a434b302163bf4f039caf6b2e963699ef98c568c023d75e61b61d773b863b'
OLD_LEGACY_C1_CKPT = OLD / 'training/C1/checkpoints/epoch_017.pt'
OLD_LEGACY_C1_SHA = '64f8c281b87e599fce882992b39cec572377bf44fbfb5cb6225f8989c46dddbd'
SHARED_INIT = OLD / 'shared_init.pt'
SHARED_INIT_SHA = '31a6451b5d325d27bf0c970a8da1bbe2db570503d6e0f43a694234a5076279df'
PROTOCOL = {'long_side': 1024, 'threshold': 0.5, 'restore': 'nearest',
            'input': 'gray replicated 3ch, ImageNet norm', 'tta': False,
            'amp': True}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def j(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=True),
                          encoding='utf-8')


def tree_hash(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob('*')):
        if p.is_file():
            h.update(str(p.relative_to(path)).replace('\\', '/').encode())
            h.update(b'\0')
            h.update(p.read_bytes())
    return h.hexdigest()


def expected_inputs():
    """Frozen dev282 + external34 identities and source image hashes."""
    from src.umud_baseline.data import UmudTrainData, read_image_gray3
    from scripts.round21_boundary_convention import parse_external
    inc = set(j(ISOLATION)['included_filenames'])
    data = UmudTrainData('fasc', 'dev', include_filenames=inc)
    if len(data.samples) != 282 or len({s['group_id'] for s in data.samples}) != 279:
        raise ValueError('dev282/279 source mismatch')
    dev = {}
    for s in data.samples:
        r = s['image']
        im = read_image_gray3(ROOT / r['path'])
        pix = hashlib.sha256(im[:, :, 0].tobytes()).hexdigest()
        if pix != r['pixel_sha256']:
            raise ValueError('dev source image pixel hash mismatch: ' + r['filename'])
        dev[('dev282', r['filename'])] = {
            'dataset': 'dev282', 'image_id': r['filename'],
            'group_id': s['group_id'], 'image_path': Path(r['path']).as_posix(),
            'width': int(r['width']), 'height': int(r['height']),
            'pixel_sha256': pix}
    ext = {}
    for r in parse_external():
        path = ROOT / r['path'] if not Path(r['path']).is_absolute() else Path(r['path'])
        im = read_image_gray3(path)
        pix = hashlib.sha256(im[:, :, 0].tobytes()).hexdigest()
        want = r['frozen'].get('pixel_sha256')
        if want is not None and pix != want:
            raise ValueError('external source image pixel hash mismatch')
        ext[('external34', r['image_id'])] = {
            'dataset': 'external34', 'image_id': r['image_id'],
            'group_id': r['group_id'],
            'image_path': Path(r['path']).as_posix(),
            'width': int(r['width']), 'height': int(r['height']),
            'pixel_sha256': pix}
    out = {**dev, **ext}
    if len(out) != 316 or Counter(r['dataset'] for r in out.values()) != {
            'dev282': 282, 'external34': 34}:
        raise ValueError('frozen input identity count mismatch')
    if ('external34', 'im_07_arch') in ext:
        raise ValueError('excluded external image present')
    return out


def validate_prediction_arm(arm, base, expected, config_path,
                            checkpoint_path, checkpoint_sha,
                            training_meta_path, success_path,
                            expected_config_sha=None,
                            allow_legacy=False):
    """Validate actual prediction inputs before producing any result."""
    base = Path(base)
    manifest_path = base / 'manifest.json'
    success_path = Path(success_path)
    if not manifest_path.exists() or not success_path.exists():
        raise ValueError(f'{arm}: manifest or success missing')
    manifest_sha = sha(manifest_path)
    success = j(success_path)
    if success.get('manifest_sha256') != manifest_sha:
        raise ValueError(f'{arm}: success/manifest hash mismatch')
    if success.get('arm') != arm:
        raise ValueError(f'{arm}: success arm mismatch')
    m = j(manifest_path)
    if m.get('reference_values_read') is not False or \
            m.get('expert_values_read') is not False:
        raise ValueError(f'{arm}: prediction read forbidden sources')
    if m.get('frozen_before_evaluation') is not True:
        raise ValueError(f'{arm}: prediction not marked frozen')
    if m.get('checkpoint') != Path(checkpoint_path).relative_to(ROOT).as_posix():
        raise ValueError(f'{arm}: manifest checkpoint path mismatch')
    if sha(checkpoint_path) != checkpoint_sha or m.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError(f'{arm}: checkpoint identity mismatch')
    ck = __import__('torch').load(checkpoint_path, map_location='cpu',
                                  weights_only=False)
    if int(ck.get('epoch', -1)) != 17:
        raise ValueError(f'{arm}: checkpoint epoch != 17')
    if ck.get('arm') != arm and not (allow_legacy and arm.startswith('legacy_')):
        raise ValueError(f'{arm}: checkpoint arm mismatch')
    cfg = j(config_path)
    cfg_sha = sha(config_path)
    if expected_config_sha is not None and cfg_sha != expected_config_sha:
        raise ValueError(f'{arm}: config hash mismatch')
    if ck.get('config_sha256') != cfg_sha or m.get('config_sha256') != cfg_sha:
        raise ValueError(f'{arm}: checkpoint/manifest config hash mismatch')
    if ck.get('shared_init_sha256') != SHARED_INIT_SHA or \
            sha(SHARED_INIT) != SHARED_INIT_SHA:
        raise ValueError(f'{arm}: shared initialization mismatch')
    if m.get('protocol') != PROTOCOL:
        raise ValueError(f'{arm}: prediction protocol mismatch')
    if not Path(training_meta_path).exists():
        raise ValueError(f'{arm}: training metadata missing')
    meta = j(training_meta_path)
    required_meta = {'arm': ck['arm'], 'config_sha256': cfg_sha,
                     'shared_init_sha256': SHARED_INIT_SHA,
                     'splits_sha256': sha(ROOT / 'manifest/splits_v2.csv'),
                     'manifest_sha256': sha(ROOT / 'manifest/manifest.csv'),
                     'seed': cfg['seed'], 'lambda_dir': ck['lambda_dir'],
                     'eval_epochs': cfg['eval_epochs'],
                     'no_early_stopping': True, 'no_lr_scheduler': True}
    for field, want in required_meta.items():
        if meta.get(field) != want:
            raise ValueError(f'{arm}: training metadata mismatch: {field}')
    if ck.get('config') != cfg or ck.get('splits_sha256') != required_meta['splits_sha256']:
        raise ValueError(f'{arm}: checkpoint config/splits content mismatch')
    if 'config' in meta and meta['config'] != cfg:
        raise ValueError(f'{arm}: training metadata config content mismatch')
    label_field = 'label_index_sha256' if arm == 'C1_corrected' else 'direction_label_train_index_sha256'
    if meta.get(label_field) != sha(OLD / 'label_audit/index.json'):
        raise ValueError(f'{arm}: training label index mismatch')
    if arm == 'C1_corrected':
        for field in ('gate_sha256', 'loss_definition', 'diagnostic_version'):
            if meta.get(field) is None or meta[field] != ck.get(field):
                raise ValueError(f'{arm}: training metadata mismatch: {field}')
    records = m.get('records')
    if not isinstance(records, list) or len(records) != 316:
        raise ValueError(f'{arm}: prediction record count != 316')
    seen = set()
    by_key = {}
    for r in records:
        key = (r.get('dataset'), r.get('image_id'))
        if key in seen:
            raise ValueError(f'{arm}: duplicate prediction identity {key}')
        seen.add(key)
        if key not in expected:
            raise ValueError(f'{arm}: extra/unknown prediction identity {key}')
        e = expected[key]
        if r.get('group_id') != e['group_id']:
            raise ValueError(f'{arm}: group mismatch {key}')
        if Path(r.get('image_path', '')).as_posix() != e['image_path']:
            raise ValueError(f'{arm}: source path mismatch {key}')
        if (int(r['image_width']), int(r['image_height'])) != (e['width'], e['height']):
            raise ValueError(f'{arm}: source dimensions mismatch {key}')
        mp = base / r['mask_path']
        if not mp.exists() or sha(mp) != r['mask_sha256']:
            raise ValueError(f'{arm}: mask hash mismatch {key}')
        with Image.open(mp) as im:
            if im.size != (e['width'], e['height']):
                raise ValueError(f'{arm}: mask dimensions mismatch {key}')
        if r.get('threshold') != 0.5 or r.get('restore') != 'nearest' or \
                r.get('checkpoint_sha256') != checkpoint_sha:
            raise ValueError(f'{arm}: per-mask protocol/checkpoint mismatch {key}')
        by_key[key] = r
    if set(by_key) != set(expected):
        missing = sorted(set(expected) - set(by_key))
        raise ValueError(f'{arm}: missing prediction identities: {missing[:5]}')
    return {'arm': arm, 'manifest_sha256': manifest_sha,
            'config_sha256': cfg_sha, 'checkpoint_sha256': checkpoint_sha,
            'n_records': len(records), 'records': by_key}


def validate_label_cache(label_root, expected_splits=('train', 'dev')):
    """Validate all actual label files and records; zero labels stay valid."""
    from src.umud_baseline.data import UmudTrainData
    from src.umud_baseline.direction_data import validated_sparse_label
    label_root = Path(label_root)
    inc = set(j(ISOLATION)['included_filenames'])
    index = j(label_root / 'label_audit/index.json')
    out = {}
    for split in expected_splits:
        data = UmudTrainData('fasc', split,
                             include_filenames=inc if split == 'dev' else None)
        rows = index.get(split)
        if not isinstance(rows, list):
            raise ValueError(f'label {split} index missing')
        required = {(Path(sm['image']['path']).as_posix(), m['pixel_sha256']): (sm, m)
                    for sm in data.samples for m in sm['masks']}
        seen = set()
        for r in rows:
            key = (r['image_path'], r['mask_pixel_sha256'])
            if key in seen:
                raise ValueError(f'duplicate label identity {key}')
            seen.add(key)
            if key not in required:
                raise ValueError(f'{split}: extra label identity {key}')
            sm, m = required[key]
            if r.get('filename') != sm['image']['filename'] or r.get('group_id') != sm['group_id']:
                raise ValueError(f'label identity/group mismatch {key}')
            if Path(r.get('mask_path', '')).as_posix() != Path(m['path']).as_posix():
                raise ValueError(f'label source mask path mismatch {key}')
            validated_sparse_label(r, (int(sm['image']['height']), int(sm['image']['width'])))
        # Actual training/dev entry: every selected variant must have a record;
        # a record with zero pixels is a valid zero-supervision variant.
        if set(required) != seen:
            raise ValueError(f'{split}: label records do not match selected variants')
        out[split] = {'n_records': len(rows), 'n_required': len(required),
                      'n_zero_variants': sum(r['n_pixels'] == 0 for r in rows),
                      'zero_images': sorted({r['filename'] for r in rows
                                             if r['n_pixels'] == 0})}
    return out


def assert_not_protected(out):
    p = Path(out).resolve()
    protected = OLD.resolve()
    if p == protected or protected in p.parents:
        raise ValueError('refusing protected old output directory')
    return p


def prepare_output(out):
    """Reject protected paths and clear only this run's stale markers."""
    out = assert_not_protected(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('success.json', 'validation_failed.json'):
        p = out / name
        if p.exists():
            p.unlink()
    return out


def atomic_write_success(out, payload):
    import os
    out = Path(out)
    tmp = out / 'success.json.tmp'
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True),
                   encoding='utf-8')
    os.replace(tmp, out / 'success.json')


def mark_failure(out, error):
    """A failed actual entry must not leave a success marker."""
    out = Path(out)
    write_json(out / 'validation_failed.json',
               {'error': str(error), 'success_written': False})
    p = out / 'success.json'
    if p.exists():
        p.unlink()
