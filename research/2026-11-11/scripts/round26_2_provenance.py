"""Round26.2 entry gates. Frozen data and current implementation are separate."""
from pathlib import Path

from scripts.round26_1_common import (
    ROOT, OLD, NEW, j, sha, write_json, expected_inputs, validate_label_cache,
)

BASE = ROOT / 'runs/round26_2_closeout_v1'
DEFAULT_GATE = BASE / 'gate.json'
BINDING = ROOT / 'manifest/round26_2_sources.json'
CODE = (
    'scripts/round26_2_provenance.py', 'scripts/round26_1_common.py',
    'scripts/round26_1_train.py', 'scripts/round26_1_evaluate.py',
    'src/umud_baseline/direction_data.py', 'src/umud_baseline/direction_model.py',
    'src/umud_baseline/data.py', 'src/umud_baseline/train.py',
)
FILES = {
    'config': ROOT / 'configs/round26_1_direction_aux_corrected.json',
    'config_old': ROOT / 'configs/round26_direction_aux.json',
    'manifest': ROOT / 'manifest/manifest.csv',
    'splits': ROOT / 'manifest/splits_v2.csv',
    'isolation': ROOT / 'runs/geometry_eval/isolation_audit_v1/isolation_view.json',
    'input_identities': NEW / 'inputs/expected_input_identities.json',
    'label_index': OLD / 'label_audit/index.json',
    'label_summary': OLD / 'label_audit/summary_stage_b.json',
    'shared_init': OLD / 'shared_init.pt',
    'original_gate': NEW / 'stage_a_gate.json',
    'C0_checkpoint': OLD / 'training/C0/checkpoints/epoch_017.pt',
    'C0_manifest': OLD / 'predictions/C0/manifest.json',
    'C0_success': OLD / 'predictions/C0/prediction_success.json',
    'C0_meta': OLD / 'training/C0/run_meta.json',
    'C1_checkpoint': NEW / 'training/C1_corrected/checkpoints/epoch_017.pt',
    'C1_manifest': NEW / 'predictions/C1_corrected/manifest.json',
    'C1_success': NEW / 'predictions/C1_corrected/success.json',
    'C1_meta': NEW / 'training/C1_corrected/run_meta.json',
    'legacy_checkpoint': OLD / 'training/C1/checkpoints/epoch_017.pt',
    'legacy_manifest': OLD / 'predictions/C1/manifest.json',
    'legacy_success': OLD / 'predictions/C1/prediction_success.json',
    'legacy_meta': OLD / 'training/C1/run_meta.json',
    'legacy_regression': NEW / 'closeout/legacy_regression.json',
    'expert': ROOT / 'external/umud_expert/expert_measurements.json',
    'workbook': ROOT / ('external/umud_expert/unpacked/benchmark_dataset_architecture_v0.1.0/'
                        'Results_benchmark_architecture_v0.1.0.xlsx'),
}


def prepare_run_output(path):
    """Only a named run under runs/ or a temp directory, never source/history."""
    p = Path(path).resolve()
    for protected in (OLD, NEW, ROOT / 'data', ROOT / 'external', ROOT / 'manifest',
                      ROOT / 'src', ROOT / 'scripts', ROOT / 'tests', ROOT / 'review'):
        q = protected.resolve()
        if p == q or q in p.parents or p in q.parents:
            raise ValueError('refusing protected output directory: ' + str(p))
    runs = ROOT / 'runs'
    if ROOT in p.parents and (runs not in p.parents or p == runs or p == BASE):
        raise ValueError('output must be a separate run directory under runs/')
    p.mkdir(parents=True, exist_ok=True)
    for name in ('success.json', 'validation_failed.json', 'training_failed.json',
                 'engineering_check_success.json', 'validation_success.json'):
        marker = p / name
        if marker.exists():
            marker.unlink()
    return p


def verify_closeout_gate(gate_path=DEFAULT_GATE, *, labels=True):
    gate = j(gate_path)
    if gate.get('version') != 'round26_2_entry_gate_v1':
        raise ValueError('Round26.2 gate version missing/mismatch')
    source = ROOT / gate['source_binding_path']
    if sha(source) != gate.get('source_binding_sha256'):
        raise ValueError('source binding hash mismatch')
    binding = j(source)
    files = binding.get('files')
    if not isinstance(files, dict) or set(files) != set(FILES):
        raise ValueError('source binding file set mismatch')
    for key, record in files.items():
        if not isinstance(record, dict) or sha(ROOT / record['path']) != record.get('sha256'):
            raise ValueError('frozen source hash mismatch: ' + key)
    # The data loader consumes these canonical files, not an arbitrary same-name copy.
    for key in ('manifest', 'splits', 'isolation', 'config', 'config_old', 'input_identities',
                'shared_init', 'original_gate', 'C0_checkpoint', 'C1_checkpoint',
                'legacy_checkpoint', 'C0_manifest', 'C1_manifest', 'legacy_manifest',
                'C0_success', 'C1_success', 'legacy_success', 'expert', 'workbook',
                'legacy_regression'):
        if sha(FILES[key]) != files[key]['sha256']:
            raise ValueError('actual consumer source mismatch: ' + key)
    if set(gate.get('code_hashes', {})) != set(CODE):
        raise ValueError('implementation binding file set mismatch')
    for rel, want in gate['code_hashes'].items():
        if sha(ROOT / rel) != want:
            raise ValueError('implementation hash mismatch: ' + rel)
    original = j(FILES['original_gate'])
    for key, field in (('input_identities', 'input_identities_sha256'),
                       ('config', 'config_sha256'), ('shared_init', 'shared_init_sha256')):
        if files[key]['sha256'] != original[field]:
            raise ValueError('original gate binding mismatch: ' + key)
    meta = j(ROOT / files['C1_meta']['path'])
    for key, field in (('manifest', 'manifest_sha256'), ('splits', 'splits_sha256'),
                       ('config', 'config_sha256'), ('shared_init', 'shared_init_sha256'),
                       ('label_index', 'label_index_sha256')):
        if files[key]['sha256'] != meta.get(field):
            raise ValueError('training provenance mismatch: ' + key)
    doc = j(FILES['input_identities'])
    records = doc['records']
    identities = {(r['dataset'], r['image_id']): r for r in records}
    if len(records) != 316 or len(identities) != 316 or identities != expected_inputs():
        raise ValueError('actual input identities differ from frozen identities')
    label_root = (ROOT / files['label_index']['path']).parent.parent
    label_result = validate_label_cache(label_root) if labels else None
    return {**gate, 'files': files, 'label_root': label_root,
            'label_validation': label_result, 'original_gate_sha256': files['original_gate']['sha256']}


def write_current_gate():
    """Bind repaired code to the separately frozen source snapshot; never rewrites it."""
    write_json(DEFAULT_GATE, {
        'version': 'round26_2_entry_gate_v1',
        'source_binding_path': BINDING.relative_to(ROOT).as_posix(),
        'source_binding_sha256': sha(BINDING),
        'code_hashes': {rel: sha(ROOT / rel) for rel in CODE},
    })


if __name__ == '__main__':
    write_current_gate()
