"""Round63 paths and binding helpers, with no imports from the default pipeline."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/round63_boundary_paths_v1'
R62 = ROOT / 'runs/round62_augmentation_quality_v1'
MODEL = R62 / 'training/20261023/C/epoch_06_model.pt'
SIDES = ('upper', 'lower')
ARMS = ('T', 'U', 'H', 'M', 'P', 'E')
LABELS = dict(T='硬阈值片段', U='连续概率片段', H='平滑重建',
              M='重建＋肌束选带', P='肌束＋软厚度先验', E='P＋有限延伸')


def j(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    s = json.dumps(clean(value), ensure_ascii=False, allow_nan=False)
    if path.suffix == '.gz':
        with gzip.open(temp, 'wt', encoding='utf-8') as f:
            f.write(s)
    else:
        temp.write_text(s + '\n', encoding='utf-8')
    temp.replace(path)


def readgz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def array_sha(a):
    h = hashlib.sha256()
    h.update(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def stats(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


def defaults():
    return dict(low=.18, high=.5, union_floor=.4, minimum_track=.02,
                minimum_anchor=.006, point_step=.007, max_column_gap=2,
                endpoint_window=.05, near_distance=.025, max_gap=.20,
                max_curvature=6., max_turn_deg=55., maximum_bulge=.025,
                corridor_probability=.10, corridor_hit_fraction=.25,
                image_hit_fraction=.60, max_paths=10, max_chain=4,
                min_pair_span=.12, min_thickness=.015,
                extension=.08, extension_blind_run=.015,
                ambiguity_margin=.025, prior_flat_mm=[15., 25.],
                prior_shoulder_mm=5., prior_maximum_penalty=.10)
