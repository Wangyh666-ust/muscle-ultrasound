"""Read the frozen P/R boundary without replaying older experiments."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'runs/frozen_boundary_20261110'

def load_boundary(dataset,image_id,view='identity',variant='P'):
    if variant not in ('P','R','B'):raise ValueError('P=fixed permission baseline, R=parent reference, B=source evidence')
    path=FROZEN/'geometry'/dataset/Path(image_id).stem/(view+'.npz')
    with np.load(path,allow_pickle=False) as arrays:
        result={side:{k.removeprefix(variant+'_'+side+'_'):arrays[k].copy() for k in arrays.files if k.startswith(variant+'_'+side+'_')} for side in ('upper','lower')}
    return None if not result['upper'] else result
