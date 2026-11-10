"""Round30-only label and loss definitions. Does not change the default model."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import (UmudSegDataset, IMAGENET_MEAN, IMAGENET_STD,
                   read_image_gray3, resize_long_side, align_mask_to_image)
from .train import build_model

LABEL_VERSION = 'two_connected_traces_common_columns_v1'
LOSS_VERSION = 'masked_image_equal_channel_mean_bce_dice_v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def extract_identity(mask):
    """Conservative native-mask identity: exactly two connected traces.

    No model, fitted curve, interpolation, morphology or image intensity is used.
    Identity is only supervised at common columns with one contiguous run per
    component and consistent ordering. All other pixels/columns are ignored.
    More/fewer components make the whole variant identity-unknown.
    """
    a = np.asarray(mask, bool)
    if a.ndim != 2: raise ValueError('mask must be 2D')
    h, w = a.shape
    cc, n = label(a, np.ones((3, 3)))
    target = np.zeros((2, h, w), np.uint8)
    valid = np.zeros((h, w), np.uint8)
    meta = {'components': int(n), 'reason': None, 'common_columns': 0}
    if n != 2:
        meta['reason'] = 'component_count_not_two'
        return target, valid, meta
    pairs = []
    for x in range(w):
        ys = [np.flatnonzero(cc[:, x] == k) for k in (1, 2)]
        if any(not len(y) or np.any(np.diff(y) != 1) for y in ys): continue
        if ys[0][-1] < ys[1][0]: pairs.append((x, 1, 2))
        elif ys[1][-1] < ys[0][0]: pairs.append((x, 2, 1))
    if len(pairs) < max(20, .25 * w):
        meta['reason'] = 'insufficient_unambiguous_common_columns'
        return target, valid, meta
    if len({p[1] for p in pairs}) != 1:
        meta['reason'] = 'trace_order_switches'
        return target, valid, meta
    for x, su, de in pairs:
        target[0, :, x] = cc[:, x] == su
        target[1, :, x] = cc[:, x] == de
        valid[:, x] = 1
    assert not np.any(target.sum(axis=0) > 1)
    assert np.array_equal(target.sum(axis=0) > 0, a & valid.astype(bool))
    meta['common_columns'] = len(pairs)
    return target, valid, meta


def align_identity(target, valid, wh):
    return (np.stack([align_mask_to_image(c, wh) for c in target]),
            align_mask_to_image(valid, wh))


def masked_bce_dice(logits, target, valid):
    """Per image+channel valid-pixel BCE/Dice; channels then images equal.

    Images without supervision do not dilute the identity loss. Padding and
    unknowns contribute exactly zero gradient. Dice smoothing fixed at 1.
    """
    logits = logits.float(); target = target.float()
    v = valid.float().expand_as(logits)
    den = v.sum((2, 3))
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * v).sum((2, 3)) / den.clamp_min(1)
    p = logits.sigmoid() * v; t = target * v
    dice = 1 - (2 * (p*t).sum((2, 3)) + 1) / (p.sum((2, 3)) + t.sum((2, 3)) + 1)
    present = den > 0
    per_image = ((.5*bce + .5*dice) * present).sum(1) / present.sum(1).clamp_min(1)
    has = present.any(1)
    return (per_image * has).sum() / has.sum().clamp_min(1)


class BoundaryIdentityUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = build_model(pretrained=False)
        channels = self.base.decoder.blocks[-1].conv2[0].out_channels
        self.identity_head = nn.Conv2d(channels, 2, 1)

    def forward(self, x):
        z = self.base.decoder(self.base.encoder(x))
        return self.base.segmentation_head(z), self.identity_head(z)

    def configure_arm(self, arm):
        if arm not in ('T0', 'T1'): raise ValueError('unknown training arm')
        for p in self.identity_head.parameters(): p.requires_grad_(arm == 'T1')
        return [p for p in self.parameters() if p.requires_grad]


class IdentityDataset(Dataset):
    def __init__(self, root, samples, long_side=512, seed=20261023, augment=True):
        self.root = Path(root); self.samples = samples; self.long_side = long_side
        self.base_seed = seed; self.augment = augment; self.epoch = 0

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return len(self.samples)

    def choice(self, sample):
        key = f'{self.base_seed}:{self.epoch}:{sample["sampling_key"]}'
        vi = random.Random(key + ':variant').randrange(len(sample['variants'])) if len(sample['variants']) > 1 else 0
        return key, vi

    def __getitem__(self, i):
        sample = self.samples[i]; key, vi = self.choice(sample)
        image = read_image_gray3(self.root / sample['image_path'])
        cache = np.load(self.root / sample['variants'][vi]['cache_path'])
        layers = np.concatenate([cache['union'][None], cache['target'], cache['valid'][None]], axis=0).transpose(1, 2, 0)
        if self.augment:
            image, layers = UmudSegDataset._augment(self, image, layers, random.Random(key + ':aug'))
        image, _ = resize_long_side(image, self.long_side, False)
        ls = [resize_long_side(layers[:, :, c], self.long_side, True)[0] for c in range(4)]
        h, w = image.shape[:2]; size = self.long_side
        image = np.pad(image, ((0,size-h),(0,size-w),(0,0)))
        ls = np.stack([np.pad(a, ((0,size-h),(0,size-w))) for a in ls])
        v = np.zeros((1,size,size),np.float32); v[:,:h,:w] = 1
        x = torch.from_numpy(image.copy()).permute(2,0,1).float()/255
        x = (x-torch.tensor(IMAGENET_MEAN)[:,None,None])/torch.tensor(IMAGENET_STD)[:,None,None]
        return {'image':x, 'union':torch.from_numpy(ls[:1].copy()).float(),
                'identity':torch.from_numpy(ls[1:3].copy()).float(),
                'identity_valid':torch.from_numpy(ls[3:4].copy()).float(),
                'valid':torch.from_numpy(v), 'index':i, 'variant':vi}
