# -*- coding: utf-8 -*-
"""Direction-auxiliary dataset for Round26.

The segmentation input and mask path are exactly the Round13 data pipeline.
Direction targets come from frozen human fascicle components (see
scripts/round26_labels.py) and receive exactly the same horizontal flip,
anisotropic resize and padding as the image/mask pair. No reference values are
used.
"""
import json
import hashlib
import io
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.umud_baseline.data import (
    ROOT, IMAGENET_MEAN, IMAGENET_STD, UmudSegDataset,
    align_mask_to_image, read_image_gray3, read_mask, resize_long_side,
)


def load_label_index(root, split):
    path = Path(root) / 'label_audit/index.json'
    if not path.exists():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text(encoding='utf-8'))[split]
    out = {}
    for r in rows:
        key = (Path(r['image_path']).as_posix(), r['mask_pixel_sha256'])
        if key in out:
            raise ValueError(f'duplicate direction label identity: {key}')
        out[key] = r
    return out


def load_sparse_label(path):
    with np.load(path, allow_pickle=False) as z:
        coords, c, s = z['coords'], z['cos2'], z['sin2']
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.dtype.kind not in 'iu':
            raise ValueError('direction coordinates must be an integer N x 2 array')
        if c.shape != (len(coords),) or s.shape != (len(coords),):
            raise ValueError('direction array shape mismatch')
        return coords.astype(np.int64), c.astype(np.float32), s.astype(np.float32)


def validated_sparse_label(rec, shape):
    """Validate the bytes actually consumed. Missing is not empty supervision."""
    h, w = shape
    if (rec.get('height'), rec.get('width')) != (h, w):
        raise ValueError('direction label/image dimensions mismatch')
    blob = (Path(ROOT) / rec['label_path']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != rec.get('label_sha256'):
        raise ValueError('direction label file/hash mismatch')
    coords, c, s = load_sparse_label(io.BytesIO(blob))
    if len(coords) != rec.get('n_pixels'):
        raise ValueError('direction label count mismatch')
    if len(coords):
        if (coords < 0).any() or (coords[:, 0] >= h).any() or (coords[:, 1] >= w).any():
            raise ValueError('direction label coordinate out of range')
        if len(np.unique(coords, axis=0)) != len(coords):
            raise ValueError('duplicate direction label pixel')
        if not np.isfinite(c).all() or not np.isfinite(s).all() or not np.allclose(
                c * c + s * s, 1.0, atol=2e-5, rtol=0):
            raise ValueError('direction label vector must be finite and unit length')
    return coords, c, s


def vector_from_cos2_sin2(c, s):
    """Unit axis vector from cos(2t), sin(2t). Axis sign is arbitrary."""
    th = 0.5 * np.arctan2(s, c)
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)


def cos2_sin2_from_vector(vx, vy):
    n = vx * vx + vy * vy
    out_c = np.zeros_like(vx, dtype=np.float32)
    out_s = np.zeros_like(vy, dtype=np.float32)
    ok = n > 1e-12
    out_c[ok] = (vx[ok] * vx[ok] - vy[ok] * vy[ok]) / n[ok]
    out_s[ok] = (2.0 * vx[ok] * vy[ok]) / n[ok]
    return out_c, out_s


def transform_direction(cos2, sin2, valid, orig_wh, target_wh, flip):
    """Apply horizontal flip then anisotropic nearest resize.

    cos2/sin2 and valid are dense arrays in original image coordinates.
    """
    W, H = int(orig_wh[0]), int(orig_wh[1])
    nw, nh = int(target_wh[0]), int(target_wh[1])
    if flip:
        cos2 = cos2[:, ::-1].copy()
        sin2 = -sin2[:, ::-1].copy()
        valid = valid[:, ::-1].copy()
    if (W, H) != (nw, nh):
        def nearest(a):
            im = Image.fromarray(a)
            return np.asarray(im.resize((nw, nh), Image.NEAREST))
        cos2 = nearest(cos2.astype(np.float32))
        sin2 = nearest(sin2.astype(np.float32))
        valid = nearest(valid.astype(np.uint8)) > 0
    sx, sy = nw / float(W), nh / float(H)
    vx, vy = vector_from_cos2_sin2(cos2, sin2)
    c, s = cos2_sin2_from_vector(vx * sx, vy * sy)
    c = np.where(valid, c, 0.0).astype(np.float32)
    s = np.where(valid, s, 0.0).astype(np.float32)
    return c, s, valid.astype(np.uint8)


class DirectionSegDataset(UmudSegDataset):
    """Segmentation dataset + sparse component-direction aux targets."""

    def __init__(self, data, label_index, long_side=512, augment=False,
                 base_seed=20261017):
        super().__init__(data, long_side=long_side, augment=augment,
                         base_seed=base_seed)
        # Keys use the manifest image path and the selected mask pixel hash.
        self.label_index = label_index
        self.missing_labels = 0
        self.used_labels = 0

    def _labels_for(self, sample, mrec, shape):
        key = (Path(sample['image']['path']).as_posix(), mrec['pixel_sha256'])
        rec = self.label_index.get(key)
        H, W = shape
        cos2 = np.zeros((H, W), np.float32)
        sin2 = np.zeros((H, W), np.float32)
        valid = np.zeros((H, W), bool)
        if rec is None:
            self.missing_labels += 1
            raise ValueError(f'missing direction label record: {key}')
        coords, c, s = validated_sparse_label(rec, (H, W))
        if len(coords):
            y = coords[:, 0]
            x = coords[:, 1]
            cos2[y, x] = c
            sin2[y, x] = s
            valid[y, x] = True
        self.used_labels += 1
        return cos2, sin2, valid, rec

    def __getitem__(self, idx):
        s = self.data.samples[idx]
        img = read_image_gray3(os.path.join(ROOT, s['image']['path']))
        mrec = self._pick_variant(s)
        mask = read_mask(os.path.join(ROOT, mrec['path']))
        mask = align_mask_to_image(mask, (img.shape[1], img.shape[0]))
        orig_wh = (img.shape[1], img.shape[0])
        cos2, sin2, dvalid, lrec = self._labels_for(
            s, mrec, (img.shape[0], img.shape[1]))

        flip = False
        if self.augment:
            rng = random.Random(self._sample_key(s) + ':aug')
            flip = rng.random() < 0.5
            # Reproduce _augment exactly; direction only follows its geometry.
            if flip:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            from PIL import ImageEnhance
            im = Image.fromarray(img[:, :, 0])
            if rng.random() < 0.8:
                im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
                im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
            img = np.stack([np.asarray(im)] * 3, axis=-1)
            if rng.random() < 0.3:
                gamma = rng.uniform(0.8, 1.2)
                lut = (np.clip((np.arange(256) / 255.0) ** gamma, 0, 1) * 255).astype(np.uint8)
                img = lut[img]
            if rng.random() < 0.3:
                img = np.clip(img.astype(np.float32) + rng.gauss(0, 5), 0, 255).astype(np.uint8)
            if rng.random() < 0.2:
                k = 3
                pad = k // 2
                a = np.pad(img.astype(np.float32),
                           ((pad, pad), (pad, pad), (0, 0)), mode='edge')
                acc = np.zeros_like(img, dtype=np.float32)
                for dy in range(k):
                    for dx in range(k):
                        acc += a[dy:dy + img.shape[0], dx:dx + img.shape[1]]
                img = (acc / (k * k)).astype(np.uint8)

        img, scale = resize_long_side(img, self.long_side, is_mask=False)
        mask, _ = resize_long_side(mask, self.long_side, is_mask=True)
        target_wh = (mask.shape[1], mask.shape[0])
        cos2, sin2, dvalid = transform_direction(
            cos2, sin2, dvalid, orig_wh, target_wh, flip)
        canvas = self.long_side
        assert canvas % 32 == 0, 'long_side must be a multiple of 32'
        h0, w0 = mask.shape
        assert h0 <= canvas and w0 <= canvas
        img = np.pad(img, ((0, canvas - h0), (0, canvas - w0), (0, 0)))
        mask = np.pad(mask, ((0, canvas - h0), (0, canvas - w0)))
        dvalid = np.pad(dvalid, ((0, canvas - h0), (0, canvas - w0)))
        cos2 = np.pad(cos2, ((0, canvas - h0), (0, canvas - w0)))
        sin2 = np.pad(sin2, ((0, canvas - h0), (0, canvas - w0)))
        pad = (canvas - w0, canvas - h0)
        h, w = mask.shape
        valid = np.zeros((h, w), dtype=np.uint8)
        valid[:h - pad[1], :w - pad[0]] = 1

        if not img.flags.writeable:
            img = img.copy()
        if not mask.flags.writeable:
            mask = mask.copy()
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        valid_t = torch.from_numpy(valid).unsqueeze(0).float()
        dir_t = torch.from_numpy(np.stack([cos2, sin2], axis=0)).float()
        dvalid_t = torch.from_numpy(dvalid).unsqueeze(0).float()
        return {'image': img_t, 'mask': mask_t, 'valid': valid_t,
                'dir_target': dir_t, 'dir_valid': dvalid_t,
                'group_id': s['group_id'],
                'filename': s['image']['filename'],
                'mask_pixel_sha256': mrec['pixel_sha256'],
                'label_path': (lrec or {}).get('label_path'),
                'orig_wh': torch.tensor(orig_wh), 'scale': scale,
                'pad': torch.tensor(pad)}
