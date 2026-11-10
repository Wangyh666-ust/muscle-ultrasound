# -*- coding: utf-8 -*-
"""Data pipeline for UMUD baseline B.

Key properties (per KIMI_BASELINE_HANDOFF.txt section 四):
- mask read: PIL decode, convert('L') > 127 (P-mode palettes handled by PIL)
- mask anisotropically resized to IMAGE coordinates (nearest) before transforms
- image+mask share identical geometric transforms; padding excluded via valid mask
- exact duplicate image+mask copies collapse to one sample; same-image-different-mask
  variants: ONE variant per epoch (seeded by epoch, chosen per training image)
- near-duplicate group sampling weight = 1/n_train_images_in_group
- (task, role, filename) keyed manifest index; image and mask sizes kept separate
"""
import os, csv, json, random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_DIR = {
    'fasc': 'data/fasc_imgs_v1/fasc_images_new_model_v1',
    'apo': 'data/apo_imgs_v1/apo_images_new_model_v1',
}


def load_manifest_index():
    """(task, role, filename) -> record with real paths and sizes."""
    idx = {}
    with open(os.path.join(ROOT, 'manifest/manifest.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            idx[(r['task'], r['role'], r['filename'])] = r
    return idx


def load_split(task, split):
    """rows of splits_v2.csv for (task, split)."""
    out = []
    with open(os.path.join(ROOT, 'manifest/splits_v2.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['task'] == task and r['split'] == split:
                out.append(r)
    return out


def read_mask(path):
    """PIL decode -> uint8 {0,1}. convert('L')>127 verified vs palette for all
    P-mode apo masks (manifest/AUDIT_SUMMARY.md section 2)."""
    with Image.open(path) as im:
        return (np.asarray(im.convert('L')) > 127).astype(np.uint8)


def read_image_gray3(path):
    """Gray-scale content replicated to 3 channels (checked: sources are gray)."""
    with Image.open(path) as im:
        g = np.asarray(im.convert('L'), dtype=np.uint8)
    return np.stack([g, g, g], axis=-1)


def align_mask_to_image(mask, img_wh):
    """Anisotropic resize of mask to image (W,H), nearest. Audit-supported mapping."""
    h, w = mask.shape
    W, H = img_wh
    if (w, h) == (W, H):
        return mask
    im = Image.fromarray(mask * 255)
    im = im.resize((W, H), Image.NEAREST)
    return (np.asarray(im) > 127).astype(np.uint8)


def resize_long_side(arr, long_side, is_mask):
    """Resize np array keeping aspect so long side == long_side."""
    h, w = arr.shape[:2]
    scale = long_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if arr.ndim == 3:
        im = Image.fromarray(arr)
        im = im.resize((nw, nh), Image.BILINEAR)
        return np.asarray(im), scale
    im = Image.fromarray(arr * 255)
    im = im.resize((nw, nh), Image.NEAREST)
    return (np.asarray(im) > 127).astype(np.uint8), scale


def pad_to_multiple(arr, multiple=32, pad_value=0, is_mask=True):
    h, w = arr.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if arr.ndim == 3:
        out = np.full((h + ph, w + pw, arr.shape[2]), pad_value, dtype=arr.dtype)
    else:
        out = np.full((h + ph, w + pw), pad_value, dtype=arr.dtype)
    out[:h, :w] = arr
    return out, (pw, ph)


class UmudTrainData:
    """Builds unique-image training samples with mask variants + group weights."""

    def __init__(self, task, split='train', manifest_index=None,
                 include_filenames=None):
        assert split in ('train', 'dev')
        self.task = task
        m = manifest_index or load_manifest_index()
        rows = load_split(task, split)
        include_set = (set(include_filenames)
                       if include_filenames is not None else None)
        # group duplicate images (same pixel content) and collect mask variants
        by_group = {}
        for r in rows:
            by_group.setdefault(r['group_id'], []).append(r)
        self.samples = []   # dict: group_id, image rec, mask recs (variants)
        self.group_sizes = {}
        for gid, rs in sorted(by_group.items()):
            # exact dedup: identical (pixel content AND mask content) -> one copy
            uniq = {}
            for r in rs:
                irec = m[(task, 'image', r['filename'])]
                mrec = m[(task, 'mask', r['filename'])]
                ikey = irec['pixel_sha256']
                mkey = mrec['pixel_sha256']
                if ikey not in uniq:
                    uniq[ikey] = {'group_id': gid, 'image': irec, 'masks': []}
                if not any(v['pixel_sha256'] == mkey for v in uniq[ikey]['masks']):
                    uniq[ikey]['masks'].append(mrec)
            if include_set is not None:
                uniq = {k: v for k, v in uniq.items()
                        if v['image']['filename'] in include_set}
            if uniq:
                self.group_sizes[gid] = len(uniq)
                self.samples.extend(uniq.values())
        self.split = split
        self.include_filenames = (sorted(include_set)
                                  if include_set is not None else None)

    def sampler_weights(self):
        """1/n_images_in_group so every near-dup group has equal expected weight."""
        return [1.0 / self.group_sizes[s['group_id']] for s in self.samples]

    def __len__(self):
        return len(self.samples)


class UmudSegDataset(Dataset):
    def __init__(self, data: UmudTrainData, long_side=512, augment=False, base_seed=20261017):
        self.data = data
        self.long_side = long_side
        self.augment = augment
        self.epoch = 0
        self.base_seed = base_seed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _sample_key(self, sample):
        """Deterministic per-(seed, epoch, image) key; independent of call
        order and of any shared/global RNG state (reproducible under multiple
        DataLoader workers, which each receive a pickled copy of the dataset)."""
        return f'{self.base_seed}:{self.epoch}:{sample["image"]["path"]}'

    def _pick_variant(self, sample):
        """Variant index determined solely by (seed, epoch, image_id): same
        image -> same variant within an epoch regardless of worker or order."""
        masks = sample['masks']
        if len(masks) == 1:
            return masks[0]
        idx = random.Random(self._sample_key(sample) + ':variant').randrange(len(masks))
        return masks[idx]

    def __len__(self):
        return len(self.data.samples)

    def __getitem__(self, idx):
        s = self.data.samples[idx]
        img = read_image_gray3(os.path.join(ROOT, s['image']['path']))
        mrec = self._pick_variant(s)
        mask = read_mask(os.path.join(ROOT, mrec['path']))
        mask = align_mask_to_image(mask, (img.shape[1], img.shape[0]))
        orig_wh = (img.shape[1], img.shape[0])

        if self.augment:
            # independent stream: augmentation must not interact with the
            # variant-selection stream; keyed per (seed, epoch, image)
            img, mask = self._augment(img, mask,
                                      random.Random(self._sample_key(s) + ':aug'))

        img, scale = resize_long_side(img, self.long_side, is_mask=False)
        mask, _ = resize_long_side(mask, self.long_side, is_mask=True)
        # uniform square canvas so batches are collatable; long_side must be
        # divisible by 32 for the UNet; right/bottom padding excluded via valid
        canvas = self.long_side
        assert canvas % 32 == 0, 'long_side must be a multiple of 32'
        h0, w0 = mask.shape
        assert h0 <= canvas and w0 <= canvas
        img = np.pad(img, ((0, canvas - h0), (0, canvas - w0), (0, 0)))
        mask = np.pad(mask, ((0, canvas - h0), (0, canvas - w0)))
        pad = (canvas - w0, canvas - h0)
        h, w = mask.shape
        valid = np.zeros((h, w), dtype=np.uint8)
        valid[:h - pad[1], :w - pad[0]] = 1

        # read-only numpy views (e.g. tifffile slices) yield non-writable
        # storages that default_collate cannot resize — force writable copies
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
        return {'image': img_t, 'mask': mask_t, 'valid': valid_t,
                'group_id': s['group_id'],
                'filename': s['image']['filename'],
                'orig_wh': torch.tensor(orig_wh), 'scale': scale, 'pad': torch.tensor(pad)}

    def _augment(self, img, mask, rng):
        if rng.random() < 0.5:  # horizontal flip only (no vertical, per plan 7.3)
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        im = Image.fromarray(img[:, :, 0])
        if rng.random() < 0.8:
            im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
            im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
        img = np.stack([np.asarray(im)] * 3, axis=-1)
        if rng.random() < 0.3:  # gamma
            gamma = rng.uniform(0.8, 1.2)
            lut = (np.clip((np.arange(256) / 255.0) ** gamma, 0, 1) * 255).astype(np.uint8)
            img = lut[img]
        if rng.random() < 0.3:  # light noise
            img = np.clip(img.astype(np.float32) + rng.gauss(0, 5), 0, 255).astype(np.uint8)
        if rng.random() < 0.2:  # light blur
            k = 3
            pad = k // 2
            a = np.pad(img.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode='edge')
            acc = np.zeros_like(img, dtype=np.float32)
            for dy in range(k):
                for dx in range(k):
                    acc += a[dy:dy + img.shape[0], dx:dx + img.shape[1]]
            img = (acc / (k * k)).astype(np.uint8)
        return img, mask


def restore_original_coords(prob_small, orig_wh, scale, pad, method='nearest'):
    """Float probability map (h,w) in network input coords -> float32 map at
    original image size. NEVER passes through uint8 (preserves precision).
    GT binary masks use nearest; probability nearest vs bilinear is a
    controlled experiment axis (fixed to nearest for this round)."""
    import PIL.Image as I
    h, w = prob_small.shape
    unpadded = prob_small[:h - pad[1], :w - pad[0]].astype(np.float32)
    W, H = orig_wh
    im = I.fromarray(unpadded, mode='F')
    resample = I.NEAREST if method == 'nearest' else I.BILINEAR
    return np.asarray(im.resize((W, H), resample), dtype=np.float32)
