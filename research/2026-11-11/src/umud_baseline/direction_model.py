# -*- coding: utf-8 -*-
"""Shared ResNet34-UNet with a 1x1 direction head.

Round26.1 fixes the auxiliary loss to the preregistered channel mean and the
axis diagnostic to the half-angle of normalized double-angle vectors.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.umud_baseline.train import build_model

DIRECTION_LOSS_VERSION = 'mean_over_two_channels_v2'
LEGACY_DIRECTION_LOSS_VERSION = 'sum_over_two_channels_v1'
DIRECTION_DIAGNOSTIC_VERSION = 'axis_half_angle_v2'
DIRECTION_EPS = 1e-8


class DirectionAuxUNet(nn.Module):
    def __init__(self, pretrained=False, encoder_weights=None):
        super().__init__()
        self.base = build_model(
            pretrained=pretrained,
            architecture='resnet34_unet',
            encoder_name='resnet34',
            encoder_weights=encoder_weights)
        dec_out = int(self.base.decoder.blocks[-1].conv2[0].out_channels)
        self.dir_head = nn.Conv2d(dec_out, 2, kernel_size=1, stride=1,
                                  padding=0)

    def forward(self, x):
        feats = self.base.encoder(x)
        dec = self.base.decoder(feats)
        logits = self.base.segmentation_head(dec)
        direction = self.dir_head(dec)
        return logits, direction


def direction_mse_loss(pred, target, valid):
    """Mean-channel MSE, averaged per image, then equally over images.

    L_i = mean_{x in V_i} [((p_cos-t_cos)^2 + (p_sin-t_sin)^2) / 2]
    L   = mean over images with |V_i| > 0.
    Images without valid pixels do not enter the denominator. A batch with no
    valid pixels returns a differentiable zero.
    """
    m = (valid[:, 0] > 0).to(pred.dtype)
    denom = m.sum(dim=(1, 2)).clamp(min=1.0)
    err = 0.5 * ((pred - target) ** 2).sum(dim=1)
    per_image = (err * m).sum(dim=(1, 2)) / denom
    has = (m.sum(dim=(1, 2)) > 0).to(pred.dtype)
    return (per_image * has).sum() / has.sum().clamp(min=1.0)


def axis_errors_deg(pred, target, valid, eps=DIRECTION_EPS):
    """Per-pixel double-angle axis error in degrees.

    Returns (errors, defined, target_norm, pred_norm). `errors` is NaN where
    the prediction is undefined; callers must keep those in separate counts.
    Targets must be finite and non-zero. The dot product is of unit vectors,
    then half-angle converted; no absolute value is taken.
    """
    if not torch.isfinite(target).all():
        raise ValueError('non-finite direction target')
    m = valid[:, 0] > 0
    t = target.permute(0, 2, 3, 1).reshape(-1, 2)[m.reshape(-1)]
    p = pred.permute(0, 2, 3, 1).reshape(-1, 2)[m.reshape(-1)]
    tn = torch.linalg.vector_norm(t, dim=1)
    if not torch.isfinite(tn).all() or (tn < eps).any():
        raise ValueError('direction target norm below eps or non-finite')
    pn = torch.linalg.vector_norm(p, dim=1)
    defined = torch.isfinite(pn) & (pn >= eps)
    cos = (p[defined] / pn[defined, None] *
           t[defined] / tn[defined, None]).sum(dim=1).clamp(-1.0, 1.0)
    err = 0.5 * torch.acos(cos) * (180.0 / math.pi)
    out = torch.full((p.shape[0],), float('nan'), device=p.device,
                     dtype=torch.float32)
    out[defined] = err
    return out, defined, tn, pn


@torch.no_grad()
def direction_diagnostics(model, loader, device, amp=True):
    """Pixel-level axis diagnostics; hierarchy aggregation happens separately.

    The caller's train/eval mode is restored exactly. Undefined predictions
    are counted, not imputed as zero error.
    """
    was_training = model.training
    model.eval()
    errs = []
    n_target = 0
    n_undefined = 0
    n_images = 0
    n_seen_images = 0
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        tgt = batch['dir_target'].to(device, non_blocking=True)
        val = batch['dir_valid'].to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=amp):
            _, pred = model(img)
        err, defined, _, _ = axis_errors_deg(pred, tgt, val)
        errs.extend(err[defined].float().cpu().tolist())
        n_target += int(defined.numel())
        n_undefined += int((~defined).sum())
        target_counts = (val[:, 0] > 0).sum(dim=(1, 2)).tolist()
        offset = 0
        for count in target_counts:
            n_images += int(defined[offset:offset + count].any())
            offset += count
        n_seen_images += img.shape[0]
    if was_training:
        model.train()
    return {'direction_axis_mae_deg': (float(sum(errs) / len(errs))
                                       if errs else None),
            'direction_defined_pixels': len(errs),
            'direction_undefined_pixels': n_undefined,
            'direction_target_pixels': n_target,
            'direction_images_with_defined_pixels': n_images,
            'direction_seen_images': n_seen_images,
            'aggregation': 'pixel_equal_monitoring_not_image_equal',
            'image_count_version': 'per_image_v2',
            'epsilon': DIRECTION_EPS,
            'version': DIRECTION_DIAGNOSTIC_VERSION}
