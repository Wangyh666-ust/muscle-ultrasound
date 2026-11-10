"""Four-input boundary model and frozen-pixel guide dataset; Round34 only."""
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image
from src.umud_baseline.boundary_identity import BoundaryIdentityUNet,IdentityDataset

class GuidedBoundaryUNet(BoundaryIdentityUNet):
    def __init__(self):
        super().__init__()
        old=self.base.encoder.conv1
        new=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        with torch.no_grad():new.weight[:,:3].copy_(old.weight);new.weight[:,3:].zero_()
        self.base.encoder.conv1=new
        self.base.encoder._in_channels=4
    def load_shared(self,state):
        state=dict(state);w=state['base.encoder.conv1.weight']
        if w.shape[1]!=3:raise ValueError('shared init must be original three channels')
        state['base.encoder.conv1.weight']=torch.cat([w,torch.zeros_like(w[:,:1])],1)
        self.load_state_dict(state,strict=True)

def guide_tensor(mask,size=512,flip=False):
    a=np.asarray(mask,np.float32)
    if a.ndim!=2 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('invalid guide')
    if flip:a=a[:,::-1].copy()
    h,w=a.shape;nw=max(1,round(w*size/max(h,w)));nh=max(1,round(h*size/max(h,w)))
    a=cv2.resize(a,(nw,nh),interpolation=cv2.INTER_AREA)
    a=np.pad(a,((0,size-nh),(0,size-nw)))
    return torch.from_numpy(a.copy())[None]

class GuidedDataset(IdentityDataset):
    def __init__(self,root,samples,long_side=512,seed=20261023,augment=True,arm='B'):
        super().__init__(root,samples,long_side,seed,augment)
        if arm not in ('A','B'):raise ValueError('arm')
        self.arm=arm
    def __getitem__(self,i):
        b=super().__getitem__(i);s=self.samples[i];key,_=self.choice(s)
        with Image.open(self.root/s['guide_path']) as im:mask=np.asarray(im.convert('L'))>127
        flip=self.augment and random.Random(key+':aug').random()<.5
        guide=guide_tensor(mask,self.long_side,flip)
        b['guide_pixels']=int(mask.sum())
        if self.arm=='A':guide=torch.zeros_like(guide)
        b['image']=torch.cat([b['image'],guide],0)
        return b
