"""E16: apply the E15 branch design AFTER the complete layer16 C3k2."""
from copy import deepcopy

import torch
from torch import nn
from ultralytics.nn.modules import C3k2
from src.models.multiscale_dw_branch import MultiScaleDWBranch, MultiScaleBottleneck

TARGET = 'model.16'


class MultiScaleOutputC3k2(C3k2):
    """Preserve baseline parameter names and graph indices for weight transfer."""

    def __init__(self, original):
        if type(original) is not C3k2:
            raise TypeError('Requires an original C3k2')
        nn.Module.__init__(self)
        self.c = original.c
        self.cv1 = deepcopy(original.cv1)
        self.cv2 = deepcopy(original.cv2)
        self.m = deepcopy(original.m)
        channels = original.cv2.conv.out_channels
        self.ms_branch = MultiScaleDWBranch(channels, channels, reduction=2, kernels=(3,5,7))
        reference = original.cv2.conv.weight
        self.ms_branch.to(device=reference.device, dtype=reference.dtype)
        for key in ('i', 'f', 'type'):
            if hasattr(original, key): setattr(self, key, deepcopy(getattr(original, key)))
        self.np = sum(p.numel() for p in self.parameters())
        self.train(original.training)

    def forward(self, x):
        y = super().forward(x)
        return y + self.ms_branch(y)

    def forward_split(self, x):
        y = super().forward_split(x)
        return y + self.ms_branch(y)


def check_architecture(model):
    if hasattr(model, 'module'): model = model.module
    branches = [n for n,m in model.named_modules() if isinstance(m, MultiScaleDWBranch)]
    if branches != [TARGET+'.ms_branch']:
        raise RuntimeError('E16 requires exactly one output branch at layer16')
    if not isinstance(model.model[16], MultiScaleOutputC3k2):
        raise RuntimeError('Missing output enhancement')
    if any(isinstance(m, MultiScaleBottleneck) for m in model.modules()):
        raise RuntimeError('E15 internal branch must not be present')
    for i in (11,14):
        if not isinstance(model.model[i], nn.Upsample) or model.model[i].mode != 'nearest':
            raise RuntimeError('Nearest upsampling must remain unchanged')
    if model.stride.tolist() != [8.,16.,32.] or model.model[23].f != [16,19,22]:
        raise RuntimeError('Detection connections or strides changed')


def attach_multiscale_branch(source):
    if any(isinstance(m, MultiScaleDWBranch) for m in source.modules()):
        raise ValueError('Refusing to stack E16 on an already modified model')
    result = deepcopy(source)
    result.model[16] = MultiScaleOutputC3k2(result.model[16])
    check_architecture(result)
    return result
