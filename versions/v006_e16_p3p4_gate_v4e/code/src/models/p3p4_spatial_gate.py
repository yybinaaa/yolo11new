"""V006: retain E16 and gate its residual with P3/P4 spatial context."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules import C3k2
from src.models.multiscale_dw_branch import MultiScaleDWBranch
from src.models.multiscale_dw_output import MultiScaleOutputC3k2


class P3P4SpatialGate(nn.Module):
    def __init__(self, c3, c4, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.p3_proj = nn.Sequential(nn.Conv2d(c3, hidden, 1, bias=False), nn.SiLU())
        self.p4_proj = nn.Sequential(nn.Conv2d(c4, hidden, 1, bias=False), nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv2d(2 * hidden, hidden, 3, padding=1, bias=False), nn.SiLU())
        self.logits = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, p3, p4):
        context = F.interpolate(self.p4_proj(p4), size=p3.shape[-2:], mode='bilinear', align_corners=False)
        return 2.0 * torch.sigmoid(self.logits(self.fuse(torch.cat((self.p3_proj(p3), context), dim=1))))


class GatedMultiScaleOutputC3k2(C3k2):
    """Preserve E16 state_dict names; only spatial_gate is new."""
    def __init__(self, original, c4, hidden=32):
        if type(original) not in (C3k2, MultiScaleOutputC3k2):
            raise TypeError('Expected original C3k2 or v002 E16 (unfused)')
        nn.Module.__init__(self)
        self.c = original.c
        self.cv1, self.cv2, self.m = deepcopy(original.cv1), deepcopy(original.cv2), deepcopy(original.m)
        channels = original.cv2.conv.out_channels
        self.ms_branch = (deepcopy(original.ms_branch) if isinstance(original, MultiScaleOutputC3k2)
                          else MultiScaleDWBranch(channels, channels, reduction=2, kernels=(3, 5, 7)))
        self.spatial_gate = P3P4SpatialGate(channels, c4, hidden)
        ref = original.cv1.conv.weight
        self.to(device=ref.device, dtype=ref.dtype)
        for key in ('i', 'f', 'type'):
            if hasattr(original, key):
                setattr(self, key, deepcopy(getattr(original, key)))
        self.f = [-1, 13]
        self.type = 'src.models.p3p4_spatial_gate.GatedMultiScaleOutputC3k2'
        self.np = sum(p.numel() for p in self.parameters())
        self.train(original.training)

    def forward(self, inputs):
        incoming, p4 = inputs
        p3 = C3k2.forward(self, incoming)
        return p3 + self.spatial_gate(p3, p4) * self.ms_branch(p3)

    def forward_split(self, inputs):
        incoming, p4 = inputs
        p3 = C3k2.forward_split(self, incoming)
        return p3 + self.spatial_gate(p3, p4) * self.ms_branch(p3)


def check_architecture(model):
    if hasattr(model, 'module'):
        model = model.module
    layer = model.model[16]
    if type(layer) is not GatedMultiScaleOutputC3k2 or layer.f != [-1, 13] or 13 not in model.save:
        raise RuntimeError('V006 gate or P4 graph connection missing')
    if layer.spatial_gate.hidden != 32:
        raise RuntimeError('V006 requires 32 hidden channels')
    branches = [n for n, m in model.named_modules() if isinstance(m, MultiScaleDWBranch)]
    if branches != ['model.16.ms_branch']:
        raise RuntimeError(f'Unexpected residual branches: {branches}')
    if layer.spatial_gate.p3_proj[0].in_channels != 256 or layer.spatial_gate.p4_proj[0].in_channels != 512:
        raise RuntimeError('V006 requires YOLO11m channel layout')
    if model.stride.tolist() != [8., 16., 32.] or model.model[23].f != [16, 19, 22] or model.model[23].nc != 9:
        raise RuntimeError('Unexpected detection heads, strides or classes')
    for i in (11, 14):
        if not isinstance(model.model[i], nn.Upsample) or model.model[i].mode != 'nearest':
            raise RuntimeError('Original upsampling must remain unchanged')


def attach_spatial_gate(source):
    result = deepcopy(source)
    result.model[16] = GatedMultiScaleOutputC3k2(source.model[16], source.model[13].cv2.conv.out_channels)
    result.save = sorted(set(result.save) | {13})
    check_architecture(result)
    return result
