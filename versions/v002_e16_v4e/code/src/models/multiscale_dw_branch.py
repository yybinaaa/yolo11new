"""E15: isolated, baseline-preserving multi-scale branch for YOLO11m."""
from copy import deepcopy

import torch
from torch import nn
from ultralytics.nn.modules import Bottleneck, C3k2

TARGET = "model.16.m.0.m.0"


class MultiScaleDWBranch(nn.Module):
    """Reduce channels, concatenate three spatial scales, then project back.

    Only the final projection is zero initialized. No batch normalization or
    fixed residual multiplier is used in the new branch.
    """

    def __init__(self, c1, c2, reduction=2, kernels=(3, 5, 7)):
        super().__init__()
        if reduction < 1 or not kernels or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("Require positive reduction and positive odd kernels")
        hidden = max(1, c1 // reduction)
        self.reduce = nn.Conv2d(c1, hidden, 1, bias=False)
        self.scales = nn.ModuleList([
            nn.Conv2d(hidden, hidden, k, padding=k // 2, groups=hidden, bias=False)
            for k in kernels
        ])
        self.act = nn.SiLU()
        self.project = nn.Conv2d(hidden * len(kernels), c2, 1, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, x):
        reduced = self.act(self.reduce(x))
        return self.project(torch.cat([self.act(layer(reduced)) for layer in self.scales], 1))


class MultiScaleBottleneck(Bottleneck):
    """Keep cv1/cv2 names and original bottleneck behavior for weight transfer."""

    def __init__(self, original, reduction=2, kernels=(3, 5, 7)):
        if type(original) is not Bottleneck:
            raise TypeError("Expected an unmodified Ultralytics Bottleneck")
        nn.Module.__init__(self)
        self.cv1 = deepcopy(original.cv1)
        self.cv2 = deepcopy(original.cv2)
        self.add = original.add
        c1, c2 = original.cv1.conv.in_channels, original.cv2.conv.out_channels
        self.ms_branch = MultiScaleDWBranch(c1, c2, reduction, kernels)
        reference = original.cv1.conv.weight
        self.ms_branch.to(device=reference.device, dtype=reference.dtype)
        self.train(original.training)

    def forward(self, x):
        return super().forward(x) + self.ms_branch(x)


def attach_multiscale_branch(source, reduction=2, kernels=(3, 5, 7)):
    """Return an independent model copy; never mutate the supplied baseline."""
    if any(isinstance(m, MultiScaleBottleneck) for m in source.modules()):
        raise ValueError("Multi-scale branch is already installed")
    if not isinstance(source.model[16], C3k2):
        raise ValueError("Layer16 must be C3k2")
    for index in (11, 14):
        layer = source.model[index]
        if not isinstance(layer, nn.Upsample) or layer.mode != "nearest":
            raise ValueError("E15 requires the original nearest-neighbor upsampling")
    original = source.get_submodule(TARGET)
    result = deepcopy(source)
    result.get_submodule("model.16.m.0").m[0] = MultiScaleBottleneck(original, reduction, kernels)
    return result
