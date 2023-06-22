# Copyright (c) Facebook, Inc. and its affiliates.
from .modeling.meta_arch import custom_rcnn
from .modeling.roi_heads import detic_roi_heads
from .modeling.roi_heads import res5_roi_heads
from .modeling.backbone import swintransformer
from .modeling.backbone import timm


from .data.datasets import lvis_v1
from .data.datasets import imagenet
# from .data.datasets import objects365
