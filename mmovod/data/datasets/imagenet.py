# Copyright (c) Facebook, Inc. and its affiliates.
# import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta
from .lvis_v1 import custom_load_lvis_json


def custom_register_imagenet_instances(name, metadata, json_file, image_root):
    """
    """
    DatasetCatalog.register(name, lambda: custom_load_lvis_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="imagenet", **metadata
    )


_CUSTOM_SPLITS_IMAGENET = {
    "imagenet_lvis_v1": (
        "imagenet/imagenet21k_P/",
        "imagenet/annotations/imagenet_lvis_image_info.json",
    ),
}


for key, (image_root, json_file) in _CUSTOM_SPLITS_IMAGENET.items():
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    custom_register_imagenet_instances(
        key,
        get_lvis_instances_meta('lvis_v1'),
        os.path.join(_root, json_file) if "://" not in json_file else json_file,
        os.path.join(_root, image_root),
    )
