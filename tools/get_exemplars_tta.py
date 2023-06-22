#!/usr/bin/env python
# coding: utf-8

import clip
import json
from PIL import Image
import sys
import argparse
# from lvis import LVIS
import os
# from pprint import pprint
import random
import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision.transforms as tvt
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# PATHS = {
#     "imagenet21k": "",
#     "visual_genome": "/scratch/local/hdd/prannay/datasets/VisualGenome/",
#     "lvis": "/scratch/local/hdd/prannay/datasets/coco/",
# }


def _convert_image_to_rgb(image: Image.Image):
    return image.convert("RGB")


def get_crop(img, bb, context=0.0, square=True):
    # print(bb)
    x1, y1, w, h = bb
    W, H = img.size
    y, x = y1 + h / 2.0, x1 + w / 2.0
    h, w = h * (1. + context), w * (1. + context)
    if square:
        w = max(w, h)
        h = max(w, h)
    # print(x, y, w, h)
    x1, x2 = x - w / 2.0, x + w / 2.0
    y1, y2 = y - h / 2.0, y + h / 2.0
    # print([x1, y1, x2, y2])
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    # print([x1, y1, x2, y2])
    bb_new = [int(c) for c in [x1, y1, x2, y2]]
    # print(bb_new)
    crop = img.crop(bb_new)
    return crop


def run_crop(d, paths, context=0.4, square=True):
    dataset = d['dataset']
    file_name = os.path.join(paths[dataset], d['file_name'])
    # with open(file_name, "rb") as f:
    img = Image.open(file_name)
    if dataset == "imagenet21k":
        bb = [0, 0, 0, 0]
        return img
    elif dataset == "lvis":
        bb = [
            int(c)
            for c in [
                d['bbox'][0] // 1,
                d['bbox'][1] // 1,
                d['bbox'][2] // 1 + 1,
                d['bbox'][3] // 1 + 1
            ]
        ]
    elif dataset == "visual_genome":
        bb = [int(c) for c in [d['x'], d['y'], d['w'], d['h']]]
    return get_crop(img, bb, context=context, square=square)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann-path",
        type=str,
        default="datasets/metadata/lvis_image_exemplar_dict_K-005_own.json"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="datasets/metadata/lvis_image_exemplar_features_avg_K-005_own.npy"
    )
    parser.add_argument(
        "--lvis-img-dir",
        type=str,
        default="datasets/coco/"
    )
    parser.add_argument(
        "--imagenet-img-dir",
        type=str,
        default="datasets/imagenet"
    )
    parser.add_argument(
        "--visual-genome-img-dir",
        type=str,
        default="datasets/VisualGenome/"
    )
    parser.add_argument("--num-augs", type=int, default=5)
    parser.add_argument("--nw", type=int, default=8)

    args = parser.parse_args()
    return args


def main(args):

    anns_path = args.ann_path
    num_augs = args.num_augs

    model, transform = clip.load("ViT-B/32", device="cpu")
    del model.transformer

    model = model.to("cuda:0")

    run(anns_path, num_augs, model, transform, args)


class CropDataset(Dataset):
    def __init__(
        self,
        exemplar_dict,
        num_augs,
        transform,
        transform2,
        args,
    ):
        self.exemplar_dict = exemplar_dict
        self.transform = transform
        self.transform2 = transform2
        self.num_augs = num_augs
        self.paths = {
            "imagenet21k": args.imagenet_img_dir,
            "visual_genome": args.visual_genome_img_dir,
            "lvis": args.lvis_img_dir,
        }

    def __len__(self):
        return len(self.exemplar_dict)

    def __getitem__(self, idx):
        chosen_anns = self.exemplar_dict[idx]
        crops = [run_crop(ann, self.paths) for ann in chosen_anns]
        # add the tta in here somewhere
        crops = [
            self.transform(self.transform2(crop))
            for crop in crops for _ in range(self.num_augs)
        ]
        return torch.stack(crops)


def run(anns_path, num_augs, model, transform, args):
    random.seed(100000 + num_augs)
    torch.manual_seed(100000 + num_augs)
    s = 0.25
    color_jitter = tvt.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    )
    transform2 = tvt.Compose([
        tvt.RandomResizedCrop(size=224 * 4, scale=(0.8, 1.0), interpolation=BICUBIC),
        tvt.RandomHorizontalFlip(),  # with 0.5 probability
        _convert_image_to_rgb,
        tvt.RandomApply([color_jitter], p=0.8),
    ])
    with open(anns_path, "r") as fp:
        exemplar_dict = json.load(fp)
    dataset = CropDataset(exemplar_dict, num_augs, transform, transform2, args)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.nw)
    feats = []
    # chosen_anns_all = []
    for crops in tqdm(dataloader, total=len(dataloader)):
        # rng = np.random.default_rng(seed)
        # synset = catid2synset[cat_id]
        # trial_anns = exemplar_dict[synset]
        # probs = [a['area'] for a in trial_anns]
        # probs = np.array(probs) / sum(probs)
        # chosen_anns = rng.choice(trial_anns, size=K, p=probs, replace=len(trial_anns) < K)
        # if len(trial_anns) < K:
        #     chosen_anns = rng.choice(trial_anns, size=K, p=[a['area'] for a in trial_anns], replace=False)
        # else:
        #     chosen_anns = rng.sample(trial_anns, k=K, counts=[a['area'] for a in trial_anns
        # crops = [run_crop(ann) for ann in chosen_anns]
        # # add the tta in here somewhere
        # crops = [transform(transform2(crop)) for crop in crops for _ in range(num_augs)]
        with torch.no_grad():
            image_embeddings = model.encode_image(crops[0].to("cuda:0"))
            # print(image_embeddings.size())
        feats.append(image_embeddings.cpu())
        # chosen_anns_all.append(chosen_anns.tolist())
        # crops.append([run_crop(ann) for ann in chosen_anns])

    feats_all = torch.stack(feats)

    save_basename = args.output_path

    np.save(save_basename, feats_all.mean(dim=1).numpy())


if __name__ == "__main__":
    args = get_args()
    main(args)
