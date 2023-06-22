import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from tqdm import tqdm
import os
import argparse
from PIL import Image
import json
from typing import Tuple, Optional
import random

import torch.nn.functional as F
import numpy as np
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
import clip
from clip.model import CLIP as CLIPModel
# import socket
import torchvision.transforms as tvt


# HOSTNAME = socket.gethostname().split(".")[0]
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# if HOSTNAME == "gnodea10":
#     PATHS = {
#         "imagenet21k": "",
#         "visual_genome": "/scratch/local/hdd/prannay/datasets/VisualGenome/",
#         "lvis": "/scratch/local/hdd/prannay/datasets/coco/",
#         "coco": "/scratch/local/hdd/prannay/datasets/coco/",
#         "o365": "/scratch/local/hdd/prannay/datasets/objects365/train/",
#     }
# else:
#     PATHS = {
#         "imagenet21k": "",
#         "visual_genome": "/scratch/local/prannay/datasets/VisualGenome/",
#         "lvis": "/scratch/local/prannay/datasets/coco/",
#         "coco": "/scratch/local/prannay/datasets/coco/",
#         "o365": "/scratch/local/prannay/datasets/objects365/train/",
#     }


LVIS_V1_UNSEEN_IDS = torch.tensor(
    [x['id'] for x in LVIS_V1_CATEGORIES if x['frequency'] == 'r'],
    dtype=torch.long,
)

LVIS_V1_SEEN_IDS = torch.tensor(
    [x['id'] for x in LVIS_V1_CATEGORIES if x['frequency'] != 'r'],
    dtype=torch.long,
)

LVIS_V1_ALL_IDS = torch.tensor(
    [x['id'] for x in LVIS_V1_CATEGORIES],
    dtype=torch.long,
)

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag


def get_crop(img, bb, context=0.0, square=True, train=True):
    x1, y1, w, h = bb
    W, H = img.size
    y, x = y1 + h / 2.0, x1 + w / 2.0
    if train:
        context_sample = random.random() * context
        h, w = h * (1. + context_sample), w * (1. + context_sample)
    else:
        h, w = h * (1. + context), w * (1. + context)
    if square:
        w = max(w, h)
        h = max(w, h)
    if train:
        x_a = random.random()
        y_a = random.random()
        x1, x2 = x - ((1. - x_a) * w / 2.0), x + ((x_a * w) / 2.0)
        y1, y2 = y - ((1. - y_a) * w / 2.0), y + ((y_a * w) / 2.0)
    else:
        x1, x2 = x - w / 2.0, x + w / 2.0
        y1, y2 = y - h / 2.0, y + h / 2.0
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    bb_new = [int(c) for c in [x1, y1, x2, y2]]
    crop = img.crop(bb_new)
    return crop


def run_crop(d, paths, context=0.4, square=True, train=True):
    dataset = d['dataset']
    file_name = os.path.join(paths[dataset], d['file_name'])
    img = Image.open(file_name)
    img = img.convert("RGB")
    img = _apply_exif_orientation(img)
    if dataset == "imagenet21k":
        bb = [0, 0, 0, 0]
        return img
    elif dataset in {"lvis", "coco", "o365"}:
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
    return get_crop(img, bb, context=context, square=square, train=train)


def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        # 2: Image.Transpose.FLIP_LEFT_RIGHT,
        2: Image.FLIP_LEFT_RIGHT,
        # 3: Image.Transpose.ROTATE_180,
        3: Image.ROTATE_180,
        # 4: Image.Transpose.FLIP_TOP_BOTTOM,
        4: Image.FLIP_TOP_BOTTOM,
        # 5: Image.Transpose.TRANSPOSE,
        5: Image.TRANSPOSE,
        # 6: Image.Transpose.ROTATE_270,
        6: Image.ROTATE_270,
        # 7: Image.Transpose.TRANSVERSE,
        7: Image.TRANSVERSE,
        # 8: Image.Transpose.ROTATE_90,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def _convert_image_to_rgb(image: Image.Image):
    return image.convert("RGB")


class TransformsSimCLR_CLIP:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, tta=False, num_augs=5):

        self.tta = tta
        if self.tta:
            random.seed(100000 + num_augs)
            torch.manual_seed(100000 + num_augs)
            s = 0.25
            color_jitter = tvt.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
            )
            self.transform2 = tvt.Compose([
                tvt.RandomResizedCrop(size=224 * 4, scale=(0.8, 1.0), interpolation=BICUBIC),
                tvt.RandomHorizontalFlip(),  # with 0.5 probability
                _convert_image_to_rgb,
                tvt.RandomApply([color_jitter], p=0.8),
            ])
        self.transform = tvt.Compose([
            tvt.Resize(size, interpolation=BICUBIC),
            tvt.CenterCrop(size),
            _convert_image_to_rgb,
            tvt.ToTensor(),
            tvt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __call__(self, x):
        if self.tta:
            x = self.transform2(x)
        return self.transform(x)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'
    Linear = 'linear'
    Identity = 'identity'


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, K, D = x.size()
        x = x.view(B * N * K, D)
        return self.model(x).view(B, N, K, D)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=F.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=F.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 4., act=F.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(
                    TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(
                    TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x) -> torch.Tensor:
        B, N, K, D = x.size()
        x = x.view(B * N * K, D)
        x = self.linear(x).view(B, N, K, D)
        prefix = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, N, *self.cls_token.shape)
        prefix = torch.cat((x, prefix), dim=-2)

        prefix = prefix.view(B * N, K + self.prefix_length, D)
        out = self.transformer(prefix)

        out = out.view(B, N, K + self.prefix_length, D)
        return out[:, :, -1:]

    def __init__(self, dim: int, prefix_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.prefix_length = prefix_length
        self.transformer = Transformer(dim, 8, num_layers)
        self.linear = nn.Linear(dim, dim)
        self.cls_token = nn.Parameter(torch.randn(prefix_length, dim), requires_grad=True)


class MappingPlus(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        mapping_type: MappingType = MappingType.Linear,
    ):
        super(MappingPlus, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.mapping_type = mapping_type

        # init modules
        if self.mapping_type == "transformer":
            self.mapper = TransformerMapper(
                dim=self.dim,
                prefix_length=1,
                num_layers=self.num_layers
            )
        else:
            raise NotImplementedError(self.mapping_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mapper(x)
        return y


class ImageShotDataset(Dataset):
    def __init__(
            self,
            exemplar_dict: str,
            args: argparse.Namespace,
            context: float = 0.2,
            tta: bool = False,
            num_augs: int = 1,
    ):
        super(ImageShotDataset, self).__init__()
        with open(exemplar_dict, "r") as fp:
            self.exemplar_dict = json.load(fp)
        self.context = context
        self.tvt_transform = TransformsSimCLR_CLIP(224, tta=tta, num_augs=num_augs)
        self.num_augs = num_augs
        self.paths = {
            "imagenet21k": args.imagenet_img_dir,
            "visual_genome": args.visual_genome_img_dir,
            "lvis": args.lvis_img_dir,
        }

    def __len__(self):
        return len(self.exemplar_dict)

    def yield_crop(self, d):
        return self.tvt_transform(
            run_crop(
                d, self.paths, context=self.context, square=True, train=False)
        )

    def __getitem__(self, idx):
        class_samples = self.exemplar_dict[idx]
        imgs = [self.yield_crop(d) for d in class_samples for _ in range(self.num_augs)]
        # print(len(imgs))
        return torch.stack(imgs)


@torch.no_grad()
def generate(dataset: ImageShotDataset, clip_model: CLIPModel, model: MappingPlus,
             args, output_path: str = ""):
    device = torch.device("cuda:{}".format(args.gpu))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    clip_model = clip_model.to(device)
    clip_model.eval()
    model = model.to(device)
    model.eval()
    gen_dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=args.nw, pin_memory=True
    )
    out_features = []
    for images in tqdm(gen_dataloader, total=len(gen_dataloader)):
        images = images.to(device)
        images = images.flatten(end_dim=-4)
        # print(images.size())
        clip_features = clip_model.encode_image(images)
        out_feature = model(clip_features[None, None])
        out_features.append(out_feature.view(-1))

    out_features = torch.stack(out_features)
    np.save(output_path, out_features.detach().cpu().half().numpy())
    print(output_path)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exemplar-list', required=True, help="path to list of features")
    parser.add_argument('--out-path', default='', required=True, help="path to output")
    parser.add_argument('--nw', default=4, type=int)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--clip-model', default="ViT-B/32")
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer/linear/identity')
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--load-path', default="")
    parser.add_argument('--tta', action="store_true")
    parser.add_argument('--num-augs', type=int, default=5)
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
    args = parser.parse_args()
    embedding_dim = 512

    gen_dataset = ImageShotDataset(
        args.exemplar_list,
        tta=args.tta,
        num_augs=args.num_augs,
        args=args,
    )

    print("Length of gen dataset: {}".format(len(gen_dataset)))

    clip_model, _ = clip.load(args.clip_model, device="cpu")
    del clip_model.transformer
    model = MappingPlus(
        embedding_dim,
        num_layers=args.num_layers,
        mapping_type=args.mapping_type,
    )
    print(model)
    if args.load_path != "":
        checkpoint = torch.load(args.load_path, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    generate(gen_dataset, clip_model, model, args, output_path=args.out_path)


if __name__ == '__main__':
    main()
