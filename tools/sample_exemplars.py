import json
import argparse
from lvis import LVIS
from tqdm.auto import tqdm
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exemplar-dict-path",
        type=str,
        default="datasets/metadata/exemplar_dict.json"
    )
    parser.add_argument(
        "--lvis-ann-path",
        type=str,
        default="datasets/lvis/lvis_v1_val.json"
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="datasets/metadata/lvis_image_exemplar_dict_K-005_own.json"
    )
    parser.add_argument(
        "-K",
        type=int,
        default=5
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    args = parser.parse_args()
    return args


def run(catid2synset, exemplar_dict, K, seed, out_path):
    chosen_anns_all = []
    for cat_id in tqdm(sorted(catid2synset.keys()), total=len(catid2synset)):
        rng = np.random.default_rng(seed)
        synset = catid2synset[cat_id]
        trial_anns = exemplar_dict[synset]
        probs = [a['area'] for a in trial_anns]
        probs = np.array(probs) / sum(probs)
        chosen_anns = rng.choice(trial_anns, size=K, p=probs, replace=False)
        chosen_anns_all.append(chosen_anns.tolist())
    with open(out_path, "w") as f:
        json.dump(chosen_anns_all, f)


def main(args):
    lvis_cats = LVIS(args.lvis_ann_path).cats
    catid2synset = {v['id']: v['synset'] for v in lvis_cats.values()}
    with open(args.exemplar_dict_path, 'r') as f:
        exemplar_dict = json.load(f)
    for k, v in exemplar_dict.items():
        for ann in v:
            dataset = ann['dataset']
            if dataset == "imagenet21k":
                ann['area'] = 100. * 100.
            elif dataset == "lvis":
                ann['area'] = float(ann['bbox'][2] * ann['bbox'][3])
            elif dataset == "visual_genome":
                ann['area'] = float(ann['w'] * ann['h'])

    run(catid2synset, exemplar_dict, args.K, args.seed, args.out_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
