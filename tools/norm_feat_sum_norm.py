import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feat1-path",
        type=str,
        default="datasets/metadata/lvis_gpt3_text-davinci-002_features_author.npy"
    )
    parser.add_argument(
        "--feat2-path",
        type=str,
        default="datasets/metadata/lvis_image_exemplar_features_avg_K-005_own.npy"
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="datasets/metadata/lvis_multi-modal_avg_K-005_own.npy"
    )
    return parser.parse_args()


def main(args):

    feat1 = np.load(args.feat1_path)
    feat2 = np.load(args.feat2_path)
    # l2 normalize each
    feat1 = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True)
    feat2 = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True)
    # take sum
    feat = feat1 + feat2
    # l2 normalize again
    feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    np.save(args.out_path, feat)


if __name__ == '__main__':
    args = get_args()
    main(args)
