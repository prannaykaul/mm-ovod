import argparse
import json
import torch
import numpy as np
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptions-path",
        type=str,
        default="datasets/metadata/lvis_gpt3_text-davinci-002_descriptions_author.json"
    )
    parser.add_argument(
        "--ann-path",
        type=str,
        default="datasets/lvis/lvis_v1_val.json"
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="datasets/metadata/lvis_gpt3_text-davinci-002_features_author.npy"
    )
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="ViT-B/32")

    args = parser.parse_args()

    print("Loading descriptions from: {}".format(args.descriptions_path))
    with open(args.descriptions_path, 'r') as f:
        descriptions = json.load(f)

    print("Loading annotations from: {}".format(args.ann_path))
    with open(args.ann_path, 'r') as f:
        ann_data = json.load(f)

    lvis_cats = ann_data['categories']
    lvis_cats = [(c['id'], c['synonyms'][0].replace("_", " ")) for c in lvis_cats]
    sentences_per_cat = [descriptions[c[1]] for c in lvis_cats]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        "Total number of sentences for {} classes: {}".format(
            len(sentences_per_cat), sum(len(x) for x in sentences_per_cat))
    )

    if args.model == 'clip':
        import clip
        print('Loading CLIP')
        model, preprocess = clip.load(args.clip_model, device=device)
        model.eval()
        all_text_features = []
        for cat_sentences in tqdm(sentences_per_cat):
            text = clip.tokenize(cat_sentences, truncate=True)
            with torch.no_grad():
                if len(text) > 10000:
                    split_text = text.split(128)
                    split_features = []
                    for t in tqdm(split_text, total=len(split_text)):
                        split_features.append(model.encode_text(t.to(device)).cpu())
                    text_features = torch.cat(split_features, dim=0)
                    # text_features = torch.cat([
                    #     model.encode_text(t) for t in text.split(128)
                    # ], dim=0)
                else:
                    text_features = model.encode_text(text.to(device))
            all_text_features.append(text_features.mean(dim=0))
        all_text_features = torch.stack(all_text_features)
        print("Output text features shape: ", all_text_features.shape)
        text_features = text_features.cpu().numpy()

    else:
        assert 0, "Model {} is not supported only clip".format(args.model)
    if args.out_path != '':
        print('saveing to', args.out_path)
        np.save(open(args.out_path, 'wb'), text_features)
