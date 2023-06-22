# edited from script in Detic from Facebook, Inc. and its affiliates.
import argparse
import json
import os
from nltk.corpus import wordnet as wn
from detectron2.data.detection_utils import read_image


def get_code(syn):
    return syn.pos() + str(syn.offset()).zfill(8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', default='datasets/imagenet/imagenet21k_P')
    parser.add_argument('--lvis-meta-path', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--out-path', default='datasets/imagenet/annotations/imagenet_lvis_image_info.json')
    args = parser.parse_args()

    print('Loading LVIS meta')
    data = json.load(open(args.lvis_meta_path, 'r'))
    print('Done')
    synset2cat = {x['synset']: x for x in data['categories']}
    count = 0
    images = []
    image_counts = {}
    synset2folders = {}
    for synset in synset2cat:
        if synset == "stop_sign.n.01":
            synset2folders["stop_sign.n.01"] = []
            continue
        code = get_code(wn.synset(synset))
        folders = [
            os.path.join(args.imagenet_path, folder, code) for folder in ["train", "val", "imagenet21k_small_classes"]
        ]
        synset2folders[synset] = [f for f in folders if os.path.exists(f)]

    for synset, synset_folders in synset2folders.items():
        if len(synset_folders) == 0:
            continue
        cat = synset2cat[synset]
        cat_id = cat['id']
        cat_name = cat['name']
        cat_images = []
        files = []
        for folder in synset_folders:
            folder_files = os.listdir(folder)
            for file in folder_files:
                count = count + 1
                # file_name only needs to be last two parts of path
                # import pdb; pdb.set_trace()
                file_name = '{}/{}/{}'.format(*(folder.split("/")[-2:]), file)
                assert os.path.join(folder, file) == os.path.join(args.imagenet_path, file_name)
                img = read_image(os.path.join(args.imagenet_path, file_name))
                h, w = img.shape[:2]
                image = {
                    'id': count,
                    'file_name': file_name,
                    'pos_category_ids': [cat_id],
                    'width': w,
                    'height': h
                }
                cat_images.append(image)
        images.extend(cat_images)
        image_counts[cat_id] = len(cat_images)
        print(cat_id, cat_name, len(cat_images))
    print('# Images', len(images))
    for x in data['categories']:
        x['image_count'] = image_counts[x['id']] if x['id'] in image_counts else 0
    out = {'categories': data['categories'], 'images': images, 'annotations': []}
    print('Writing to', args.out_path)
    json.dump(out, open(args.out_path, 'w'))
