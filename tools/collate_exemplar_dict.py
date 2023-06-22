import time
import json
import glob
import copy
import pickle
import os
import argparse

from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from nltk.corpus import wordnet as wn
from lvis import LVIS
from tqdm.auto import tqdm


def get_code(syn):
    return syn.pos() + str(syn.offset()).zfill(8)


def get_lemma_names(syn):
    return [lemma.name() for lemma in syn.lemmas()]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lvis-dir",
        type=str,
        default="datasets/lvis",
    )
    parser.add_argument(
        "--imagenet-dir",
        type=str,
        default="datasets/imagenet",
    )
    parser.add_argument(
        "--visual-genome-dir",
        type=str,
        default="datasets/VisualGenome",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="datasets/metadata/exemplar_dict.json",
    )
    args = parser.parse_args()
    return args


def main(args):
    lvis_train_dataset = LVIS(os.path.join(args.lvis_dir, "lvis_v1_train.json"))
    # In LVIS the 'stopsign' category is not a wordnet synset and so replace with 'street_sign', a near cousin
    street_sign_synset = wn.synset("street_sign.n.01")
    lvis_synsets = [
        wn.synset(c['synset']) if "stop_sign" not in c['name'] else street_sign_synset
        for c in lvis_train_dataset.cats.values()
    ]
    synset2catid = {v['synset']: v['id'] for v in lvis_train_dataset.cats.values()}
    catid2synset = {v['id']: v['synset'] for v in lvis_train_dataset.cats.values()}

    # Lets start by collecting images in ImageNet
    # we do this using all ImageNet21k with tree hierarchy as defined in the dataset preparation file
    synsets_pos_off_paths = (
        glob.glob(os.path.join(args.imagenet_dir, "train/*"))
        + glob.glob(os.path.join(args.imagenet_dir, "imagenet21k_small_classes/*"))
    )
    # import pdb; pdb.set_trace()
    synsets_pos_off2path = {os.path.basename(d): d for d in synsets_pos_off_paths}
    synsets_pos_off = [os.path.basename(d) for d in synsets_pos_off_paths]
    available_imagenet_synsets = [wn.synset_from_pos_and_offset(a[0], int(a[1:])) for a in synsets_pos_off]
    direct_match_imagenet_synsets = [v for v in available_imagenet_synsets if v.name() in synset2catid.keys()]
    assert len(direct_match_imagenet_synsets) == 997
    exemplar_dict_imagenet = defaultdict(list)

    for v in tqdm(direct_match_imagenet_synsets, total=len(direct_match_imagenet_synsets)):
        code = get_code(v)
        filenames = glob.glob(os.path.join(synsets_pos_off2path[code], "*.JPEG"))
        anns = []
        for filename in filenames:
            # need the last three parts of the path
            ann = {
                "file_name": "/".join(filename.split("/")[-3:]),
                "category_id": synset2catid[v.name()],
                "dataset": "imagenet21k",
                "synset": v.name(),
            }
            anns.append(ann)
        exemplar_dict_imagenet[v.name()].extend(anns)

    # Now lets collect images from LVIS with area > 32*32
    exemplar_dict_lvis = defaultdict(list)
    lvis_anns = lvis_train_dataset.load_anns(
        lvis_train_dataset.get_ann_ids(
            cat_ids=lvis_train_dataset.get_cat_ids(),
            area_rng=[32.0**2, float('inf')]
        ))

    for ann in tqdm(lvis_anns):
        img = lvis_train_dataset.load_imgs([ann['image_id']])[0]
        ann['dataset'] = "lvis"
        ann['file_name'] = "/".join(img['coco_url'].split("/")[-2:])
        ann['synset'] = catid2synset[ann['category_id']]
        exemplar_dict_lvis[catid2synset[ann['category_id']]].append(ann)

    # get keys from both dictionaries
    keys = list(set(exemplar_dict_imagenet.keys()).union(set(exemplar_dict_lvis.keys())))
    exemplar_dict_combined_two = defaultdict(list)
    for key in keys:
        exemplar_dict_combined_two[key] = exemplar_dict_imagenet[key] + exemplar_dict_lvis[key]

    # let's find the lacking synsets
    lacking_synsets = [k.name() for k in lvis_synsets if len(exemplar_dict_combined_two[k.name()]) < 40]
    print(f"After collecting exemplars from ImageNet and LVIS, there are still {len(lacking_synsets)} without at least 40 exemplars")

    # Now let's collect images from Visual Genome
    exemplar_dict_vg = defaultdict(list)
    vg_objects_path = os.path.join(args.visual_genome_dir, "objects.json")
    with open(vg_objects_path, "r") as f:
        visual_genome_objects = json.load(f)

    vg_images_path = os.path.join(args.visual_genome_dir, "image_data.json")
    with open(vg_images_path, "r") as f:
        visual_genome_images = json.load(f)
    visual_genome_iid2path = {v['image_id']: "/".join(v['url'].split("/")[-2:]) for v in visual_genome_images}

    synsets2boxes = defaultdict(list)
    for i, img in tqdm(enumerate(visual_genome_objects), total=len(visual_genome_objects)):
        for j, obj in enumerate(img['objects']):
            if len(obj['synsets']) != 1:
                continue
            synsets2boxes[obj['synsets'][0]].append((i, j, obj['w'] * obj['h']))

    # We shall only use visual genome for synsets with less than 40 exemplars
    for k in tqdm(lacking_synsets, total=len(lacking_synsets)):
        visual_genome_ids = synsets2boxes[k]
        anns = []
        for a in visual_genome_ids:
            if a[-1] < 32**2:
                continue
            img_objects = visual_genome_objects[a[0]]
            iid = img_objects['image_id']
            ann = {
                'image_id': iid,
                'dataset': 'visual_genome',
                'file_name': visual_genome_iid2path[iid],
                'category_id': synset2catid[k],
                'synset': k,
            }
            ann.update(img_objects['objects'][a[1]])
            assert ann['synsets'][0] == k
            anns.append(ann)
        exemplar_dict_vg[k] = anns

    # Now let's combine all the dictionaries
    exemplar_dict_combined_three = defaultdict(list)
    for syn in synset2catid.keys():
        exemplar_dict_combined_three[syn].extend(
            exemplar_dict_lvis[syn]
            + exemplar_dict_imagenet[syn]
            + exemplar_dict_vg[syn]
        )

    # At this point there should be at least TEN exemplars in 1160 out of 1203 synsets
    # as described in the appendix of the paper, some other synsets in imagenet are suitable
    # and in some cases we use close cousins

    manual_synsets = {
        "anklet.n.03": ["anklet.n.02"],
        "beach_ball.n.01": ["volleyball.n.02"],
        "bible.n.01": ["book.n.11"],
        "black_flag.n.01": ["flag.n.01"],
        "bob.n.05": ["spinner.n.03"],
        "bowl.n.08": ["pipe_smoker.n.01"],
        "brooch.n.01": ["pectoral.n.02", "bling.n.01"],
        "card.n.02": ["business_card.n.01", "library_card.n.01"],
        "checkbook.n.01": ["daybook.n.02"],
        "coil.n.05": ["coil_spring.n.01"],
        "coloring_material.n.01": ["crayon.n.01"],
        "crab.n.05": ["shellfish.n.01", "lobster.n.01"],
        "cube.n.05": ["die.n.01"],
        "cufflink.n.01": ["bling.n.01"],
        "dishwasher_detergent.n.01": ["laundry_detergent.n.01", "cleansing_agent.n.01"],
        "diving_board.n.01": ["springboard.n.01"],
        "dollar.n.02": ["money.n.01", "paper_money.n.01"],
        "eel.n.01": ["electric_eel.n.01"],
        "escargot.n.01": ["snail.n.01"],
        "gargoyle.n.02": ["statue.n.01"],
        "gem.n.02": ["crystal.n.01", "bling.n.01"],
        "grits.n.01": ["congee.n.01"],
        "hardback.n.01": ["book.n.07"],
        "jewel.n.01": ["bling.n.01"],
        "keycard.n.01": ["magnetic_stripe.n.01"],
        "lamb_chop.n.01": ["porkchop.n.01", "rib.n.03"],
        "mail_slot.n.01": ["maildrop.n.01", "mailbox.n.01"],
        "milestone.n.01": ["cairn.n.01"],
        "pad.n.03": ["handstamp.n.01"],
        "paperback_book.n.01": ["book.n.07"],
        "paperweight.n.01": ["letter_opener.n.01"],
        "pennant.n.02": ["bunting.n.01"],
        "penny.n.02": ["coin.n.01"],
        "plume.n.02": ["headdress.n.01"],
        "poker.n.01": ["fire_tongs.n.01"],
        "rag_doll.n.01": ["doll.n.01"],
        "road_map.n.02": ["map.n.01"],
        "scarecrow.n.01": ["creche.n.02"],
        "sparkler.n.02": ["firework.n.01"],
        "sugarcane.n.01": ["cane_sugar.n.02"],
        "water_pistol.n.01": ["pistol.n.01"],
        "wedding_ring.n.01": ["ring.n.01"],
        "windsock.n.01": ["weathervane.n.01"],
    }

    manual_exemplar_dict = defaultdict(list)

    remaining_lacking_synsets = [(k, len(v)) for k, v in exemplar_dict_combined_three.items() if len(v) < 10]
    assert len(remaining_lacking_synsets) == len(manual_synsets) == 43

    for k, v in tqdm(manual_synsets.items(), total=len(remaining_lacking_synsets)):
        for manual_synset in v:
            code = get_code(wn.synset(manual_synset))
            if code not in synsets_pos_off2path:
                continue

            filenames = glob.glob(os.path.join(synsets_pos_off2path[code], "*.JPEG"))
            anns = []
            for filename in filenames:
                ann = {
                    "file_name": "/".join(filename.split("/")[-3:]),
                    "category_id": synset2catid[k],
                    "dataset": "imagenet21k",
                    "synset": manual_synset,
                }
                anns.append(ann)
            # if k in remaining_lacking_synsets:
            #     print(k, manual_synset, len(anns))
            manual_exemplar_dict[k].extend(anns)

    for k, v in tqdm(manual_synsets.items(), total=len(remaining_lacking_synsets)):
        for manual_synset in v:
            if manual_synset not in synsets2boxes.keys():
                continue
            visual_genome_ids = synsets2boxes[manual_synset]
            anns = []
            for a in visual_genome_ids:
                if a[-1] < 32**2:
                    continue
                img_objects = visual_genome_objects[a[0]]
                iid = img_objects['image_id']
                ann = {
                    'image_id': iid,
                    'dataset': 'visual_genome',
                    'file_name': visual_genome_iid2path[iid],
                    'category_id': synset2catid[k],
                    'synset': manual_synset,
                }
                ann.update(img_objects['objects'][a[1]])
                # assert ann['synsets'][0] == k
                anns.append(ann)
            manual_exemplar_dict[k].extend(anns)

    # combine manual_exemplar_dict with exemplar_dict_combined_three
    for k, v in manual_exemplar_dict.items():
        exemplar_dict_combined_three[k].extend(v)

    assert min([len(v) for k, v in exemplar_dict_combined_three.items()]) >= 10, "some synsets have less than 10 exemplars"
    with open(args.output_path, "w") as f:
        json.dump(exemplar_dict_combined_three, f)


if __name__ == "__main__":
    args = get_args()
    main(args)
