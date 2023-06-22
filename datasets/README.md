# Prepare datasets for MM-OVOD (borrows and edits from Detic)

The basic training of our model uses [LVIS](https://www.lvisdataset.org/) (which uses [COCO](https://cocodataset.org/) images) and [ImageNet-21K](https://www.image-net.org/download.php). 
<!-- Optionally, we use [Objects365](https://www.objects365.org/) and [OpenImages (Challenge 2019 version)](https://storage.googleapis.com/openimages/web/challenge2019.html) for cross-dataset evaluation.  -->
Before starting processing, please download the (selected) datasets from the official websites and place or sim-link them under `${mm-ovod_ROOT}/datasets/` with details shown below.

```
${mm-ovod_ROOT}/datasets/
    metadata/
    lvis/
    coco/
    imagenet/
    VisualGenome/
```
`metadata/` is our preprocessed meta-data (included in the repo). See the below [section](#Metadata) for details.
Please follow the following instruction to pre-process individual datasets.

### COCO and LVIS

First, download COCO images and LVIS data place them in the following way:

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
coco/
    train2017/
    val2017/
```

Next, prepare the open-vocabulary LVIS training set using 

```
python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
```

This will generate `datasets/lvis/lvis_v1_train_norare.json`.

### ImageNet-21K

The imagenet folder should look like the below, after following the data-processing
[script](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_script.sh) from ImageNet-21K Pretraining for the Masses ensuring to use the FALL 2011 version.
After this script has run, please rename folders to give the below structure:
```
imagenet/
    imagenet21k_P/
        train/
            n00005787/
                n00005787_*.JPEG
            n00006484/
                n00006484_*.JPEG
            ...
        val/
            n00005787/
                n00005787_*.JPEG
            n00006484/
                n00006484_*.JPEG
            ...
        imagenet21k_small_classes/
            n00004475/
                n00004475_*.JPEG
            n00006024/
                n00006024_*.JPEG
            ...
```

The subset of ImageNet that overlaps with LVIS (IN-L in the paper) will be created from this directory
structure.

~~~
cd ${mm-ovod_ROOT}/datasets/
mkdir imagenet/annotations
python tools/create_imagenetlvis_json.py --imagenet-path datasets/imagenet/imagenet21k_P --out-path datasets/imagenet/annotations/imagenet_lvis_image_info.json
~~~
This creates `datasets/imagenet/annotations/imagenet_lvis_image_info.json`.


### VisualGenome

Some of our image exemplars are sourced from VisualGenome and so download the dataset ensuring the following
files are present with the below structure:
```
VisualGenome/
    VG_100K/
        *.jpg
    VG_100K_2/
        *.jpg
    objects.json
    image_data.json
```

### Metadata

```
metadata/
    lvis_v1_train_cat_info.json
    lvis_gpt3_text-davinci-002_descriptions_author.json
    lvis_gpt3_text-davinci-002_features_author.npy
    lvis_image_exemplar_dict_K-005_author.json
    lvis_image_exemplar_features_agg_K-005_author.npy
    lvis_image_exemplar_features_avg_K-005_author.npy
    lvis_multi-modal_agg_K-005_author.npy
    lvis_multi-modal_avg_K-005_author.npy
    lvis_v1_clip_a+cname.npy
```

`lvis_v1_train_cat_info.json` is used by the Federated loss.
This is created by 
~~~
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train.json
~~~

`lvis_gpt3_text-davinci-002_descriptions_author.json` are the descriptions for each LVIS class
we found using the (now deprecated) text-davinci-002 model from OpenAI.

Users may create their own descriptions by:
~~~
python tools/generate_descriptions.py --ann-path datasets/lvis/lvis_v1_val.json --openai-model text-davinci-003
~~~
which will create a file called `lvis_gpt3_text-davinci-003_descriptions_own.json`.
Be sure to include your own OpenAI API key at the top of `tools/generate_descriptions.py`.

`lvis_gpt3_text-davinci-002_features_author.npy` is the CLIP embeddings for each class in the LVIS
dataset using the descriptions we generate using GPT-3.
~~~
python tools/dump_clip_features_lvis_sentences.py --descriptions-path datasets/metadata/lvis_gpt3_text-davinci-002_descriptions_author.json --ann-path datasets/lvis/lvis_v1_val.json --out-path datasets/metadata/lvis_gpt3_text-davinci-002_features_author.npy
~~~

`lvis_image_exemplar_dict_K-005_author.json` is the dictionary of image exemplars for each LVIS class
used in the paper and produce our results.
One can create their own as follows:
~~~
python tools/sample_exemplars.py --lvis-ann-path datasets/lvis/lvis_v1_val.json --exemplar-dict-path datasets/metadata/exemplar_dict.json -K 5 --out-path datasets/metadata/lvis_image_exemplar_dict_K-005_own.json
~~~

`lvis_image_exemplar_features_agg_K-005_author.npy` is the CLIP embeddings for each class in the LVIS dataset
when using image examplars AND our trained visual aggregator for combining multiple exemplars.
One can create their own using our trained visual aggregator as follows (see [INSTALL.md](../../docs/INSTALL.md) for downloading
visual aggregator weights):
~~~
python tools/generate_vision_classifier_agg.py --exemplar-list datasets/metadata/lvis_image_exemplar_dict_K-005_own.json --out-path datasets/metadata/lvis_image_exemplar_features_agg_K-005_own.npy --num-augs 5 --tta --load-path checkpoints/visual_aggregator/visual_aggregator_ckpt_4_transformer.pth
~~~

`lvis_image_exemplar_features_avg_K-005_author.npy` is the CLIP embeddings for each class in the LVIS dataset
when using image examplars AND averaging the CLIP embeddings of multiple exemplars (not using our trained
aggregator).
One can create their own for example:
~~~
python tools/get_exemplars_tta.py --ann-path /users/prannay/mm-ovod/datasets/metadata/lvis_image_exemplar_dict_K-005_own.json --output-path datasets/metadata/lvis_image_exemplar_features_avg_K-005_own.npy --num-augs 5
~~~

`lvis_multi-modal_agg_K-005_author.npy` is the CLIP embeddings for each class in the LVIS dataset
when using image examplars AND descriptions AND our trained visual aggregator for combining multiple exemplars.
One can create their own for example:
~~~
python tools/norm_feat_sum_norm.py --feat1-path datasets/metadata/lvis_gpt3_text-davinci-002_features_own.npy --feat2-path datasets/metadata/lvis_image_exemplar_features_agg_K-005_own.npy --out-path datasets/metadata/lvis_multi-modal_agg_K-005_own.npy
~~~

`lvis_multi-modal_avg_K-005_author.npy` is the CLIP embeddings for each class in the LVIS dataset
when using image examplars AND descriptions AND averaging the CLIP embeddings of multiple exemplars (not using our trained aggregator).
One can create their own for example:
~~~
python tools/norm_feat_sum_norm.py --feat1-path datasets/metadata/lvis_gpt3_text-davinci-002_features_own.npy --feat2-path datasets/metadata/lvis_image_exemplar_features_avg_K-005_own.npy --out-path datasets/metadata/lvis_multi-modal_avg_K-005_own.npy
~~~

`lvis_clip_a+cname.npy` is the pre-computed CLIP embeddings for each class in the LVIS dataset (from Detic)
They are created by:
~~~
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --out_path metadata/lvis_v1_clip_a+cname.npy
~~~

### Collating Image Exemplars

We provide the exact image exemplars used (5 per LVIS category) in our results in the metadata folder defined
above.
However, if you wish to create your own, one first needs to create a full dictionary of exemplars for each
LVIS category.
This is done by:
~~~
python tools/collate_exemplar_dict.py --lvis-dir datasets/lvis --imagenet-dir datasets/imagenet --visual-genome-dir datasets/VisualGenome --output-path datasets/metadata/exemplar_dict.json
~~~
This will create `datasets/metadata/exemplar_dict.json` which is a dictionary of exemplars with
at least 10 exemplars per LVIS category.