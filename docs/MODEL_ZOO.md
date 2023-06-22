# Multi-Modal Open-Vocabulary Object Detection Model Zoo

## Introduction

This file documents a collection of models reported in our paper.
Training in all cases is done with 4 32GB V100 GPUs.

#### How to Read the Tables

The "Name" column contains a link to the config file. 
To train a model, run 

```
python train_net_auto.py --num-gpus 4 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/ pretrained model, run 

```
python train_net_auto.py --num-gpus 4 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
```


## Open-vocabulary LVIS

| Name                                                                                                                   |  APr |  mAP | Weights                                                          |
|------------------------------------------------------------------------------------------------------------------------|:----:|:----:|------------------------------------------------------------------|
| [lvis-base_r50_4x_clip_gpt3_descriptions](../configs/lvis-base_r50_4x_clip_gpt3_descriptions.yaml)                     | 19.3 | 30.3 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_r50_4x_clip_gpt3_descriptions.pth.tar) |
| [lvis-base_r50_4x_clip_image_exemplars_avg](../configs/lvis-base_r50_4x_clip_image_exemplars_avg.yaml)                 | 14.8 | 28.8 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_r50_4x_clip_image_exemplars_avg.pth.tar) |
| [lvis-base_r50_4x_clip_image_exemplars_agg](../configs/lvis-base_r50_4x_clip_image_exemplars_agg.yaml)                 | 18.3 | 29.2 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_r50_4x_clip_image_exemplars_agg.pth.tar) |
| [lvis-base_r50_4x_clip_multi_modal_avg](../configs/lvis-base_r50_4x_clip_multi_modal_avg.yaml)                         | 20.7 | 30.5 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_r50_4x_clip_multi_modal_avg.pth.tar) |
| [lvis-base_r50_4x_clip_multi_modal_agg](../configs/lvis-base_r50_4x_clip_multi_modal_agg.yaml)                         | 19.2 | 30.6 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_r50_4x_clip_multi_modal_agg.pth.tar) |
| [lvis-base_in-l_r50_4x_4x_clip_gpt3_descriptions](../configs/lvis-base_in-l_r50_4x_4x_clip_gpt3_descriptions.yaml)     | 25.8 | 32.6 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_in-l_r50_4x_4x_clip_gpt3_descriptions.pth.tar) |
| [lvis-base_in-l_r50_4x_4x_clip_image_exemplars_avg](../configs/lvis-base_in-l_r50_4x_4x_clip_image_exemplars_avg.yaml) | 21.6 | 31.3 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_in-l_r50_4x_4x_clip_image_exemplars_avg.pth.tar) |
| [lvis-base_in-l_r50_4x_4x_clip_image_exemplars_agg](../configs/lvis-base_in-l_r50_4x_4x_clip_image_exemplars_agg.yaml) | 23.8 | 31.3 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_in-l_r50_4x_4x_clip_image_exemplars_agg.pth.tar) |
| [lvis-base_in-l_r50_4x_4x_clip_multi_modal_avg](../configs/lvis-base_in-l_r50_4x_4x_clip_multi_modal_avg.yaml)         | 26.5 | 32.8 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_in-l_r50_4x_4x_clip_multi_modal_avg.pth.tar) |
| [lvis-base_in-l_r50_4x_4x_clip_multi_modal_agg](../configs/lvis-base_in-l_r50_4x_4x_clip_multi_modal_agg.yaml)         | 27.3 | 33.1 | [model](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_in-l_r50_4x_4x_clip_multi_modal_agg.pth.tar) |


#### Note

- The open-vocabulary LVIS setup is LVIS without rare class annotations in training. We evaluate rare classes as novel classes in testing.

- All models use [CLIP](https://github.com/openai/CLIP) embeddings as classifiers. This makes the box-supervised models have non-zero mAP on novel classes.

- The models with `in-l` use the overlap classes between ImageNet-21K and LVIS as image-labeled data.

- The models which are trained on `in-l` require the corresponding models _without_ `in-l` (indicated by MODEL.WEIGHTS in the config files). Please train or download the model without `in-l` and place them under `${mm-ovod_ROOT}/output/..` before training the model using `in-l` (check the config file).

