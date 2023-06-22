# Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.8.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).


### Author conda environment setup
```bash
conda create --name mm-ovod python=3.8 -y
conda activate mm-ovod
conda install pytorch torchvision=0.9.2 torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
git checkout 2b98c273b240b54d2d0ee6853dc331c4f2ca87b9
pip install -e .

cd ..
git clone https://github.com/prannaykaul/mm-ovod.git --recurse-submodules
cd mm-ovod
pip install -r requirements.txt
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

Our project (like Detic) use a submodule: [CenterNet2](https://github.com/xingyizhou/CenterNet2.git). If you forget to add `--recurse-submodules`, do `git submodule init` and then `git submodule update`.


### Downloading pre-trained ResNet-50 backbone
We use the ResNet-50 backbone pre-trained on ImageNet-21k-P from [here](
https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth). Please download it from the previous link, place it in the `${mm-ovod_ROOT}/checkpoints` folder and use the following command to convert it for use with detectron2:
```bash
cd ${mm-ovod_ROOT}
mkdir checkpoints
cd checkpoints
wget https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth
python ../tools/convert-thirdparty-pretrained-model-to-d2.py --path resnet50_miil_21k.pth
```

### Downloading pre-trained visual aggregator
The pretrained model for the visual aggregator is required if one wants to use their own image exemplars to produce a vison-based
classifier. The model can be downloaded from [here](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/visual_aggregator_ckpt_4_transformer.pth.tar) and should be placed in the `${mm-ovod_ROOT}/checkpoints` folder.
```bash
cd ${mm-ovod_ROOT}
mkdir checkpoints
cd checkpoints
wget https://robots.ox.ac.uk/~prannay/public_models/mm-ovod/visual_aggregator_ckpt_4_transformer.pth.tar
tar -xf visual_aggregator_ckpt_4_transformer.pth.tar
rm visual_aggregator_ckpt_4_transformer.pth.tar
```
