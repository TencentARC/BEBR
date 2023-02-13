# BEBR

## Approach
![BEBR](BEBR.png)

## Usage
First, install Pytorch 1.13.1 (or later) and torchvision, as well as some additional depencecices
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
$ git clone https://github.com/ganyk/BEBR.git
$ pip install -r requirements
```

## Evaluation
Pre-computed hash features, recurrent binary features and float features are provided for evaluation.
```bash
# run evaluation on hash features
$ python tools/eval.py --image_feat dataset/hash/feat_image.npy --txt_feat dataset/hash/feat_txt.npy

# float features
$ python tools/eval.py --image_feat dataset/float_finetune/feat_image.npy --txt_feat dataset/float_finetune/feat_txt.npy

# recurrent binary features
$ python tools/eval.py --image_feat dataset/bebr/feat_image.npy --txt_feat dataset/bebr/feat_txt.npy
```

The results should be consistent with those in the paper:

| Embedding | Bits  | Recall@1 | Recall@5 | Recall@10 |
|:---------:|:-----:| :-------:| :-------:| :--------:|
| hash      | 1024  | 0.348    | 0.632    | 0.730     |
| bebr      | 1024  | 0.360    | 0.646    | 0.751     |
| float     | 16384 | 0.361    | 0.649    | 0.744     |


## Train

### Prepare data
1. Download COCO datasets and uncompress them
```bash
$ mkdir dataset/coco & cd dataset/coco
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/zips/val2017.zip
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip & unzip val2017.zip & unzip annotations_trainval2017.zip
```

2. Use clip RN101 model to generate float features which will be used as input to train binary model
```bash
$ python tools/gen_clip_feat_train.py
$ python tools/gen_clip_feat_eval.py
```

### Train binary model
```bash
# train hash model
$ sh train_local.sh configs/hash.yaml
# train bebr model
$ sh train_local.sh configs/bebr.yaml
```

### finetune float feature
The training process of binary model can be seened as a finetune process. For fair comparison, we also finetune the clip features using MLPs whose input and output are both float features.
```bash
$ sh train_local.sh configs/float.yaml
```