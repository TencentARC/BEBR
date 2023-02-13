import numpy as np
from torch import nn
import torch
import clip
import yaml
import os
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoCaptions

single_caption = True # choose if evalating only using the first caption
model_name = "RN101" #"ViT-B/32" #"RN50" #"RN50x4"  #

print(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device=device)

data_root = "./dataset/coco"
valid_root = os.path.join(data_root, 'val2017')
valid_captions = os.path.join(data_root, 'annotations/captions_val2017.json')

valid_dataset = CocoCaptions(root = valid_root,
                        annFile = valid_captions,
                        transform = preprocess)

valid_dataloader = DataLoader(valid_dataset, batch_size = 1)

# fwd all samples
image_features = []
text_features = []
for batch_idx, batch in enumerate(valid_dataloader):
    print('Evaluating batch {}/{}'.format(batch_idx, len(valid_dataloader)), end = "\r")
    images, texts = batch
    if single_caption:
        texts = [texts[0][0]]
    else:
        texts = [txt[0] for txt in texts]

    texts = clip.tokenize(texts).cuda() #tokenize
    text_emb = model.encode_text(texts) #embed with text encoder
    if not single_caption:
        text_emb = text_emb.unsqueeze(0)

    image_emb = model.encode_image(images.cuda()) #embed with image encoder

    text_features.append(text_emb.detach().cpu())
    image_features.append(image_emb.detach().cpu())

image_features = torch.cat(image_features, 0)
text_features = torch.cat(text_features, 0)
print('Done forward')

# normalized features
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
np.save('./dataset/train_eval/val_img_feat_RN101.npy', image_features)
np.save('./dataset/train_eval/val_txt_feat_RN101.npy', text_features)
