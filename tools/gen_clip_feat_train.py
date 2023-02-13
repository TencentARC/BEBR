import numpy as np
from torch import nn
import torch
import clip
import yaml
import os
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoCaptions

single_caption = False # choose if evalating only using the first caption
model_name = "RN101" #"ViT-B/32" #"RN50" #"RN50x4"  #


print(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device=device)

data_root = "./dataset/coco"
train_root = os.path.join(data_root, 'train2017')
valid_root = os.path.join(data_root, 'val2017')
train_captions = os.path.join(data_root, 'annotations/captions_train2017.json')
valid_captions = os.path.join(data_root, 'annotations/captions_val2017.json')

valid_dataset = CocoCaptions(root = train_root,
                        annFile = train_captions,
                        transform = preprocess)
print(len(valid_dataset))
exit(0)
valid_dataloader = DataLoader(valid_dataset, batch_size = 1)

train_feat_all = np.zeros((len(valid_dataset) * 20, 512), dtype=np.float32)
train_label = np.zeros(len(valid_dataset) * 20, dtype=np.int32)
cnt = 0

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
    # if not single_caption:
    #     text_emb = text_emb.unsqueeze(0)

    image_emb = model.encode_image(images.cuda()) #embed with image encoder
    
    text_emb  = text_emb.detach().cpu().numpy()
    image_emb = image_emb.detach().cpu().numpy()
    
    step = text_emb.shape[0] * 2
    train_feat_all[cnt:(cnt+step):2] = image_emb
    train_feat_all[(cnt+1):(cnt+step):2] = text_emb
    train_label[cnt : (cnt + len(texts) * 2)] = batch_idx
    cnt = cnt + step

train_label = train_label[:cnt]
np.save('train_label.npy', train_label)

train_feat_all = train_feat_all[:cnt]
# normalized features
train_feat_all = train_feat_all / np.linalg.norm(train_feat_all, axis=1)[:, np.newaxis]
np.save('train_feat_RN101.npy', train_feat_all)
