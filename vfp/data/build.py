import os
import torch
import numpy as np
import torch.distributed as dist
import torch.utils.data as data

from .samplers import DistributedClassSampler, SubsetRandomSampler

class FeatFromNpy(data.Dataset):
    def __init__(
            self,
            feat_file_list,
            label_file=None,
    ):
        self.db = np.load(feat_file_list)

        self.labels = None
        if label_file is not None:
            self.labels = np.load(label_file)

    def __getitem__(self, index):
        feat = self.db[index]

        if self.labels is not None:
            label = self.labels[index]
            return feat, label
        else:
            return feat

    def __len__(self):
        return self.db.shape[0]

def build_loader(config, is_train=True):

    if is_train:
        dataset_train = build_dataset(is_train=True, config=config)
        sampler_train = DistributedClassSampler(len(dataset_train), pair_size=config.DATA.PAIR_SIZE, num_instances=2, seed=config.SEED)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            # shuffle=True,
            sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )
        return data_loader_train, None
    else:
        dataset_query, dataset_gallery = build_dataset(is_train=False, config=config)

        data_loader_query = torch.utils.data.DataLoader(
            dataset_query,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
        )

        data_loader_gallery = torch.utils.data.DataLoader(
            dataset_gallery,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
        )
        return data_loader_query, data_loader_gallery


def build_dataset(is_train, config):
    if is_train:
        return FeatFromNpy(config.DATA.TRAIN_FEAT, config.DATA.TRAIN_LABEL)

    else:
        query = FeatFromNpy(config.DATA.QUERY_FEAT)
        gallery = FeatFromNpy(config.DATA.GALLERY_FEAT)
        return query, gallery
