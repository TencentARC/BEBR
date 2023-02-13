from collections import defaultdict
import numpy as np
import math
import copy

import torch
import torch.distributed as dist


def get_cls_samples(imlist):
    cls = defaultdict(list)
    for idx, (_, imlabel, _) in enumerate(imlist):
        cls[imlabel].append(idx)
    return cls


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle=False):
        self.epoch = 0
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch



class DistributedClassSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, pair_size=1, num_instances=2, seed=0):
        self.num_replicas = 1
        self.rank = 0

        self.dataset_size = dataset_size
        # self.pid_index = get_cls_samples(dataset)
        self.num_instances = num_instances
        self.pair_size = pair_size
        # assert (pair_size < num_instances)
        self.epoch = 0
        self.seed = seed

        assert dataset_size % (pair_size+1) == 0
        self.base_indexes = torch.arange(dataset_size)[::(pair_size+1)]
        self.num_samples = len(self.base_indexes)
        self.total_size = self.num_samples

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.base_indexes), generator=g)
        indices = self.base_indexes[indices].tolist()  # type: ignore

        indices = torch.LongTensor(indices).view(-1,1)
        ret = None
        for i in self.selected_groups:
            if ret is None:
                ret = indices + i
            else:
                ret = torch.cat((ret, indices+i), dim=1)
        ret = ret.view(-1).tolist()
        yield from ret

    def set_epoch(self, epoch):
        self.epoch = epoch
        np.random.seed(self.seed + self.epoch)
        self.selected_groups = np.random.choice(torch.arange(self.pair_size+1).tolist(), size=self.num_instances, replace=False)
