import torch
import random
import itertools
import numpy
import random
from torch.utils.data import Sampler
from functools import partial
from copy import deepcopy
from abc import abstractclassmethod


class LevelSampler(Sampler):
    def __init__(self, targets_sampler, **kwargs):
        self.__dict__.update(kwargs)
        self.targets_sampler = targets_sampler
        self.has_up_sampler = False
        if isinstance(targets_sampler, LevelSampler):
            self.has_up_sampler = True
        self.targets = self.get_targets()
        # self.idx_list = self.generate_idxs_list(targets)

    @abstractclassmethod
    def generate_idxs_list(self, targets):
        pass

    def get_targets(self):
        if self.has_up_sampler:
            sampler = self.targets_sampler
            up_idx = list(sampler.__iter__())
            target = sampler.get_targets()[up_idx]
            self.up_idx = up_idx
        else:
            target = self.targets_sampler

        if type(target) is numpy.ndarray:
            target = torch.Tensor(target)
        return target

    def __iter__(self):
        idx = self.generate_idxs_list(self.targets)
        if self.has_up_sampler:
            idx = [self.up_idx[i] for i in idx]
            assert False
        return iter(idx)


class BalancedSampler(LevelSampler):
    """ sampler for balanced sampling
    """

    def __init__(self, targets, max_per_cls=None):
        super().__init__(targets_sampler=targets, max_per_cls=max_per_cls)

    def generate_idxs_list(self, targets):
        all_classes = torch.unique(targets)
        cls_idxs = list()
        for curr_cls in all_classes:
            sample_indexes = [
                i for i in range(len(targets)) if targets[i] == curr_cls
            ]

            if self.max_per_cls:
                random.shuffle(sample_indexes)
                sample_indexes = sample_indexes[0 : self.max_per_cls]

            cls_idxs.append(sample_indexes)

        cls_num = len(cls_idxs)

        self.total = cls_num * min([len(i) for i in cls_idxs])

        def shuffle_with_return(l):
            random.shuffle(l)
            return l

        cls_idxs = list(map(shuffle_with_return, cls_idxs))
        cls_idxs = list(zip(*cls_idxs))

        sample_idx = list(itertools.chain(*cls_idxs))
        return sample_idx

    def __len__(self):
        return self.total


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    trans = transforms.Compose([transforms.ToTensor()])

    minist = MNIST(
        root="./DATASET/MNIST", train=True, download=True, transform=trans
    )
    a = minist.targets.numpy()
    pa = PartialSampler(a, [1, 2, 3, 4, 5, 6, 7])
    balanced = BalancedSampler(pa)
    data = DataLoader(
        minist, batch_size=120, shuffle=False, sampler=balanced, drop_last=True
    )

    for batch_idx, samples in enumerate(data):
        d, t = samples

    for batch_idx, samples in enumerate(data):
        d, t = samples
        print(t)

