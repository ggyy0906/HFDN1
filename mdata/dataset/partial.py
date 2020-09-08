from torch.utils.data import Dataset
import numpy as np
import torch


import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from torch.utils.data import Subset

# from utils import get_targets
from mdata.dataset.utils import get_targets
# from mground.plot_utils import show_a_tenosr


class PartialDataset(Dataset):
    def __init__(self, dataset, accepted_cls, ncls_mapping=None):

        targets = get_targets(dataset)
        if targets is None: raise Exception('target not found.')

        # generate mask of idx for sample in accepted_cls.
        accepted_mask = torch.sum(
            torch.stack([(targets == i) for i in accepted_cls], dim=1), dim=1
        ).bool()

        accepted_idx = torch.Tensor(np.arange(len(dataset)))[accepted_mask].long()

        # select targets and mapping cls if needed.

        targets = targets[accepted_mask]

        cls_mapping = dict()
        for i in accepted_cls:
            cls_mapping[str(torch.tensor(i))] = torch.tensor(i)
        
        ncls_mapping = {str(torch.tensor(k)): torch.tensor(v) for k,v in ncls_mapping.items()}
        
        cls_mapping.update(ncls_mapping)
        self.cls_mapping = cls_mapping

        self.dataset = dataset
        self.accepted_idx = accepted_idx
        self.targets = targets.long()
    

    def __getitem__(self, idx):
        img = self.dataset[self.accepted_idx[idx]][0]
        lable = self.cls_mapping[str(self.targets[idx])]
        return img, lable

    def __len__(self):
        return len(self.accepted_idx)


if __name__ == "__main__":
    channel = 1
    trans = [
        transforms.Resize(28),
        transforms.Grayscale(channel),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,) * channel, std=(0.5,) * channel),
    ]
    trans = transforms.Compose(trans)
    dataset = MNIST(
        root="./DATASET/MNIST", train=True, download=True, transform=trans
    )

    pd = PartialDataset(dataset, accepted_cls=[3, 4, 5, 6, 7, 8, 9])

