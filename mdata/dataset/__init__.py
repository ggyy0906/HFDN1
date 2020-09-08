import torchvision.transforms as transforms
import torch
import numpy as np
from torchvision.datasets import ImageFolder

ROOT = "./DATASET/"
ROOT_MNIST = "./DATASET/MNIST"


def for_dataset(name, split="train", transfrom=None, with_targets=False):
    data_set = None
    data_info = dict()
    if name.upper() == "MNIST":
        from torchvision.datasets import MNIST

        dataset = MNIST(
            root=ROOT + name.upper(),
            train=(split == "train"),
            transform=transfrom,
            download=True,
        )
        if split=="train":
            data_info["labels"] = dataset.targets.clone().detach()
        else:
            data_info["labels"] = dataset.targets.clone().detach()

    elif name.upper() == "USPS":
        from mdata.dataset.usps import USPS

        dataset = USPS(
            root=ROOT + name.upper(),
            train=(split == "train"),
            transform=transfrom,
            download=True,
        )
        data_info["labels"] = torch.tensor(dataset.targets)


    elif name.upper() == "SVHN":
        raise Exception("Not Implemet Datase.")
        from torchvision.datasets import SVHN

        dataset = SVHN(
            root=ROOT + name.upper(), split=split, transform=transfrom, download=True
        )
        data_info["labels"] = torch.from_numpy(dataset.labels)
    elif name.upper() == "OFFICE31":

        dataset = ImageFolder(
            root = ROOT + name.upper() + "/" + split,
            transform=transfrom,
        )
    elif name.upper() == "OFFICEHOME":

        dataset = ImageFolder(
            root = ROOT + name.upper() + "/" + split,
            transform=transfrom,
        )


    return dataset #, data_info

class _ToTensorWithoutScaling(object):
    def __call__(self, picture):
        return torch.FloatTensor(np.array(picture)).permute(2, 0, 1).contiguous()

def alex_net_transform(is_train):

    trans = [
        _ToTensorWithoutScaling(),
        transforms.Normalize(
            mean=[104.0069879317889, 116.66876761696767, 122.6789143406786],
            std=[1, 1, 1],
        ),
    ]

    trans = (
        [
            transforms.Resize(227),
            transforms.RandomHorizontalFlip(),
        ]
        + trans
        if is_train
        else [transforms.Resize(256), transforms.CenterCrop(227)] + trans
    )

    return transforms.Compose(trans)

def resnet_transform(is_train):

    t_norm = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    trans = (
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        if is_train
        else [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ] 
    )

    return transforms.Compose(trans+t_norm)

def for_digital_transforms(is_rgb=True):
    channel = 3 if is_rgb else 1
    trans = [
        transforms.Resize(28),
        transforms.Grayscale(channel),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,) * channel, std=(0.5,) * channel),
    ]
    return transforms.Compose(trans)

if __name__ == "__main__":
    d = for_dataset("office31", sub_set='A')
    