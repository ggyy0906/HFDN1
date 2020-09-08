import numpy as np
import torch

possible_label_attr = ["targets"]


def get_targets(dataset):
    for attr in possible_label_attr:
        targets = getattr(dataset, attr, None)
        if targets is not None:
            break

    if targets is not None:
        if type(targets) is not torch.Tensor:
            targets = torch.Tensor(targets)

    return targets


def universal_label_mapping(sou_cls, tar_cls):

    # if min(tar_cls) < min(sou_cls):
    #     raise Exception("not considered situation")

    sou_cls = set(sorted(sou_cls))
    tar_cls = set(sorted(tar_cls))
    total = sou_cls | tar_cls

    cls_num = len(total)

    order_mapping = None
    if min(sou_cls) is not 0 or max(sou_cls) is not cls_num:
        rearranged_cls = list(range(cls_num))
        sou_order_mapping = {
            original: rearranged_cls[idx]
            for idx, original in enumerate(sou_cls)
        }
        tar_order_mapping = {
            original: rearranged_cls[idx + len(sou_cls)]
            for idx, original in enumerate(tar_cls - sou_cls)
        }

        order_mapping = {**sou_order_mapping, **tar_order_mapping}

        sou_cls = set([order_mapping[o] for o in sou_cls])
        tar_cls = set([order_mapping[o] for o in tar_cls])

    open_mapping = {o: len(sou_cls) for o in tar_cls - sou_cls}

    if order_mapping is None:
        return open_mapping

    final_mapping = {
        k: open_mapping.get(v, v) for k, v in order_mapping.items()
    }

    return final_mapping


if __name__ == "__main__":

    shared = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22]
    sou_private = [2, 3, 4, 6, 7, 8, 9, 13, 14, 18]
    tar_private = [19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]
    a = universal_label_mapping(shared + sou_private, shared + tar_private)

    print(a)
