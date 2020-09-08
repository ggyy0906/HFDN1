import torch
from mground.math_utils import euclidean_dist
from functools import partial


def gaussian_kernel(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = euclidean_dist(x, y)
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)
    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss(x, y):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]

    K = partial(
        gaussian_kernel, sigmas = torch.cuda.FloatTensor(sigmas)
    )

    loss = torch.mean(K(x, x))
    loss += torch.mean(K(y, y))
    loss -= 2 * torch.mean(K(x, y))

    return loss