import torch

def euclidean_dist(x, y=None, x_wise=True):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    order = [1,0] if x_wise else [0,1]
    x = x.unsqueeze(order[0])
    y = y.unsqueeze(order[1])

    return torch.pow(x - y, 2).sum(2)