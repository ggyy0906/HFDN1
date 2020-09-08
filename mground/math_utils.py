import torch
import torch.nn.functional as F

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

def make_weighted_sum(vw):
    assert len(vw) == 2
    v,w = vw
    assert len(v) == len(w)
    r = sum(v[i] * w[i] for i in range(len(w))) / sum(w)
    return r

def entropy(inputs, reduction="none", binary = True):
    """given a propobility inputs in range [0-1], calculate entroy
    
    Arguments:
        inputs {tensor} -- inputs
    
    Returns:
        tensor -- entropy
    """

    def entropy(p):
        return -1 * p * torch.log(p)

    if binary:
        e = entropy(inputs) + entropy(1 - inputs)
    else:
        e = entropy(inputs)

    if reduction == "none":
        return e
    elif reduction == "mean":
        return torch.mean(e)
    elif reduction == 'sum':
        return torch.sum(e)
    else:
        raise Exception("Not have such reduction mode.")

def ent(self, output):
    return - torch.mean(torch.log(F.softmax(output + 1e-6)))