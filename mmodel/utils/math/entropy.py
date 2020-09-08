import torch
import torch.nn.functional as F

def ent(output, c=1):
    p = F.softmax(output, dim=-1)
    ee = -1 * p * torch.log(p+1e-5)
    me = torch.mean(ee*c)
    return me

def sigmoid_ent(output, c=1):
    p = torch.sigmoid(output)
    ee1 = -1 * p * torch.log(p+1e-5)
    ee2 = -1 * (1-p) * torch.log(1-p+1e-5)
    ee = ee1 + ee2
    me = torch.mean(ee*c)
    return me

def binary_ent(p):
    ee1 = -1 * p * torch.log(p+1e-5)
    ee2 = -1 * (1-p) * torch.log(1-p+1e-5)
    ee = ee1 + ee2
    return ee

def ent_v2(self, output):
    return - torch.mean(torch.log(F.softmax(output + 1e-6)))


