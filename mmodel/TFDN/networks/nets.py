import torch
import torch.nn as nn
import torch.nn.functional as F
from mmodel.utils.gradient_reverse_layer import GradReverseLayer

from mmodel.utils.backbone import ResnetFeat


class Disentangler(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
    
    def forward(self, feats):
        dist_feats = self.net(feats)
        return dist_feats

class SDisentangler(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048, adv_coeff_fn=lambda:-1):
        super().__init__()
        self.grl = GradReverseLayer(adv_coeff_fn)

        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
    
    def forward(self, feats, adv=False):
        if adv:
            feats = self.grl(feats)  
        dist_feats = self.net(feats)
        return dist_feats


class DomainDis(nn.Module):
    def __init__(self, in_dim=2048, adv_coeff_fn=lambda:-1):
        super().__init__()
        self.grl = GradReverseLayer(adv_coeff_fn)

        self.D = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, feats, adv=False):
        if adv:
            feats = self.grl(feats)            
        domain = self.D(feats)
        return domain


class SDomainDis(nn.Module):
    def __init__(self, in_dim=512, adv_coeff_fn=lambda:-1):
        super().__init__()
        self.grl = GradReverseLayer(adv_coeff_fn)
        self.D = nn.Sequential(
            nn.Linear(in_dim, 1),
        )

    def forward(self, feats, adv=False):
        if adv:
            feats = self.grl(feats)            
        domain = self.D(feats)
        return domain


class ClassPredictor(nn.Module):
    def __init__(self, cls_num, adv_coeff_fn=lambda:-1):
        super().__init__()
        self.grl = GradReverseLayer(adv_coeff_fn)
        self.C = nn.Sequential(
            nn.Linear(512, cls_num),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats, adv=False):
        if adv:
            feats = self.grl(feats)   
        cls = self.C(feats)
        return cls

class Reconstructor(nn.Module):
    def __init__(self, indim=2):
        super().__init__()

        self.R1 = nn.Sequential(
            nn.Linear(512, 512*2),
            # nn.ReLU(inplace=True),
            # nn.Linear(512*2, 512*2),
        )
        self.R2 = nn.Sequential(
            nn.Linear(512, 512*2),
            # nn.ReLU(inplace=True),
            # nn.Linear(512*2, 512*2),
        )

    def forward(self, feats):
        # ori_feats = torch.cat(feats, dim=1)
        # rec_feats = self.R(ori_feats)
        r1 = self.R1(feats[0])
        r2 = self.R1(feats[1])
        rec_feats = r1 + r2
        return rec_feats

class Conver(nn.Module):
    def __init__(self, in_dim=2048,adv_coeff_fn=lambda:-1):
        super().__init__()
        self.grl = GradReverseLayer(adv_coeff_fn)
        self.D = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
        )

    def forward(self, feats, adv=False):
        if adv:
            feats = self.grl(feats) 
        domain = self.D(feats)
        return domain
    
class Mine(nn.Module):
    def __init__(self,f=2048, s=2048):
        super().__init__()
        self.fc1_x = nn.Linear(f, 512)
        self.fc1_y = nn.Linear(s, 512)
        self.fc2 = nn.Linear(512,1)
    def forward(self, x,y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2

    def mutual_est(self,x,y):
        shuffile_idx = torch.randperm(y.shape[0])
        y_ = y[shuffile_idx]
        joint, marginal = self(x, y), self(x, y_)
        return torch.mean(joint) - torch.log(
            torch.mean(torch.exp(marginal))
        )