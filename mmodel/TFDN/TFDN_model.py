from .TFDN_params import params

import itertools
from functools import partial

from ..basic_module import TrainableModule
from ..utils.math.entropy import ent

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from mdata.dataset import for_dataset, resnet_transform
from mdata.data_iter import inf_iter
from mdata.sampler import BalancedSampler
from mdata.dataset.partial import PartialDataset
from mdata.dataset.utils import universal_label_mapping


class TFDN(TrainableModule):
    def __init__(self):
        self.CE = torch.nn.CrossEntropyLoss()
        self.NCE = torch.nn.CrossEntropyLoss(reduction="none")
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.KL = torch.nn.KLDivLoss()

        # setting for office-home
        shared = list(range(0, 20))
        sou_private = list(range(20, 45))
        tar_private = list(range(45, 65))

        self.rec_step = 1600
        self.ent_step = 3000
        self.adv_step = 3000

        self.share = shared
        self.private = sou_private

        self.cls_info = {
            "sou_cls": shared + sou_private,
            "tar_cls": shared + tar_private,
            "cls_num": len(shared) + len(sou_private),
            "mapping": universal_label_mapping(
                shared + sou_private, shared + tar_private
            ),
        }

        size = params.batch_size
        self.S = torch.ones([size, 1], dtype=torch.float).cuda()
        self.T = torch.zeros([size, 1], dtype=torch.float).cuda()
        self.zeros = torch.zeros([size, 512]).cuda()
        super().__init__(params)

    def _prepare_data(self):

        D = params.dataset
        S = params.source
        T = params.target

        sou_set = for_dataset(D, split=S, transfrom=resnet_transform(is_train=True))
        tar_set = for_dataset(D, split=T, transfrom=resnet_transform(is_train=True))
        val_set = for_dataset(D, split=T, transfrom=resnet_transform(is_train=False))

        _ParitalDataset = partial(PartialDataset, ncls_mapping=self.cls_info["mapping"])

        sou_set = _ParitalDataset(sou_set, self.cls_info["sou_cls"])
        tar_set = _ParitalDataset(tar_set, self.cls_info["tar_cls"])
        val_set = _ParitalDataset(val_set, self.cls_info["tar_cls"])

        _DataLoader = partial(
            DataLoader,
            batch_size=params.batch_size,
            drop_last=True,
            num_workers=params.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        sou_iter = inf_iter(_DataLoader(sou_set))
        tar_iter = inf_iter(_DataLoader(tar_set))
        val_iter = inf_iter(_DataLoader(val_set), with_end=True)

        def data_feeding_fn(mode):
            if mode == "train":
                s_img, s_label = next(sou_iter)
                t_img, t_label = next(tar_iter)
                return s_img, s_label, t_img, t_label
            elif mode == "valid":
                return next(val_iter)

        return self.cls_info["cls_num"] + 1, data_feeding_fn

    def _regist_networks(self):
        from .networks.nets import (
            ResnetFeat,
            Disentangler,
            DomainDis,
            ClassPredictor,
            Reconstructor,
            Mine,
            SDomainDis,
            SDisentangler,
            Conver,
        )

        def dy_adv_coeff(iter_num, high=1.0, low=0.0, alpha=10.0):
            iter_num = max(iter_num - self.adv_step, 0)
            return np.float(
                2.0
                * (high - low)
                / (1.0 + np.exp(-alpha * iter_num / self.total_steps))
                - (high - low)
                + low
            )

        return {
            "F": ResnetFeat(),
            "N_d": Disentangler(in_dim=2048, out_dim=1024),
            "N_c": Disentangler(in_dim=2048, out_dim=512),
            "D": DomainDis(in_dim=1024),
            "C": ClassPredictor(cls_num=self.cls_info["cls_num"]),
            "N_dd": SDisentangler(in_dim=1024, out_dim=512),
            "N_dc": SDisentangler(in_dim=1024, out_dim=512),
            "D_dc": SDomainDis(in_dim=512),
            "D_dd": SDomainDis(
                in_dim=512, adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            ),
            "M_d": Mine(f=512, s=512), # MIME to maximize difference
            "R_d": Reconstructor(),
            "Cr": Conver(
                in_dim=512, adv_coeff_fn=lambda: dy_adv_coeff(self.current_step)
            ),
        }

    def _regist_losses(self):
        def dy_lr_coeff(iter_num, alpha=10, power=0.75):
            iter_num = max(iter_num - self.ent_step, 0)
            return np.float((1 + alpha * (iter_num / self.total_steps)) ** (-power))

        optimer = {
            "type": torch.optim.SGD,
            "lr": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "lr_mult": {"F": 0.1},
        }

        decay_op = {
            "type": torch.optim.lr_scheduler.LambdaLR,
            "last_epoch": 0,
            "lr_lambda": lambda step: dy_lr_coeff(step),
        }

        define_loss = partial(self._define_loss, optimer=optimer, decay_op=decay_op)

        define_loss("GlobalDisCls", networks_key=["F", "N_d", "N_c", "D", "C"])

        define_loss("Distangle", networks_key=["N_dd", "N_dc", "D_dd", "D_dc", "M_d", "R_d"])

        define_loss("Rec", networks_key=["R_d"])

        define_loss("Distangle_cls_dis", networks_key=["C", "N_dc"])

        define_loss("Distangle_cls_adv", networks_key=["N_dd"])

        define_loss("Domain_adv", networks_key=["F", "N_c", "Cr", "D_dd"])

    def get_loss(self, imgs, labels=None):

        t = True if labels is None else False

        L = dict()

        """ distengled features """
        feats = self.F(imgs)
        feats_d = self.N_d(feats)
        feats_c = self.N_c(feats)
        feats_dd = self.N_dd(feats_d)
        feats_dc = self.N_dc(feats_d)

        """ recon loss """
        rec_d_feats = self.R_d([feats_dd, feats_dc])
        L["rec"] = {
            "d": torch.sum((feats_d - rec_d_feats) ** 2)
            / (feats_d.shape[0] * feats_d.shape[1])
        }

        """ mutual info loss """
        L["diff"] = {
            "mut": self.M_d.mutual_est(feats_dc, feats_dd),
            "norm": torch.sum(feats_dd * feats_dc) / feats_dd.shape[0],
        }

        """ domain predictions based on different feats """
        # training different domain discriminator
        domain_preds_d = self.D(feats_d)
        domain_preds_dd = self.D_dd(feats_dd)
        domain_preds_dc = self.D_dc(feats_dc)
        # to ensure reconstruct feature is maintain domain information
        domain_preds_d_r = self.D(rec_d_feats)
        # domain adversarial between
        domain_preds_dd_c = self.D_dd(self.Cr(feats_c, adv=True))

        _BCE = partial(self.BCE, target=self.T if t else self.S)
        L["dom"] = {
            "dis_d": _BCE(domain_preds_d),
            "dis_dd": _BCE(domain_preds_dd),
            "dis_dc": _BCE(domain_preds_dc),
            "dis_d_r": _BCE(domain_preds_d_r),
            "adv_dd_c": _BCE(domain_preds_dd_c),
        }
        partial_rec_dc = self.R_d([self.zeros, feats_dc]).detach()
        domain_p_preds_dc = self.D(partial_rec_dc)
        domain_p_preds_dc = torch.sigmoid(domain_p_preds_dc)

        partial_rec_dd = self.R_d([feats_dd, self.zeros]).detach()
        domain_p_preds_dd = self.D(partial_rec_dd)
        domain_p_preds_dd = torch.sigmoid(domain_p_preds_dd)

        """ classify loss """
        b = params.batch_size
        if t:
            rdomain_preds_dc = domain_p_preds_dc
            w = b * (rdomain_preds_dc / rdomain_preds_dc.sum())
            loss_cls = ent(self.C(feats_c), w)
            loss_cls_dc = ent(self.C(feats_dc))
            loss_cls_dd = ent(self.C(feats_dd))
            loss_cls_adv_dd = -loss_cls_dd
        else:
            rdomain_preds_dc = 1 - domain_p_preds_dc
            w = b * (rdomain_preds_dc / rdomain_preds_dc.sum())
            if self.current_step > self.rec_step:
                loss_cls = torch.mean(self.NCE(self.C(feats_c), labels) * w)
            else:
                loss_cls = self.CE(self.C(feats_c), labels)
            loss_cls_dc = self.CE(self.C(feats_dc), labels)
            loss_cls_dd = self.CE(self.C(feats_dd), labels)
            loss_cls_adv_dd = -ent(self.C(feats_dd))

        L["cls"] = {
            "l": loss_cls,
            "cls_dc": loss_cls_dc,
            "cls_dd": loss_cls_dd,
            "cls_adv_dd": loss_cls_adv_dd,
        }

        return L

    def _train_process(self, datas):

        s_imgs, s_labels, t_imgs, t_labels = datas

        engage_rec = True if self.current_step > self.rec_step else False
        engage_ent = True if self.current_step > self.ent_step else False
        engage_adv = True if self.current_step > self.adv_step else False

        Ls = self.get_loss(s_imgs, s_labels)
        Lt = self.get_loss(t_imgs)

        def L(f, s, c=[1, 1]):
            c = c if isinstance(c, list) else [c] * 2
            return (c[0] * Ls[f][s] + c[1] * Lt[f][s]) / 2

        # disentangle losses
        L_diff_mut = L("diff", "mut", 0)  # larger than 0 to active mutual information.
        L_diff_norm = L("diff", "norm", params.c_norm)
        # recon losses
        L_rec_d = L("rec", "d", c=params.c_norm)
        L_dis_d_r = L("dom", "dis_d_r", c=0.1 if engage_rec else 0)
        # discriminator losses
        L_dis_d = L("dom", "dis_d")
        L_dis_dd = L("dom", "dis_dd")
        L_dis_dc = L("dom", "dis_dc")
        L_adv_d_c = L("dom", "adv_dd_c", c=0.3 if engage_adv else 0)
        # classifier losss
        L_cls = L("cls", "l", c=[1, params.c_ent] if engage_ent else [1, 0])
        L_cls_dc = L("cls", "cls_dc", c=[1, 0])
        L_cls_dd = L("cls", "cls_dd", c=[0, 1])
        L_cls_adv_dd = L("cls", "cls_adv_dd", c=[0, 1])

        self._update_losses(
            {
                "GlobalDisCls": L_dis_d + L_cls,
                "Distangle": L_dis_dd + L_dis_dc + L_diff_mut + L_diff_norm + L_rec_d,
                "Rec": L_dis_d_r,
                "Distangle_cls_dis": L_cls_dc + L_cls_dd,
                "Distangle_cls_adv": L_cls_adv_dd,
                "Domain_adv": L_adv_d_c,
            }
        )

        self._update_logs(
            {
                "DomDis/d": L_dis_d,
                "DomDis/dd": L_dis_dd,
                "DomDis/dc": L_dis_dc,
                "DomDis/d_r": L_dis_d_r,
                "DomDis/adv": L_adv_d_c,
                "Ent/t": Lt["cls"]["l"],
                "Cls/c_s": L_cls,
                "Cls/dc": L_cls_dc,
                "Cls/dd": L_cls_dd,
                "rec/d": L_rec_d,
                "mut": L_diff_mut,
            }
        )

    def _eval_process(self, datas):
        imgs, _ = datas
        feats = self.F(imgs)

        feats_dc = self.N_dc(self.N_d(feats))
        partial_rec_feats = self.R_d([self.zeros, feats_dc])
        domain_preds_d = self.D(partial_rec_feats)
        domain_preds_d = torch.sigmoid(domain_preds_d).squeeze()

        feats_di = self.N_c(feats)
        preds = F.softmax(self.C(feats_di), dim=-1)
        props, predcition = torch.max(preds, dim=1)
        predcition[props < 0.2] = self.cls_info["cls_num"]
        return predcition
