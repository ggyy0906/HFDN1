import torch
import torch.nn as nn
import torch.nn.init as init

from mground.gpu_utils import anpai

import os

from abc import ABC, abstractclassmethod

from mtrain.loss_driven.lossholder import LossHolder
from mtrain.loss_driven.loger import LogCapsule
from mtrain.loss_driven.trainer import TrainCapsule

from mtrain.recorder.info_print import tabulate_print_losses
from mtrain.recorder.info_print import cprint, HINTS, WARMN, VALID, TRAIN, BUILD

from mdata.data_iter import EndlessIter


def _basic_weights_init_helper(modul, params=None):
    """give a module, init it's weight
    
    Args:
        modul (nn.Module): torch module
        params (dict, optional): Defaults to None. not used.
    """

    for m in modul.children():
        # init Conv2d with norm
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            init.constant_(m.bias, 0)
        # init BatchNorm with norm and constant
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                init.normal_(m.weight, mean=1.0, std=0.02)
                init.constant_(m.bias, 0)
        # init full connect norm
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

        elif isinstance(m, nn.Module):
            _basic_weights_init_helper(m)

        if isinstance(m, WeightedModule):
            m.has_init = True

class TrainableModule(ABC):

    """ An ABC, a Tranable Module is a teample class need to define
    data process, train step and eval_step

    """

    def __init__(self, params):

        super(TrainableModule, self).__init__()

        self.params = params

        self.writer = None

        # all key point of training steps
        self.log_step = self.params.log_per_step
        self.eval_step = self.params.eval_per_step
        self.eval_after_step = -1
        self.total_steps = self.params.steps
        self.current_step = 0.0
        self.current_epoch = 0.0
        self.need_pre_eval = False
        self.has_pre_eval = False
        self.best_accurace = 0.0

        # loss changing driven
        self.loss_holder = LossHolder()
        self.train_caps = dict()
        self.train_loggers = dict()
        self.proce_loggers = dict()
        self.valid_loggers = dict()

        # get all networks and init weights
        networks = self._regist_networks()
        assert type(networks) is dict

        def init_weight_and_key(n, k):
            # n.weight_init()
            n.tag = k

        for k, i in networks.items():
            if type(i) is nn.Sequential:
                i.tag = k
                for c in i.children():
                    init_weight_and_key(c, k)
            else:
                init_weight_and_key(i, k)

        # send networks to gup
        networks = {
            i: anpai(j, use_gpu=self.params.use_gpu)
            for i, j in networks.items()
        }

        # make network to be class attrs
        for i, j in networks.items():
            self.__setattr__(i, j)
        self.networks = networks

        # generate train dataloaders and valid dataloaders
        # data_info is a dict contains basic data infomations
        cls_num, data_fn = self._prepare_data()
        confusion_matrix = torch.zeros(cls_num, cls_num)

        self.confusion_matrix = confusion_matrix
        self.data_feeding_fn = data_fn
        self._define_log("valid_accurace", group="valid")

        # regist losses
        self._regist_losses()

    def _all_ready(self):
        raise Exception("Not Use Anymore")

    @abstractclassmethod
    def _prepare_data(self):
        """ handle dataset to produce dataloader
        
        Returns:
            list -- a dict of datainfo, a list of train dataloader
            and a list of valid dataloader.
        """

        data_info = dict()
        train_loaders = list()
        valid_loaders = list()
        return data_info, train_loaders, valid_loaders

    # @abstractclassmethod
    def _feed_data(self, mode):
        """ feed example based on dataloaders

        Returns:
            list -- all datas needed.
        """
        return self.data_feeding_fn(mode)

    @abstractclassmethod
    def _regist_losses(self):
        """ regist lossed with the help of regist loss

        Returns:
            list -- all datas needed.
        """
        return

    @abstractclassmethod
    def _regist_networks(self):
        """ feed example based on dataloaders

        Returns:
            list -- all datas needed.
        """
        networks = dict()
        return networks

    @abstractclassmethod
    def _train_process(self, datas, **kwargs):
        """process to train 
        """
        pass

    @abstractclassmethod
    def _eval_process(self, datas, **kwargs):
        """process to eval 
        """
        end_epoch = datas is None

        pass

    def _feed_data_with_anpai(self, mode):
        data = self._feed_data(mode)
        if data is not None:
            data = anpai(data, self.params.use_gpu, need_logging=False)
        return data

    def train_module(self, **kwargs):

        ls, vs = 0, 0
        add1 = lambda x: x + 1

        for _ in range(self.total_steps):

            # set all networks to train mode
            for _, i in self.networks.items():
                i.train(True)

            datas = self._feed_data_with_anpai(mode="train")
            self._train_process(datas, **kwargs)
            ls, vs, self.current_step = map(add1, [ls, vs, self.current_step])

            # log training
            if ls == self.log_step:
                ls = 0
                self._handle_loss_log(mode="process")
                self._handle_loss_log(mode="train")


            # begain eval
            if vs == self.eval_step:
                vs = 0
                self.eval_module(**kwargs)
                for _, i in self.networks.items():
                    i.train(True)

    def eval_module(self, **kwargs):

        # set all networks to eval mode
        for _, i in self.networks.items():
            i.eval()

        # reset confusion matrix
        self.confusion_matrix.fill_(0)

        while True:
            datas = self._feed_data_with_anpai(mode="valid")
            if datas is None:
                break
            prediction = self._eval_process(datas)
            targets = datas[-1]
            for t, p in zip(targets.view(-1), prediction.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1

        accurace = None
        cm = self.confusion_matrix
        accurace = cm.diag().sum() / cm.sum()

        self._update_loss("valid_accurace", accurace)

        cprint(VALID, "End a evaling step.")
        self._handle_loss_log(mode="valid")

    def _define_loss(
        self, loss_name, networks_key, optimer: dict, decay_op: dict = None
    ):
        """registe loss according loss_name and relative networks.
        after this process the loss will bind to the weight of networks, which means this loss will used to update weights of provied networks.
        
        Arguments:
            loss_name {str} -- loss name
            networks_key {list} -- relative network names
        """

        networks_key = (
            [networks_key]
            if not isinstance(networks_key, (tuple, list))
            else networks_key
        )

        networks = [self.networks[i] for i in networks_key]

        optimer = optimer.copy()
        optimer_info = [optimer.pop("type"), optimer]

        decay_info = None
        if decay_op is not None:
            decay_op = decay_op.copy()
            decay_info = [decay_op.pop("type"), decay_op]

        loss_buck = self.loss_holder.new_loss(loss_name)
        t = TrainCapsule(
            loss_buck,
            networks,
            optimer_info=optimer_info,
            decay_info=decay_info,
        )

        self._define_log(loss_name, group="train")

        self.train_caps[loss_name] = t

    def _define_log(self, *loss_name, group="train"):
        if group == "train":
            step = self.log_step
            group = self.train_loggers
        elif group == "process":
            step = self.log_step
            group = self.proce_loggers
        else:
            step = self.eval_step
            group = self.valid_loggers

        for name in loss_name:
            loss_buck = self.loss_holder.get_loss(name)
            group[name] = LogCapsule(
                loss_buck, name, step=step, file_writer=self.writer
            )

    def _update_loss(self, loss_name, value):
        self.loss_holder.update_loss(loss_name, value)

    def _update_losses(self, a: dict):
        for key in a:
            self._update_loss(key, a[key])
    
    def _update_logs(self, a: dict):
        if self.current_step == 0:
            for i in a:
                self._define_log(i, group="process")
        self._update_losses(a)

    def save(self):
        nets = self.networks
        torch.save(
            {i: nets[i].state_dict() for i in nets},
            'saved/test_module.pt'
        )

        trainers = self.train_caps
        torch.save(
            {i: trainers[i].optimer.state_dict() for i in trainers},
            'saved/test_optimer.pt'
        )

    def _handle_loss_log(self, mode):
        if mode == "train":
            loggers = self.train_loggers
        elif mode == "valid":
            loggers = self.valid_loggers
        elif mode == "process":
            loggers = self.proce_loggers
        else:
            raise Exception("Not such loggers!")

        losses = {
            k: v.avg_range_loss() for k, v in loggers.items()
        }

        for loss_name in losses:
            tags = loss_name.split("/", 1)
            if len(tags) == 1:
                tag = tags[0]
                self.writer.add_scalar(
                    mode + "/" + tag, losses[loss_name], self.current_step
                )
            else:
                pre, sub = tags
                self.writer.add_scalars(
                    mode + "/" + pre,
                    {sub: losses[loss_name]},
                    self.current_step,
                )


    def clean_up(self):
        if self.writer:
            self.writer.close()

