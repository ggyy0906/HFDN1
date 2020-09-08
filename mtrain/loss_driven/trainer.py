from mtrain.loss_driven.lossholder import LossChangeListener
import torch

class TrainCapsule(LossChangeListener):
    """this is a tool class for helping training a network
    """
    
    def __init__(
        self,
        optim_loss,
        optim_networks,
        optimer_info,
        decay_info,
        tagname=None,
    ):
        super(TrainCapsule, self).__init__()

        self.tag = tagname
        self.optim_loss = optim_loss

        # get all networks, and store them as list
        if not isinstance(optim_networks, (tuple, list)):
            networks_list = list()
            networks_list.append(optim_networks)
        else:
            networks_list = optim_networks
        self.optim_network = networks_list

        # get all parameters in network list
        self.all_params = list()
        optimer_type, optimer_kwargs = optimer_info

        base_lr = optimer_kwargs["lr"]
        base_decay = optimer_kwargs.get('weight_decay', 0)
        lr_mult_map = optimer_kwargs.pop("lr_mult", dict())
        assert type(lr_mult_map) is dict


        for i in networks_list:
            if isinstance(i, torch.nn.DataParallel):
                i = i.module
            lr_mult = lr_mult_map.get(i.tag, 1)
            param_info = [{
                "params": i.parameters(),
                "lr_mult": lr_mult,
                "lr": lr_mult * base_lr,
                "initial_lr": lr_mult * base_lr,
                'tag' : i.tag,
            },]
            self.all_params += param_info
        
        # init optimer base on type and args
        self.optimer = optimer_type(self.all_params, **optimer_kwargs)

        # init optimer decay option
        self.lr_scheduler = None
        if decay_info is not None:
            decay_type, decay_arg = decay_info
            self.lr_scheduler = decay_type(self.optimer, **decay_arg)
        
        # regist to master
        optim_loss.add_lister(self)

    def before_change(self):
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        # for param_group in self.optimer.param_groups:
        #     print(param_group['lr'])
        self.optimer.zero_grad()
        return 
        
    def in_change(self, value):
        self.optim_loss.value.backward(retain_graph=True)
        self.optimer.step()

    def after_change(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # for param_group in self.optimer.param_groups:
        #     print(param_group['lr'])


    ##UGLY wait to delete

    def train_step(self, retain_graph=True):
        raise Exception('not use anymore!!')
        self.optimer.zero_grad()
        self.optim_loss.value.backward(retain_graph=retain_graph)
        self.optimer.step()

    def make_zero_grad(self):
        raise Exception('not use anymore!!')
        self.optimer.zero_grad()

    def decary_lr_rate(self):
        raise Exception('not use anymore!!')
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
