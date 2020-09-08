from torch import optim as optim
import torch


from abc import ABC, abstractclassmethod

class LossChangeListener(ABC):
    
    @abstractclassmethod
    def before_change(self):
        pass
    
    @abstractclassmethod
    def in_change(self, value):
        pass
    
    @abstractclassmethod
    def after_change(self):
        pass


class LossBuket(object):
    def __init__(self):
        self.value = None
        self.listener = list()

    def add_lister(self, l: LossChangeListener):
        self.listener.append(l)

    def update(self, value):
        for i in self.listener:
            i.before_change()
        
        self.value = value
        for i in self.listener:
            i.in_change(value)
        
        for i in self.listener:
            i.after_change()

    def loss(self):
        raise Exception('no use anymore!!!')
        return self.value


class LossHolder(object):
    def __init__(self):
        self.loss_dic = dict()
        self.fixed = False

    def new_loss(self, name):
        self.loss_dic[name] = LossBuket()
        return self.loss_dic[name]
    
    def get_loss(self, name, need_create=True):
        if name not in self.loss_dic:
            if need_create:
                self.loss_dic[name] = LossBuket()        
        return self.loss_dic.get(name, None)

    def update_loss(self, name, value):
        self.loss_dic[name].update(value)


if __name__ == "__main__":

    pass

