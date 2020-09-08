
import os
from mtrain.watcher import watcher
from mtrain.loss_driven.lossholder import LossChangeListener, LossBuket
from torch.utils.tensorboard import SummaryWriter
from functools import partial
# writer = SummaryWriter()

class LogCapsule(LossChangeListener):
    def __init__(
        self,
        loss_bucker: LossBuket,
        name,
        file_writer = None,
        step=1,
    ):

        writer = None
        if file_writer:
            writer = partial(file_writer.add_scalar, tag=name)
        self.writer = writer

        self.tag = name
        self.current_step = 0
     
        self.range_loss = 0.0
        self.range_step = 0.0

        loss_bucker.add_lister(self)

    def avg_range_loss(self):
        try:
            result = self.range_loss / self.range_step
        except:
            result = 0.0
        self.range_loss = 0.0
        self.range_step = 0.0
        self.range_step = 0.0
        try:
            loss = result.item()
        except:
            loss = result
        # if self.writer
        return loss

    def before_change(self):
        pass
    
    def in_change(self, value):
        try:
            value = value.detach()
        except:
            value = value
        
        self.range_loss += value


    def after_change(self):
        self.current_step += 1
        self.range_step += 1


if __name__ == "__main__":
    pass

