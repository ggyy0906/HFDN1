from torch import nn
from torch.nn import functional as F
class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, c=1):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b * c
        b = -1.0 * b.sum()
        return b