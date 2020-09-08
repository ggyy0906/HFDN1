# request: Python3.7, Pytorch1.2, TensorBoard
# place dataset at folder ./DATASET/[DATASET_NAME]

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf_port = 8008

import torch
from main_aid import TBHandler
from mmodel import get_module

if __name__ == "__main__":

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(000000)
    torch.cuda.manual_seed_all(000000)

    model_name = "TFDN"
    tb = TBHandler(model_name)
    param, model = get_module(model_name)

    try:
        model.writer = tb.get_writer()
        tb.star_shell_tb(tf_port)
        model.train_module()
    finally:
        tb.kill_shell_tb()
        raise

