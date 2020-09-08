
import numpy as np

import matplotlib.pyplot as plt

def show_a_tenosr(tensor):
    plt.imshow(tensor.permute(1, 2, 0))
    plt.show()

