import torch
from torch import nn
import numpy
from lib.model.utils.config import cfg

class anchor_target_layer(nn.Module):
    def __init__(self):
        super(anchor_target_layer,self).__init__()

