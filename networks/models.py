import os
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from networks.network_utils import ConvBlock
from networks.network_utils import ResnetBlock


##############################################################################
# Custom Model
##############################################################################


class CustomNetwork(nn.Module):
    def __init__(self, input_nc=3, **kwargs):
        super(CustomNetwork, self).__init__()
        pass

        ###
        # Network layers init
        ###

    def forward(self, input_tensor):
        pass

        ###
        # Network structure and sequence
        ###

        return output_tensor