import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pdb
from absl import logging
logging.set_verbosity(logging.WARNING) 

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
#quant_nn.TensorQuantizer.use_fb_fake_quant = True

quant_modules.initialize()

model = models.resnet101(pretrained=True)
dummy_input = torch.randn(10, 3, 224, 224)
print(model)
model(dummy_input)