from typing import DefaultDict
import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn.modules import instancenorm
from torch.nn.modules.activation import ReLU6

def get_op_names(model):
    op_dict = DefaultDict(list)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            op_dict['conv2d'].append(name)
        elif isinstance(module, nn.Linear):
            op_dict['fc'].append(name)
        elif isinstance(module, nn.ReLU):
            op_dict['relu'].append(name)
        elif isinstance(module, ReLU6):
            op_dict['relu6'].append(name)
    return op_dict

if __name__ == '__main__':
    model = models.resnet18()
    op_dict = get_op_names(model)
    
    print(op_dict.values())