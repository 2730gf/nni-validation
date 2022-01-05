"""
Usage:
    python nni_qat_practice.py
Ps:
    这个代码使用的是nni的QAT量化代码
    
"""
from shutil import disk_usage
import time
import logging

import argparse
import os

from copy import deepcopy
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
import torchvision.models as models

from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer, LsqQuantizer, DoReFaQuantizer
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.quantization.settings import set_quant_scheme_dtype

import time
import pdb
from absl import logging
logging.set_verbosity(logging.WARNING) 

from models.cifar10.resnet import ResNet101, ResNet18, ResNet50
from utils.get_op_names import get_op_names

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, 1)
        self.conv2 = torch.nn.Conv2d(5, 5, 3,1)
        
        self.relu1 = torch.nn.ReLU6()
        self.relu2 = torch.nn.ReLU6()
        
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.bn2 = torch.nn.BatchNorm2d(5)
        
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)
        
        self.sf = nn.Softmax(dim=1)
        self.fc = nn.Linear(180, 10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1)
        x = self.sf(x)
        x = self.fc(x)
        return x

    
def save_qat_onnx(model, dummy_input, save_path):

    # Export the model
    torch.onnx.export(model,               # model being run
                    dummy_input,                         # model input (or a tuple for multiple inputs)
                    save_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names=['output'],  # the model's output names
                    dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}}
    )

def get_data(data_dir, batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)
    criterion = torch.nn.CrossEntropyLoss()
    return train_loader, test_loader, criterion

def get_model_optimizer(args, device):
    if args.model == "res18":
        model = ResNet18()
    elif args.model == 'res101':
        model = ResNet101()
    elif args.model == 'res50':
        model = ResNet50()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    if os.path.exists(args.pretrained_model_dir):
        print("load from state_dict: {}".format(args.pretrained_model_dir))
        model.load_state_dict(torch.load(args.pretrained_model_dir,map_location=device), strict=False) 
    return model, optimizer, scheduler

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 120 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    print('Test Loss: {}  Accuracy: {}%\n'.format(
        test_loss, acc))
    return acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)   

    # prepare model and data
    train_loader, test_loader, criterion = get_data('./data', args.batch_size, args.test_batch_size)
    model, optimizer, scheduler = get_model_optimizer(args, device) 
    # pdb.set_trace()
    
    op_dict = get_op_names(model)
    print(op_dict)
    test(model, device, criterion, test_loader)
    if args.qat:
        # SUPPORTED_OPS = ['Conv2d', 'Linear', 'ReLU', 'ReLU6']
        config_list = [{
        'quant_types': ['weight', 'input'],
        'quant_bits': {'weight': 8, 'input': 8,},
            'op_names': op_dict['conv2d']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8, },
            'op_names': op_dict['relu']
        }, {
            'quant_types': ['output', 'weight', 'input'],
            'quant_bits': {'output': 8, 'weight': 8, 'input': 8},
            'op_names': op_dict['fc'],
        }]
        # 选择是否per_channel来做量化
        set_quant_scheme_dtype('weight', 'per_channel_symmetric', 'int')
        set_quant_scheme_dtype('output', 'per_tensor_symmetric', 'int')
        set_quant_scheme_dtype('input', 'per_tensor_symmetric', 'int')

        dummy_input = torch.randn(3,3,32,32).to(device)
        quantizer = QAT_Quantizer(model, config_list, optimizer, dummy_input=dummy_input)
        # quantizer = LsqQuantizer(model, config_list, optimizer, dummy_input=dummy_input)
        quantizer.compress()

    train_time  =time.time()
    best_acc = 0
    
    if args.qat:
        save_path = os.path.join(args.experiment_data_dir, "{}_nniqat_finetune.pth".format(args.model))
    else:
        save_path = os.path.join(args.experiment_data_dir, "{}_best.pth".format(args.model))
    
    test(model, device, criterion, test_loader)
    pdb.set_trace()
    for epoch in range(args.train_epochs):
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        acc = test(model, device, criterion, test_loader)
        # pdb.set_trace()
        if acc > best_acc:
            if not args.qat:
                torch.save(model.state_dict(), save_path)
                best_acc = acc
            else:
                state_dict = model.state_dict()
                best_acc = acc
    
    if args.qat:            
        model.load_state_dict(state_dict)
        calibration_path = "./exp/{}_nni_calibration.pth".format(args.model)
        pdb.set_trace()
        calibration_config = quantizer.export_model(save_path, calibration_path) # , onnx_path, input_shape, device)
        # print("Generated calibration config is: ", calibration_config) 
        pdb.set_trace()
    acc = test(model, device, criterion, test_loader)
    train_time = time.time() - train_time
    print("train_time: {:.4f}s".format(train_time))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="qat practice")

    parser.add_argument('--model', default="res18", type=str,
        choices=["res18", "res50", "res101",])    
    
    parser.add_argument("--pretrained_model_dir", default="./exp/pretrain_cifar10_resnet18.pth", type=str)
    parser.add_argument("--experiment_data_dir", default="./exp", type=str)
    
    parser.add_argument("--train_epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--test_batch_size", default=200, type=int)

    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--qat", action="store_true")

    args = parser.parse_args()

    # Debug TODO
    # args.model = "res18"
    # args.pretrained_model_dir = "./exp/pretrain_cifar10_resnet18.pth"
    # args.experiment_data_dir = "./exp"
    
    # args.train_epochs = 120

    # args.lr = 0.01
    # args.batch_size = 256
    # args.test_batch_size = 200

    # args.test_only = False
    args.qat = True
    
    
    main(args)