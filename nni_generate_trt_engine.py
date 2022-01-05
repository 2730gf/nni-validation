"""
Usage:
    python nni_generate_trt_engine.py
Ps:
    这个代码用来生成tensor engine
"""
from shutil import disk_usage
import time
import logging

import argparse
import os

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
import torchvision.models as models

from quantization_speedup import ModelSpeedupTensorRT 

import time
import pdb
from absl import logging
logging.set_verbosity(logging.WARNING) 
from models.cifar10.resnet import ResNet18, ResNet50, ResNet101
    
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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(
                optimizer, milestones=[int(args.train_epochs * 0.5), int(args.train_epochs * 0.75)], gamma=0.1)

    if os.path.exists(args.pretrained_model_dir):
        print("load from state_dict: {}".format(args.pretrained_model_dir))
        model.load_state_dict(torch.load(args.pretrained_model_dir, map_location=device), strict=False)

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

def test_trt(engine, test_loader):
    test_loss = 0
    correct = 0
    time_elasped = 0
    for data, target in test_loader:
        output, time = engine.inference(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        time_elasped += time
    test_loss /= len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))
    print("Inference elapsed_time (whole dataset): {}s".format(time_elasped))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)   

    # prepare model and data
    train_loader, test_loader, criterion = get_data('./data', args.batch_size, args.test_batch_size)
    model, optimizer, scheduler = get_model_optimizer(args, device) 
    
    pth_acc = test(model, device, criterion, test_loader)
    calibration_config = torch.load(args.calib_path, map_location="cpu")
    # Model Speedup
    batch_size = 32
    input_shape = (batch_size, 3, 32, 32)
    engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=32, calibration_cache = "./calib/model.calib")
    engine.compress()
    engine.export_quantized_model("./exp/resnet18.trt")
    
    # 测试trt精度
    test_trt(engine, test_loader)
    print("pth acc: {}".format(pth_acc))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="qat practice")

    parser.add_argument('--model', default="res18", type=str,
        choices=["res18", "res50", "res101",])    
    
    parser.add_argument("--pretrained_model_dir", default="./exp/pretrain_cifar10_resnet18.pth", type=str)
    parser.add_argument("--experiment_data_dir", default="./exp", type=str)
    parser.add_argument("--calib_path", default="./exp/res18_nni_calibration.pth", type=str)
    
    parser.add_argument("--train_epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--test_batch_size", default=200, type=int)

    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--qat", action="store_true")

    args = parser.parse_args()

    # Debug TODO
    # args.model = "res18"
    # args.pretrained_model_dir = "./exp/res18_best.pth"
    # args.experiment_data_dir = "./exp"
    
    # args.train_epochs = 120

    # args.lr = 0.01
    # args.batch_size = 256
    # args.test_batch_size = 200

    # args.test_only = False
    args.qat = True
    args.pretrained_model_dir = "./exp/res18_nniqat_finetune.pth"

    main(args)