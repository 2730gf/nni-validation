"""
Usage:
    python qat_practice.py
Ps:
    这个代码使用的是nvidia提供的pytorch_quantization
    个人觉得，缺点:在于没有融合BN层，所以导致精度还可以进一步提升
    优点:添加量化的节点非常方便
"""
import time
import logging

import argparse
import os
os.chdir("/home/gongfu/workspace/baidu/personal-code/qat-validation")

from copy import deepcopy
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
import torchvision.models as models

import pdb
import time

import pdb
from absl import logging
logging.set_verbosity(logging.WARNING) 

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
#quant_nn.TensorQuantizer.use_fb_fake_quant = True


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
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512,10)
    elif args.model == 'res101':
        model = models.resnet101(pretrained=True)
        model.fc = torch.nn.Linear(2048, 10)
    elif args.model == 'res50':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 10)
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(
                optimizer, milestones=[int(args.train_epochs * 0.5), int(args.train_epochs * 0.75)], gamma=0.1)

    if os.path.exists(args.pretrained_model_dir):
        print("load from state_dict: {}".format(args.pretrained_model_dir))
        model.load_state_dict(torch.load(args.pretrained_model_dir,map_location=device))

    return model, optimizer, scheduler

def init_calib_from_data_loader(model, data_loader, num_batches=10):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # pdb.set_trace()
            if module.axis is not None:
                print("weight module not fixed for finetune.")
                continue
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    # Feed data to the network for collecting stats
    #from tqdm import tqdm
    #for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
    for i, (image, _) in enumerate(data_loader):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module.axis is not None:
                continue
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
                module.load_calib_amax()
            else:
                module.enable()
            print(F"{name:40}: {module}")


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
    
    # qat初始化
    if args.qat:
        quant_modules.initialize()

    # prepare model and data
    train_loader, test_loader, criterion = get_data('./data', args.batch_size, args.test_batch_size)
    
    # normal train
    model, optimizer, scheduler = get_model_optimizer(args, device) 
    print(model)
    pdb.set_trace()

    save_path = os.path.join(args.experiment_data_dir, "{}_best.pth".format(args.model))

    if args.qat:
        with torch.no_grad():
            init_calib_from_data_loader(model, train_loader)
        save_path = os.path.join(args.experiment_data_dir, "{}_nvqat_finetune.pth".format(args.model))
    
    best_acc = 0       
    train_time  =time.time()
    test(model, device, criterion, test_loader)
    for epoch in range(args.train_epochs):
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        acc = test(model, device, criterion, test_loader)
        
        if acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = acc

    train_time = time.time() - train_time
    print("train_time: {:.4f}s".format(train_time))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="qat practice")

    parser.add_argument('--model', default="res18", type=str,
        choices=["res18", "res50", "res101"])
    
    parser.add_argument("--pretrained_model_dir", default="./exp/res18_best.pth", type=str)
    parser.add_argument("--experiment_data_dir", default="./exp", type=str)
    
    parser.add_argument("--train_epochs", default=120, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--test_batch_size", default=200, type=int)

    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--check", action="store_true")

    
    # parser.add_argument("model", default="res101", Optional)
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
    # args.check = True
    # args.qat = True
    
    main(args)