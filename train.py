"""
Usage:
    python train.py
Ps:
    
"""

import logging

import argparse
import os
from copy import deepcopy
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
import torchvision.models as models

from apex.contrib.sparsity import ASP
import pdb

def save_onnx(model, dummy_input, save_path):
    
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

def get_model_optimizer_scheduler(args, device, train_loader, test_loader, criterion):

    model = models.resnet18(pretrained=True).to(device)
    if args.pretrained_model_dir is None and not args.test_only:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[int(args.pretrain_epochs * 0.5), int(args.pretrain_epochs * 0.75)], gamma=0.1)
    
        print('start pre-training...')
        best_acc = 0
        for epoch in range(args.pretrain_epochs):
            train(model, device, train_loader, criterion, optimizer, epoch)
            scheduler.step()
            acc = test(model, device, criterion, test_loader)
            if acc > best_acc:
                best_acc = acc
                state_dict = model.state_dict()

        model.load_state_dict(state_dict)
        acc = best_acc

        torch.save(state_dict, os.path.join(args.experiment_data_dir, f'pretrain_cifar10_resnet18.pth'))
        print('Model trained saved to %s' % args.experiment_data_dir)

    elif not args.test_only:
        model.load_state_dict(torch.load(args.pretrained_model_dir, map_location=device))
        best_acc = test(model, device, criterion, test_loader)
    
        print('Pretrained model acc:', best_acc)
    
    # setup new opotimizer for pruning
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(args.finetune_epochs * 0.5), int(args.finetune_epochs * 0.75)], gamma=0.1)

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
    
    model, optimizer, scheduler = get_model_optimizer_scheduler(args, device, train_loader, test_loader, criterion)
    
    dummy_input = torch.randn(1,3,224,224).cuda()
    onnx_path = os.path.join(args.experiment_data_dir, f'resnet18.onnx')
    save_onnx(model, dummy_input, onnx_path)
    
    pdb.set_trace()
    # sparse model
    ASP.prune_trained_model(model, optimizer)
    pdb.set_trace()
    
    save_path = os.path.join(args.experiment_data_dir, f'finetuned.pth')
    onnx_path = os.path.join(args.experiment_data_dir, f'resnet18_sparse.onnx')
    save_onnx(model, dummy_input, onnx_path)

    for epoch in range(args.finetune_epochs):
        print('# Epoch {} #'.format(epoch))

        train(model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        top1 = test(model, device, criterion, test_loader)
        best_top1 = 0 
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), save_path)
            save_onnx(model, dummy_input, onnx_path)
        pdb.set_trace()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')
    args = parser.parse_args()
    
    # Debug TODO
    args.pretrained_model_dir = "./exp/pretrain_cifar10_resnet18.pth"
    args.experiment_data_dir = "./exp"
    
    args.pretrain_epochs = 120
    args.finetune_epochs = 120

    args.batch_size = 128
    args.test_batch_size = 200

    args.test_only = True
    
    main(args)
