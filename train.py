# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models import resnet101,resnet50
from torch.utils.data import DataLoader
import time
import argparse
import math
import json
import pickle
import numpy as np
from torchnet import meter
from PIL import ImageFile
from afsnet import AFSNet, _Graph, Bottleneck

ImageFile.LOAD_TRUNCATED_IMAGES = True

from data.load import load_datasets

parser = argparse.ArgumentParser(description='DELTA')
parser.add_argument('--data_dir')
parser.add_argument('--save_model', default='')
parser.add_argument('--base_model', choices=['resnet101','resnet50'], default='resnet101')
parser.add_argument('--max_iter', type=int, default=9000)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr_scheduler', choices=['steplr', 'explr'], default='steplr')
parser.add_argument('--lr_init', type=float, default=0.01)
parser.add_argument('--reg_type', choices=['l2', 'l2_sp', 'fea_map', 'att_fea_map'], default='l2')
parser.add_argument('--channel_wei', default='')
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--ratio', type=float, default=0.5)
args = parser.parse_args()

print(args)
device = torch.device("cuda:0")


# Load dataset
set_name = 'stanford_dogs'
train_data, test_data, class_names = load_datasets(set_name=set_name, input_size=args.image_size)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
dataloaders = {'train': train_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_data), 'test': len(test_data)}
num_classes = len(class_names)

# Initial ratio
_Graph.set_global_var('ratio', args.ratio)

# Define model
def get_base_model(base_model):
    return eval(base_model)(pretrained=True)

model_source = get_base_model(args.base_model)
model_source.to(device)
for param in model_source.parameters():
    param.requires_grad = False
model_source.eval()

model_source_weights = {}
for name, param in model_source.named_parameters():
    model_source_weights[name] = param.detach()

model_pretrained = models.resnet101(pretrained=True)
model_target = AFSNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
pretrained_dict = model_pretrained.state_dict()
print(pretrained_dict.__len__())
model_dict = model_target.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
print(pretrained_dict.__len__())
model_dict.update(pretrained_dict)
model_target.load_state_dict(model_dict)
print(model_dict.__len__())

#model_target=models.resnet101(pretrained=True)
# model_target = get_base_model(args.base_model)
# model_target.fc = nn.Linear(2048, num_classes)
model_target.to(device)

channel_weights = []
if args.reg_type == 'att_fea_map' and args.channel_wei:
    for js in json.load(open(args.channel_wei)):
        js = np.array(js)
        js = (js - np.mean(js)) / np.std(js)
        cw = torch.from_numpy(js).float().to(device)
        cw = F.softmax(cw / 5).detach()
        channel_weights.append(cw)

layer_outputs_source = []
layer_outputs_target = []


def for_hook_source(module, input, output):
    layer_outputs_source.append(output)


def for_hook_target(module, input, output):
    layer_outputs_target.append(output)


fc_name = 'fc.'
if args.base_model == 'resnet101':
    hook_layers = ['layer1.2.conv3', 'layer2.3.conv3', 'layer3.22.conv3', 'layer4.2.conv3']
elif args.base_model == 'resnet50':
    hook_layers = ['layer1.2.conv3', 'layer2.3.conv3', 'layer3.22.conv3', 'layer4.2.conv3']
else:
    assert False


def register_hook(model, func):
    for name, layer in model.named_modules():
        if name in hook_layers:
            layer.register_forward_hook(func)


register_hook(model_source, for_hook_source)
register_hook(model_target, for_hook_target)


def reg_classifier(model):
    l2_cls = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if name.startswith(fc_name):
            l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls


def reg_l2sp(model):
    fea_loss = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if not name.startswith(fc_name):
            fea_loss += 0.5 * torch.norm(param - model_source_weights[name]) ** 2
    return fea_loss


def reg_fea_map(inputs):
    _ = model_source(inputs)
    fea_loss = torch.tensor(0.).to(device)
    for fm_src, fm_tgt in zip(layer_outputs_source, layer_outputs_target):
        b, c, h, w = fm_src.shape
        fea_loss += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
    return fea_loss


def reg_att_fea_map(inputs):
    _ = model_source(inputs)
    fea_loss = torch.tensor(0.).to(device)
    for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)
        fm_tgt = flatten_outputs(fm_tgt)
        div_norm = h * w
        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
        distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)
        fea_loss += 0.5 * torch.sum(distance)
    return fea_loss


def flatten_outputs(fea):
    return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))



def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            nstep = len(dataloaders[phase])
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # if args.data_aug == 'improved' and phase == 'test':
                #     bs, ncrops, c, h, w = inputs.size()
                #     inputs = inputs.view(-1, c, h, w)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # if args.data_aug == 'improved' and phase == 'test':
                    #     outputs = outputs.view(bs, ncrops, -1).mean(1)
                    loss_main = criterion(outputs, labels)
                    loss_classifier = 0
                    loss_feature = 0
                    if not args.reg_type == 'l2':
                        loss_classifier = reg_classifier(model)
                    if args.reg_type == 'l2_sp':
                        loss_feature = reg_l2sp(model)
                    elif args.reg_type == 'fea_map':
                        loss_feature = reg_fea_map(inputs)
                    elif args.reg_type == 'att_fea_map':
                        loss_feature = reg_att_fea_map(inputs)
                    loss = loss_main + args.alpha * loss_feature + args.beta * loss_classifier

                    _, preds = torch.max(outputs, 1)
                    #confusion_matrix.add(preds.data, labels.data)
                    if phase == 'train' and i % 10 == 0:
                        corr_sum = torch.sum(preds.data == labels.data)
                        step_acc = corr_sum.double() / len(labels)
                        print('step: %d/%d, loss = %.4f(%.4f, %.4f, %.4f), top1 = %.4f' % (
                            i, nstep, loss, loss_main, args.alpha * loss_feature, args.beta * loss_classifier,
                            step_acc))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.data == labels.data)

                layer_outputs_source.clear()
                layer_outputs_target.clear()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} epoch: {:d} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch, epoch_loss, epoch_acc))
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            if epoch == num_epochs - 1:
                print('{} epoch: last Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'train' and abs(epoch_loss) > 1e8:
                break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


if args.reg_type == 'l2':
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_target.parameters()),
                             lr=args.lr_init, momentum=0.9, weight_decay=1e-4)
else:
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_target.parameters()),
                             lr=args.lr_init, momentum=0.9)

num_epochs = int(args.max_iter * args.batch_size / dataset_sizes['train'])
decay_epochs = int(0.67 * args.max_iter * args.batch_size / dataset_sizes['train']) + 1
print('StepLR decay epochs = %d' % decay_epochs)

if args.lr_scheduler == 'steplr':
    lr_decay = optim.lr_scheduler.StepLR(optimizer_ft, step_size=decay_epochs, gamma=0.1)
elif args.lr_scheduler == 'explr':
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=math.exp(math.log(0.1) / decay_epochs))

criterion = nn.CrossEntropyLoss()
train_model(model_target, criterion, optimizer_ft, lr_decay, num_epochs)

if args.save_model != '':
    torch.save(model_target, args.save_model)
