import torchvision.models as models
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class TorchGraph(object):
    def __init__(self):
        self._graph = {}

    def add_tensor_list(self, name):
        self._graph[name] = []

    def append_tensor(self, name, val):
        self._graph[name].append(val)

    def clear_tensor_list(self, name):
        self._graph[name].clear()

    def get_tensor_list(self, name):
        return self._graph[name]

    def set_global_var(self, name, val):
        self._graph[name] = val

    def get_global_var(self, name):
        return self._graph[name]


_Graph = TorchGraph()
#_Graph.add_tensor_list('gate_values')
_Graph.set_global_var('ratio', 1.0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.ratio = 1.0  # refer to d in winner take all
        # 1x1 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gate1 = nn.Linear(inplanes, planes)  # refer to hl fc layer
        self.s1 = 1.0
        self.m1 = 1.0
        self.gamma1 = nn.Parameter(torch.rand(planes))  # refer to a trainable gamma

        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate2 = nn.Linear(planes, planes)  # refer to hl fc layer
        self.s2 = 1.0
        self.m2 = 1.0
        self.gamma2 = nn.Parameter(torch.rand(planes))

        # 1x1 conv
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.gate3 = nn.Linear(planes, planes*4)  # refer to hl fc layer
        self.s3 = 1.0
        self.m3 = 1.0
        self.gamma3 = nn.Parameter(torch.rand(planes*4))

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        self.ratio = _Graph.get_global_var('ratio')
        residual = x

        # bottleneck one
        upsampled = F.avg_pool2d(x, x.shape[2])
        upsampled = upsampled.view(x.shape[0], x.shape[1])  # upsampled = [batch *  input_channel]
        gates1 = self.relu(self.gate1(upsampled))  # gates=[batch * output_channel]  == h_l
        gamma1 = self.gamma1.repeat(x.shape[0], 1)
        if self.ratio < 1:
            inactive_channels = self.conv1.out_channels - round(self.conv1.out_channels * self.ratio)
            inactive_idx = (-gates1).topk(inactive_channels, 1)[1]
            gates1.scatter_(1, inactive_idx, 0)  # set inactive channels as zeros
            gamma1.scatter_(1, inactive_idx, 0)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.s1 * gates1.unsqueeze(2).unsqueeze(3) * out
        out = out + (1.0 - self.s1) * gamma1.unsqueeze(2).unsqueeze(3)*out
        out = self.m1 * out
        out = self.relu(out)

        # bottleneck two
        upsampled = F.avg_pool2d(out, out.shape[2])
        upsampled = upsampled.view(out.shape[0], out.shape[1])
        gates2 = self.relu(self.gate2(upsampled))
        gamma2 = self.gamma2.repeat(out.shape[0], 1)
        if self.ratio < 1:
            inactive_channels = self.conv2.out_channels - round(self.conv2.out_channels * self.ratio)
            inactive_idx = (-gates2).topk(inactive_channels, 1)[1]
            gates2.scatter_(1, inactive_idx, 0)  # set inactive channels as zeros
            gamma2.scatter_(1, inactive_idx, 0)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.s2 * gates2.unsqueeze(2).unsqueeze(3) * out
        out = out + (1.0 - self.s2) * gamma2.unsqueeze(2).unsqueeze(3)*out
        out = self.m2 * out
        out = self.relu(out)

        # bottleneck three
        upsampled = F.avg_pool2d(out, out.shape[2])
        upsampled = upsampled.view(out.shape[0], out.shape[1])
        gates3 = self.relu(self.gate3(upsampled))
        gamma3 = self.gamma3.repeat(out.shape[0], 1)

        if self.ratio < 1:
            inactive_channels = self.conv2.out_channels - round(self.conv2.out_channels * self.ratio)
            inactive_idx = (-gates3).topk(inactive_channels, 1)[1]
            gates3.scatter_(1, inactive_idx, 0)  # set inactive channels as zeros
            gamma3.scatter_(1, inactive_idx, 0)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.s3 * gates3.unsqueeze(2).unsqueeze(3) * out
        out = out + (1.0 - self.s3) * gamma3.unsqueeze(2).unsqueeze(3)*out
        out = self.m3 * out
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AFSNet(nn.Module):

    def __init__(self, block, layers, num_classes=120):
        self.inplanes = 64
        super(AFSNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fclass = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # suggest for adaptive pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fclass(x)

        return x
