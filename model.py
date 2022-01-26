import torch.nn as nn
import torch
from weights import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LHPF_0(nn.Module):
    def __init__(self):
        super(LHPF_0, self).__init__()
        self.weight = nn.Parameter(data=init, requires_grad=True)
        self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(x, self.weight.cuda(device), padding=1)
        return out

class LHPF_1(nn.Module):
    def __init__(self):
        super(LHPF_1, self).__init__()
        self.weight = nn.Parameter(data=init, requires_grad=True)
        self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(x, self.weight.cuda(device), padding=3,dilation=2)
        return out

class LHPF_2(nn.Module):
    def __init__(self):
        super(LHPF_2, self).__init__()
        self.weight = nn.Parameter(data=init, requires_grad=True)
        self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(x, self.weight.cuda(device), padding=5,dilation=3)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        self.conv9 = nn.Conv2d(9, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, 1)
        self.fc2 = nn.Linear(512, 1)

        self.avgpl = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.f = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)

        self.calsim_up = nn.Conv2d(256, 1, kernel_size=(2, 1), stride=1, bias=True)
        self.calsim_left = nn.Conv2d(256, 1, kernel_size=(1, 2), stride=1, bias=True)
        self.calsim_up_bank = nn.Conv2d(256, 1, kernel_size=(2, 1), stride=1, dilation=2, padding=1, bias=True)
        self.calsim_left_bank = nn.Conv2d(256, 1, kernel_size=(1, 2), stride=1, dilation=2, padding=1, bias=True)
        self.fc_last = nn.Linear(512, 1)
        self.clas = nn.Conv2d(4, 256, kernel_size=(3, 3), stride=(1, 1), bias=True, padding=0)

        self.LHPF_0 = LHPF_0()
        self.LHPF_1 = LHPF_1()
        self.LHPF_2 = LHPF_2()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.LHPF_0(x)
        x2 = self.LHPF_1(x)
        x3 = self.LHPF_2(x)
        x = torch.cat((x1, x2,x3),1)

        x = self.conv9(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_2 = self.layer3(x)
        x = self.layer4(x_2)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_last(x)

        x_2 = self.avgpl(x_2)
        x_2 = self.f(x_2)
        x_2 = nn.LeakyReLU(negative_slope=0.1)(x_2)

        up = self.calsim_up(x_2)
        left = self.calsim_left(x_2)
        up_bank = self.calsim_up_bank(x_2)
        left_bank = self.calsim_left_bank(x_2)

        return up, up_bank, left, left_bank, x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])