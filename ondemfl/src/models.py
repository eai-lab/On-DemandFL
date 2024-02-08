import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


"""task models"""
class fashionmnist_cnn(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, num_hiddens=512, num_classes=10):
        super(fashionmnist_cnn, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x    
    
class emnist_cnn(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, num_hiddens=2048, num_classes=62):
        super(emnist_cnn, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x    


class femnist_cnn(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_hiddens=2048, num_classes=62):
        super(femnist_cnn, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x    


class mnist_fc(nn.Module):
    def __init__(self, num_classes=10):
        super(mnist_fc, self).__init__()
        self.activation = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=784, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        return x

class svhn_cnn(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_hiddens=512, num_classes=10):
        super(svhn_cnn, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x    

# class fashionmnist_cnn(nn.Module):
#     def __init__(self, in_channels=1, hidden_channels=32, num_hiddens=512, num_classes=10):
#         super(fashionmnist_cnn, self).__init__()
#         self.activation = nn.ReLU(True)

#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
#         self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
#         self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

#     def forward(self, x):
#         x = self.activation(self.conv1(x))
#         x = self.maxpool1(x)

#         x = self.activation(self.conv2(x))
#         x = self.maxpool2(x)
#         x = self.flatten(x)
    
#         x = self.activation(self.fc1(x))
#         x = self.fc2(x)
        
#         return x    


class cifar10_cnn(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_hiddens=512, num_classes=10):
        super(cifar10_cnn, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x    


class stl10_cnn(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_hiddens=512, num_classes=10):
        super(stl10_cnn, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x    


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class cifar100_cnn(nn.Module):
    '''
    https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/mobilenetv2.py

    mobilenetv2 in pytorch
    [1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
        MobileNetV2: Inverted Residuals and Linear Bottlenecks
        https://arxiv.org/abs/1801.04381
    '''
    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x


    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)



class tinyimgnet_cnn(nn.Module):
    pass


"""distribution models"""
class distribution_fcn(nn.Module):
    def __init__(self, input, output):
        super(distribution_fcn, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(input),
            nn.Linear(input, 256, bias=True),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Linear(256, output, bias=True)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = nn.functional.softmax(logits, dim=1)
        return logits

class distribution_fcn2(nn.Module):
    def __init__(self, input, output):
        super(distribution_fcn2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(input),
            nn.Linear(input, 256, bias=True),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),

            nn.BatchNorm1d(256),
            nn.Linear(256, output, bias=True)

        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = nn.functional.softmax(logits, dim=1)
        return logits

class distribution_fcn3(nn.Module):
    def __init__(self, input, output):
        super(distribution_fcn3, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, output, bias=True)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = nn.functional.softmax(logits, dim=1)
        return logits

class distribution_fcn4(nn.Module):
    def __init__(self, input, output):
        super(distribution_fcn4, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, output, bias=True)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = nn.functional.softmax(logits, dim=1)
        return logits


class distribution_fcn5(nn.Module):
    def __init__(self, input, output):
        super(distribution_fcn5, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(input),
            nn.Linear(input, 512, bias=True),
            nn.ReLU(),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, output, bias=True)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = nn.functional.softmax(logits, dim=1)
        return logits

