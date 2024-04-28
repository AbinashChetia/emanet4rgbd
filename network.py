import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

EM_MOMENTUM = 0.9

import torch
import torch.nn as nn
import torchvision.models as models

class ProjectionBlock(nn.Module):
    '''Projection block for RGB and Depth CNNs'''
    def __init__(self, in_channels, out_channels):
        super(ProjectionBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.global_max_pooling = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_max_pooling(x)
        return x

class RGB_CNN(nn.Module):
    '''Class for RGB's CNN'''
    def __init__(self, num_features):
        super(RGB_CNN, self).__init__()

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   
        resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)    
        self.resnet = nn.Sequential(*list(resnet18.children())[:-2])     

        self.projection_block_rgb = ProjectionBlock(512, 64)  
        skip_connections = [2, 2, 2, 2]    
        layers = [2, 2, 2, 2]  
        self.L = sum(skip_connections) * sum(layers) 

        self.features = nn.ModuleList()
        for i in range(len(layers)):
            for j in range(layers[i]):
                self.features.append(nn.Conv2d(64, num_features, kernel_size=1))

    def forward(self, rgb):
        x_rgb = self.resnet(rgb)
        x_rgb = self.projection_block_rgb(x_rgb)

        features = []
        for feature_layer in self.features:
            x_rgb = feature_layer(x_rgb)
            features.append(x_rgb)
        return features

class Depth_CNN(nn.Module):
    '''Class for Depth's CNN'''
    def __init__(self, num_features):
        super(Depth_CNN, self).__init__()
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-2])
        self.projection_block_depth = ProjectionBlock(512, 64) 
        skip_connections = [2, 2, 2, 2]
        layers = [2, 2, 2, 2]
        self.L = sum(skip_connections) * sum(layers)

        self.features = nn.ModuleList()
        for i in range(len(layers)):
            for j in range(layers[i]):
                self.features.append(nn.Conv2d(64, num_features, kernel_size=1))

    def forward(self, depth):
        depth = depth.unsqueeze(1)
        x_depth = self.resnet(depth)
        x_depth = self.projection_block_depth(x_depth)

        features = []
        for feature_layer in self.features:
            x_depth = feature_layer(x_depth)
            features.append(x_depth)
        return features

class RCFusion(nn.Module):
    '''Class for RCFusion architecture'''

    def __init__(self, num_feats):
        super(RCFusion, self).__init__()

        self.rgb_cnn = RGB_CNN(num_feats)
        self.depth_cnn = Depth_CNN(num_feats)

    def forward(self, rgb, depth):
        rgb_features = self.rgb_cnn(rgb)
        depth_features = self.depth_cnn(depth)

        rgb_features_last_layer = rgb_features[-1:]
        depth_features_last_layer = depth_features[-1:]

        # Concatenate the last layer features
        concatenated_feature = torch.cat((rgb_features_last_layer[0], depth_features_last_layer[0]), dim=1)
        return concatenated_feature

class EMAU(nn.Module):
    '''
    Expectation-Maximization Attention Unit
    '''
    def __init__(self, c, k, n_stages):
        super(EMAU, self).__init__()
        self.n_stages = n_stages
        mu  = torch.Tensor(1, c, k)
        mu.normal_(0, np.sqrt(2. / k)) # Kaiming's Initialization
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            BatchNorm2d(c)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        residue = x
        # 1st 1x1 Conv
        x = self.conv1(x)

        # EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        mu = self.mu.repeat(b, 1, 1)
        with torch.no_grad():
            for i in range(self.n_stages):
                x_t = x.permute(0, 2, 1)
                z = torch.bmm(x_t, mu)
                z = F.softmax(z, dim=2)
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)
                mu = self._l2norm(mu, dim=2)
        z_t = z.permute(0, 2, 1)
        x = mu.matmul(z_t)
        x = x.view(b, c, h, w)
        x = F.relu(x, inplace=True)
        
        # 2nd 1x1 Conv
        x = self.conv2(x)
        x = x + residue
        x = F.relu(x, inplace=True)

        return x, mu
    
    def _l2norm(self, x, dim):
        return x / (1e-6 + x.norm(dim=dim, keepdim=True))
    
class ConvBNReLU(nn.Module):
    '''
    Convolutional Layer with Batch Normalization and ReLU Activation
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class CrossEntropyLoss2d(nn.Module):
    '''
    Cross Entropy Loss for 2D Image Segmentation
    '''
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)
    
class RCFusionEMA(nn.Module):
    '''
    RGB-D Image Segmentation using Expectation-Maximization Attention Network
    '''
    def __init__(self, n_classes, rcf_feats=64, emau_stages=3):
        super(RCFusionEMA, self).__init__()
        self.rcfusion = RCFusion(num_feats=rcf_feats)
        self.fc0 = ConvBNReLU(128, 64, 3, 1, 1, 1)
        self.emau = EMAU(64, 512, emau_stages)
        self.fc1 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, 1),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(256, n_classes, 1)
        self.fc1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.fc2 = nn.Conv2d(64, n_classes, 1)

        self.criterion = CrossEntropyLoss2d(ignore_index=255, reduction='none')

    def forward(self, rgb, depth, lbl=None, size=None):
        x = self.rcfusion(rgb, depth)
        x = self.fc0(x)
        x, mu = self.emau(x)
        x = self.fc1(x)
        x = self.fc2(x)
        if size is None:
            size = rgb.size()[-2:]
        pred = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        if lbl is not None:
            loss = self.criterion(pred, lbl)
            return loss, mu
        else:
            return pred
        
    def train(self, rgb, depth, label, optimizer):
        loss, mu = self.forward(rgb, depth, label, size=rgb.size()[-2:])

        with torch.no_grad():
            mu = mu.mean(dim=0, keepdim=True)
            momentum = EM_MOMENTUM
            self.emau.mu *= momentum
            self.emau.mu += mu * (1 - momentum)

        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def validate(self, rgb, depth, label):
        loss, _ = self.forward(rgb, depth, label, size=rgb.size()[-2:])
        return loss
    
    def predict(self, image):
        return self.forward(image, size=image.size()[-2:])