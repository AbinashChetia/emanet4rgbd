import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

class ResNet(nn.Module):
    '''Class used to create a ResNet-18 model'''

    def __init__(self, num_classes, mode):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self._create_model()
        
    def _create_model(self):       
        # Initial conv & pool
        if self.mode == 'rgb':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        elif self.mode == 'd':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks (x4)
        self.res2 = self._residual_block(64, 64, 1)
        self.res3 = self._residual_block(64, 128, 2)
        self.res4 = self._residual_block(128, 256, 2)
        self.res5 = self._residual_block(256, 512, 2)
        
        # Final pool & classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)
        
    def _residual_block(self, in_channels, out_channels, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = self.res2(x) + x
        x = self.res3(x) + x
        x = self.res4(x) + x
        x = self.res5(x) + x
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

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
    
class ResNetEMA(nn.Module):
    '''
    RGB-D Image Segmentation using Expectation-Maximization Attention Network
    '''
    def __init__(self, n_classes, emau_stages):
        super(ResNetEMA, self).__init__()
        self.resnet_rgb = ResNet(n_classes, mode='rgb')
        self.resnet_d = ResNet(n_classes, mode='d')
        self.fc0 = ConvBNReLU(2048, 512, 3, 1, 1, 1)
        self.emau = EMAU(512, 64, emau_stages)
        self.fc1 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, 1),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(256, n_classes, 1)

        self.criterion = CrossEntropyLoss2d(ignore_index=255, reduction='none')

    def forward(self, img, lbl=None, size=None):
        x_rgb = self.resnet_rgb(img['rgb'])
        x_d = self.resnet_d(img['d'])
        x = torch.cat([x_rgb, x_d], dim=1)
        x = self.fc0(x)
        x, mu = self.emau(x)
        x = self.fc1(x)
        x = self.fc2(x)
        if size is None:
            size = img.size()[-2:]
        pred = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        if lbl is not None:
            loss = self.criterion(pred, lbl)
            return loss, mu
        else:
            return pred