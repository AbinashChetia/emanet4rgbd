import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as Func
from torch.nn.modules.batchnorm import BatchNorm2d

class EMAU(nn.Module):
    '''
    Expectation-Maximization Attention Unit
    '''
    def __init__(self, c, k, n_stages):
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
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)
    
class ResNetEMA(nn.Module):
    '''
    RGB-D Image Segmentation using Expectation-Maximization Attention Network
    '''
    def __init__(self, n_classes, n_layers, emau_stages):
        self.fc0 = ConvBNReLU(2048, 512, 3, 1, 1, 1)
        self.emau = EMAU(512, 64, emau_stages)
        self.fc1 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, 1),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(256, n_classes, 1)

        self.criterion = CrossEntropyLoss2d(ignore_index=255, reduction='none')

    def forward(self, img, lbl=None, size=None):
        x = self.fc0(img)
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