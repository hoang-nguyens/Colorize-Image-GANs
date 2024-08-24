import torch
from torch import nn
from torch import optim
import numpy as np


class UnetRecursion(nn.Module):
    def __init__(self, n_features, n_inputs, submodule = None, input_channels = None, dropout = False,
                 innermost = False, outermost = False):
        super().__init__()
        if input_channels is None:
            input_channels = n_features
        self.outermost = outermost # use to make outermost layers don't have skip connection
        downconv = nn.Conv2d(input_channels, n_inputs, kernel_size = 4,
                             stride = 2, padding = 1, bias = False)
        downact = nn.LeakyReLU(0.2, inplace = True)
        downnorm = nn.BatchNorm2d(n_inputs)

        upact = nn.ReLU(inplace = True)
        upnorm = nn.BatchNorm2d(n_features)

        if innermost:
            upconv = nn.ConvTranspose2d(n_inputs, n_features, kernel_size=4,
                                        stride = 2, padding = 1, bias = False)
            down = [downact, downconv]
            up = [upact, upconv, upnorm]
            model = down + up
        elif outermost:
            upconv = nn.ConvTranspose2d(n_inputs * 2, n_features, kernel_size=4,
                                        stride = 2, padding = 1, bias = False)
            down = [downconv]
            up = [upact, upconv, nn.Tanh()]
            model = down + [submodule] + up

        else:
            upconv = nn.ConvTranspose2d(n_inputs * 2, n_features, kernel_size=4,
                                        stride = 2, padding = 1, bias = False)
            down = [downact, downconv, downnorm]
            up = [upact, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
class Unet(nn.Module):
    def __init__(self, input = 1, output = 2, n_filters = 512, n_down = 8):
        super().__init__()

        unet = UnetRecursion(n_features=n_filters, n_inputs=n_filters, innermost = True)

        for _ in range(n_down - 5):
            unet = UnetRecursion(n_filters, n_filters,submodule= unet, dropout= True)
        n_features = n_filters
        for _ in range(3):
            unet = UnetRecursion(n_features=n_features // 2,n_inputs= n_features,submodule= unet)
            n_features //= 2
        unet = UnetRecursion(n_features=output, n_inputs=n_features, submodule=unet,input_channels= input, outermost= True)

        self.model = unet
    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input = 3, output = 1, n_filters  = 64, n_down = 3):
        super().__init__()
        model = [self.layers(input, n_filters, norm = False )]

        for i in range(n_down):
            model += [self.layers(n_filters * 2**i, n_filters * 2**(i+1), stride = 1 if i == (n_down -1) else 2)]

        model += [self.layers(n_filters * 2**n_down, 1 , stride=1, norm = False, act = False )]

        self.model = nn.Sequential(*model)
    def layers(self, in_channels, out_channels, kernel = 4, stride = 2, padding = 1,act = True, norm = True):
        conv = [nn.Conv2d(in_channels, out_channels, kernel_size= kernel, stride = stride, padding = padding)]
        if norm:
            conv += [nn.BatchNorm2d(out_channels)]
        if act:
            conv += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*conv)

    def forward(self, x):
        return self.model(x)

def init_weight(model, init_mode = 'norm', gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__()
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init_mode == 'norm':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_mode == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain)
            elif init_mode == 'kaiming':
                nn.init.kaiming_normal_((m.weight.data, 0.0))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 0.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)
    return model

def init_model(model, device):
    model = model.to(device)
    model = init_weight(model)
    return model


