import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Function

class st_var(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input<-1] = 0
        grad_input[input>1] = 0
        return grad_input

def st_var_hash_layer(input):
    return st_var.apply(input)

class LinearBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, bias=True):
        super(LinearBnRelu, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class RBE_NORM(nn.Module):
    def __init__(self,
                in_planes,
                out_planes,
                num_layers=1,
                hidden_dim=0,
                bias=True,
                binary_func='st_var',
                annealing_temp=1.,
                transform_blocks=1,
                proxy_loss=False):
        super(RBE_NORM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.annealing_temp = annealing_temp
        self.proxy_loss = proxy_loss
        self.transform_blocks = transform_blocks
        self.W = self._make_transformation(in_planes, out_planes, num_blocks=self.transform_blocks)

        for i in range(num_layers):
            setattr(self, 'B{}'.format(i), self._make_transformation(out_planes, in_planes, num_blocks=self.transform_blocks))
            setattr(self, 'R{}'.format(i), self._make_transformation(in_planes, out_planes, num_blocks=self.transform_blocks))

        if binary_func == 'st_var':
            self.binary_func = st_var_hash_layer
        elif binary_func == 'identity':
            self.binary_func = torch.nn.Identity()
        else:
            raise NotImplementedError("Unknown binary func.")

        self.apply(self._init_params)

    def _init_params(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    
    def _make_layer(self, in_planes, out_planes, blocks):
        layers = []
        layers.append(LinearBnRelu(in_planes, self.hidden_dim))

        for i in range(1, blocks):
            layers.append(LinearBnRelu(self.hidden_dim, self.hidden_dim))
        
        layers.append(nn.Linear(self.hidden_dim, out_planes, bias=self.bias))

        return nn.Sequential(*layers)

    def _make_transformation(self, in_planes, out_planes, num_blocks=1):
        if self.hidden_dim == 0:
            transform = nn.Linear(in_planes, out_planes, bias=self.bias)
        elif num_blocks == 1:
            transform = nn.Sequential(
                nn.Linear(in_planes, self.hidden_dim, bias=self.bias),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, out_planes, bias=self.bias),
            )
        else:
            transform = self._make_layer(in_planes, out_planes, num_blocks)
        return transform

    def forward(self, input):
        b = self.binary_func(self.W(input))
        for i in range(self.num_layers):
            f = F.normalize(getattr(self, 'B{}'.format(i))(b), p=2, dim=1)
            d = getattr(self, 'R{}'.format(i))(input - f)
            d = self.binary_func(d)
            b = b + ((1/2)**(i+1))*d
        return b

