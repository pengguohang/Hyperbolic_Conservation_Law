# baseline implementation of PINNs
# paper: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
# link: https://www.sciencedirect.com/science/article/pii/S0021999118307125
# code: https://github.com/maziarraissi/PINNs

import torch
import torch.nn as nn
from collections import OrderedDict

def get_activation_function(name):
    """
    通过字符串名称获取激活函数
    :param name: 激活函数的名称，例如 'Tanh', 'ReLU', 'Sigmoid' 等
    :return: 对应的激活函数类
    """
    try:
        activation_class = getattr(nn, name)
        return activation_class()
    except AttributeError:
        raise ValueError(f"激活函数 '{name}' 不存在于 torch.nn 模块中")


class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, acti):
        super(PINNs, self).__init__()

        layers = []
        self.acti = get_activation_function(acti)
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(self.acti)
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(self.acti)

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x, t):
        src = torch.cat((x,t), dim=-1)
        return self.linear(src)
    
class Resnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, acti):
        super(Resnet, self).__init__()

        layers = []
        self.acti = get_activation_function(acti)
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(self.acti)
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(self.acti)

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x, t):
        src = torch.cat((x,t), dim=-1)
        # print('src: ', src.shape)  # 40401, 2
        x = torch.cat((x, t), dim=-1)
        # print('x: ', x.shape)
        src = src + x
        return self.linear(src)

    
