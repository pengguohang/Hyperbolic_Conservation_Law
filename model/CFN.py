# baseline implementation of PINNs
# paper: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
# link: https://www.sciencedirect.com/science/article/pii/S0021999118307125
# code: https://github.com/maziarraissi/PINNs

import torch
import torch.nn as nn

class CFN(nn.Module):
    def __init__(self, in_dim,  hidden_dim, out_dim, num_layer, delta_x, delta_t, p=1, q=0):
        super(CFN, self).__init__()

        self.p = p
        self.q = q
        self.delta_x = delta_x
        self.delta_t = delta_t

        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        '''
        自回归训练，输入t时刻的解，输出t+1时刻的解

        x: [batch_size, x_dim]
        return: [batch_size, out_dim]
        '''
        # 两层MLP近似通量函数
        F_1 = []
        F_2 = []
        for j in range(self.p, x.shape[1]-self.q):
            window_1 = x[:, j-self.p:j+self.q+1]
            window_2 = x[:, j-1-self.p:j-1+self.q+1]
            u_1 = self.linear(window_1)
            u_2 = self.linear(window_2)
            F_1.append(u_1)
            F_2.append(u_2)

        F = - 1/self.delta_x * (F_1-F_2)
        return F
    
    def train_one_step(self, x, y):
        '''
        自回归训练，输入t时刻的解，输出t+1时刻的解

        第一层添加周期边界
        第二层NN计算通量函数
        第三层时间离散计算下一时刻

        x: [batch_size, x_dim]
        return: [batch_size, out_dim]
        '''
        info = {}

        # 周期边界
        x = torch.cat([x[:, -self.p:], x, x[:, :self.q]], dim=1)

        # 通量计算
        F = self.forward(x)
        
        # TVD-RK3
        x = x[:, self.p:(x.shape[1]-self.q)]
        u1 = x + self.delta_t * F
        u2 = 0.75*x + 0.25*u1 + 0.25 * self.delta_t * self.forward(u1)  # ???????????????????????????
        pred = 1/3*x + 2/3*u2 + 2/3 * self.delta_t * self.forward(u2)

        # loss
        loss = torch.mean((pred - y)**2)

        return loss, pred, info


    