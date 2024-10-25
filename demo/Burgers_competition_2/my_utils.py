import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm
import scipy.io
import sys
import h5py
import pandas as pd
import argparse
from torch.optim import lr_scheduler

sys.path.append("../../")
from metric import *
from util import *
from model.pinn import PINNs, Resnet
from model.pinnsformer import PINNsformer

# h控制体大小
# 区域位置

def write_res(metrics, filename, tag, append=True):
    # 创建DataFrame
    df = pd.DataFrame()

    # 获取第一个度量值列表的长度来确定维度
    _, values = next(iter(metrics.items()))
    dim = len(values)

    # 遍历metrics字典中的所有度量值
    for metric, values in metrics.items():
        values = [x.item() if hasattr(x, 'item') else x for x in values]  # 处理可能存在的tensor
        df[metric] = values

    # 如果有平均推理时间或其他额外信息，可以在这里添加
    # if "Mean inference time" in metrics:
    #     df["Mean inference time"] = [metrics["Mean inference time"]]*len(df)

    # 写入CSV文件
    if append:
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)

def save_fig(pred, u, folder_name, model_name):

    rl1 = np.sum(np.abs(u-pred)) / np.sum(np.abs(u))
    rl2 = np.sqrt(np.sum((u-pred)**2) / np.sum(u**2))
    L2re = L2RE(torch.tensor(u), torch.tensor(pred)).mean()
    MaxE = MaxError(torch.tensor(u), torch.tensor(pred)).mean()
    Mse = MSE(torch.tensor(u), torch.tensor(pred)).mean()
    Rmse = RMSE(torch.tensor(u), torch.tensor(pred)).mean()
    metrics = {
    'model_name': [model_name],
    'RL1': [rl1],
    'RL2': [rl2],
    'L2RE': [L2re],
    'MaxE': [MaxE],
    'MSE': [Mse],
    'RMSE': [Rmse]}
    write_res(metrics, './output.csv', 'test_tag')
    print(metrics)

    if not os.path.exists(f'./image/{folder_name}'):
        os.makedirs(f'./image/{folder_name}')

    plt.figure(figsize=(5,5))
    plt.imshow(pred, extent=[-1,6,0,7], aspect='auto',  cmap='coolwarm')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Prediction')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'./image/{folder_name}/{model_name}_pred.png')

    plt.figure(figsize=(5,5))
    plt.imshow(np.abs(pred - u), extent=[-1, 6, 0, 7], aspect='auto',  cmap='coolwarm')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Absolute Error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'./image/{folder_name}/{model_name}_error.png')

    # 绘制2 s时刻的折线图
    
    x = np.linspace(-1, 6, pred.shape[1])
    print(x.shape, u.shape)
    u_real = u[-57, :]
    u_pred = pred[-57, :] 

    plt.figure(figsize=(5, 5))
    plt.plot(x, u_real, label='u_real', color='blue')  # 实际值曲线
    plt.plot(x, u_pred, label='u_pred', color='red', linestyle='--')  # 预测值曲线
    plt.title('2 s')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'./image/{folder_name}/{model_name}_2.png')

    # 绘制8 s时刻的折线图
    print(x.shape, u.shape, pred.shape)
    u_real = u[0, :]
    u_pred = pred[0, :]
    plt.figure(figsize=(5, 5))
    plt.plot(x, u_real, label='u_real', color='blue')  # 实际值曲线
    plt.plot(x, u_pred, label='u_pred', color='red', linestyle='--')  # 预测值曲线
    plt.title('7 s')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'./image/{folder_name}/{model_name}_7.png')
def jacobian(y, x, i, j):
    '''
    input: y[], x[]
    '''
    # print('in jacobian: ', y.shape, x.shape)
    y = y[:,i]
    grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0][:,j]  # y 相对于 x 的第 j 维的偏导数
    return grad.reshape(-1, 1)

def pde_burgers1D(x, model, nu):
    '''
    u_t + 0.5*uu_x = 0

    input: x:[x, xL, xR] [内部点、CV左边界点、CV右边界点]
    '''
    x, t, x_inL, t_inL, x_inR, t_inR, = x
    y = model(x, t)
    yL = model(x_inL, t_inL)
    yR = model(x_inR, t_inR)
    dy_t = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

    res = dy_t*0.0002 + (yR**2/4 - yL**2/4)
    
    return res.reshape(-1, 1)

def compute_entropy(x, model):
    """
    (u^2)_t + (2/3u^3)_x <= 0

    """
    x, t, x_inL, t_inL, x_inR, t_inR, = x
    y = model(x, t)
    yL = model(x_inL, t_inL)
    yR = model(x_inR, t_inR)
    dy_t = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

    entropy = 2*y*dy_t*0.0002 + 2/3*(yR**3 - yL**3)

    return entropy.reshape(-1, 1)

def pinn_burgers1D(x_res, t_res, model):
    u = model(x_res, t_res)   # (n, 1)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    return ((u_t + 0.5 * u * u_x) ** 2).reshape(-1, 1)

def pinn_dfvm_region(x, model, c, range_value):

    x_res, t_res, x_inL, t_inL, x_inR, t_inR, = x
    ind1 = t_res < 4 
    ind21 = (1 + c*t_res/2 - range_value) <= x_res
    ind22 = x_res <= (1 + c*t_res/2 + range_value)
    ind2 = ind21 & ind22
    ind31 = (torch.sqrt(2*c*t_res) - range_value) <= x_res
    ind32 = x_res <= (torch.sqrt(2*c*t_res) + range_value)
    ind3 = ind31 & ind32
    ind = (ind1 & ind2) | (~ind1 & ind3)

    dfvm_loss = pde_burgers1D((x_res, t_res, x_inL, t_inL, x_inR, t_inR,), model, 0.001)**2
    pinn_loss = pinn_burgers1D(x_res, t_res, model)

    loss_res = torch.mean(torch.where(ind, dfvm_loss, pinn_loss))
    
    return loss_res

def pinn_dfvm_all(x, model, c, range_value):
    x_res, t_res, x_inL, t_inL, x_inR, t_inR, = x
    ind1 = t_res < 4 
    ind21 = (1 + c*t_res/2 - range_value) <= x_res
    ind22 = x_res <= (1 + c*t_res/2 + range_value)
    ind2 = ind21 & ind22
    ind31 = (torch.sqrt(2*c*t_res) - range_value) <= x_res
    ind32 = x_res <= (torch.sqrt(2*c*t_res) + range_value)
    ind3 = ind31 & ind32
    ind = (ind1 & ind2) | (~ind1 & ind3)

    dfvm_loss = pde_burgers1D((x_res, t_res, x_inL, t_inL, x_inR, t_inR,), model, 0.001)**2
    pinn_loss = pinn_burgers1D(x_res, t_res, model)

    loss_res = torch.mean(torch.where(ind.reshape(-1, 1), dfvm_loss, dfvm_loss + pinn_loss))
    
    return loss_res

def pinn_dfvm_ga(x, model):
    x_res, t_res, x_inL, t_inL, x_inR, t_inR, = x
    
    u = model(x_res, t_res)   # (n, 1)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    pinn_loss = ((u_t + 0.5 * u * u_x) ** 2).reshape(-1, 1)

    alpha = 1 # 1.5
    beta = 1.25  # 2
    Lambda = ( 1 / (1 +  alpha * torch.abs(u_x)**beta ) ) # Works fine

    dfvm_loss = pde_burgers1D((x_res, t_res, x_inL, t_inL, x_inR, t_inR,), model, 0.001)**2
    loss_res = torch.mean(dfvm_loss + Lambda * pinn_loss)
    
    return loss_res

def compute_loss(model, pde, x_init, t_init, x_res, t_res, x_inL, t_inL, x_inR, t_inR, c, range_value, args):
    pred_init = model(x_init, t_init)  # init
        
    if args.model_name == 'DFVM':
        # print('DFVM')
        loss_res = torch.mean(pde((x_res, t_res, x_inL, t_inL, x_inR, t_inR,), model)**2)
    elif args.model_name == 'PINNs':
        # print('PINNs')
        # print(x_res.shape, t_res.shape)
        loss_res = torch.mean(pinn_burgers1D(x_res, t_res, model))
    elif args.model_name == 'MIX_all':
        # print('DFVM+PINNs')
        loss_res = pinn_dfvm_all((x_res, t_res, x_inL, t_inL, x_inR, t_inR,), model, c, range_value)
    elif args.model_name == 'MIX_region':
        # print('DFVM+PINNs')
        loss_res = pinn_dfvm_region((x_res, t_res, x_inL, t_inL, x_inR, t_inR,), model, c, range_value)
    elif args.model_name == 'GA':
        loss_res = pinn_dfvm_ga((x_res, t_res, x_inL, t_inL, x_inR, t_inR,), model)
    else:
        raise NotImplementedError
    
    if args.entropy:
        loss_res += torch.mean(compute_entropy((x_res, t_res, x_inL, t_inL, x_inR, t_inR,), model))
    
    ind1 = x_init < 1
    ind2 = x_init > 0
    ind = ind1 & ind2
    loss_ic = torch.mean(torch.where(ind, (pred_init-1) ** 2, pred_init ** 2))  # 初值：u(x, 0) = 1, x<=0; u(x, 0) = 0, x>0
        
    return loss_res, loss_ic

def get_model(device, args):
    acti = args.activate

    if args.model == 'MLP':
        model = PINNs(in_dim=2, hidden_dim=args.param, out_dim=1, num_layer=args.layer, acti=acti).to(device)
    elif args.model == 'ResNet':
        model = Resnet(in_dim=2, hidden_dim=args.param, out_dim=1, num_layer=args.layer, acti=acti).to(device)
    else:
        model = PINNsformer(d_out=1, d_hidden=128, d_model=32, N=1, heads=2).to(device)

    model.apply(init_weights)

    return model

def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

