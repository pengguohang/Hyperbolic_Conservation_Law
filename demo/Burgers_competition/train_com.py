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
from model.pinn import PINNs

# h控制体大小
# 区域位置
# 

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

    plt.figure(figsize=(8,10))
    plt.imshow(pred, extent=[-1,7,0,10], aspect='auto',  cmap='coolwarm')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Prediction')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'./image/{folder_name}/{model_name}_pred.png')

    plt.figure(figsize=(8,10))
    plt.imshow(np.abs(pred - u), extent=[-1, 7, 0, 10], aspect='auto',  cmap='coolwarm')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Absolute Error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'./image/{folder_name}/{model_name}_error.png')

    # 绘制2 s时刻的折线图
    
    x = np.linspace(-1, 7, pred.shape[1])
    print(x.shape, u.shape)
    u_real = u[-100, :]
    u_pred = pred[-100, :] 

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
    u_real = u[-400, :]
    u_pred = pred[-400, :] 

    plt.figure(figsize=(5, 5))
    plt.plot(x, u_real, label='u_real', color='blue')  # 实际值曲线
    plt.plot(x, u_pred, label='u_pred', color='red', linestyle='--')  # 预测值曲线
    plt.title('8 s')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'./image/{folder_name}/{model_name}_8.png')
def jacobian(y, x, i, j):
    '''
    input: y[], x[]
    '''
    y = y[:,i]
    grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0][:,j]  # y 相对于 x 的第 j 维的偏导数
    return grad.reshape(-1, 1)

def pde_burgers1D(x, model, nu):
    '''
    u_t + 0.5*uu_x = 0

    input: x:[x, xL, xR] [内部点、CV左边界点、CV右边界点]
    '''
    
    x, t, xL, xR = x
    y = model(x, t)
    # print('in pde: ', y.shape, x.shape)
    # dy_x = jacobian(y, x, i=0, j=0)
    dy_t = jacobian(y, t, i=0, j=0)
    yL = model(xL[:, 0:1], t)
    yR = model(xR[:, 0:1], t)
    # dyR_x = jacobian(yR, xR, i=0, j=0)
    # dyL_x = jacobian(yL, xL, i=0, j=0)
    # res = (dy_t + 0.5 * y * dy_x)*0.0002  # 控制体积守恒性质
    res = dy_t*0.0002 + (yR**2/4 - yL**2/4)  
    
    return res.reshape(-1, 1)

def pinn_burgers1D(x_res, t_res, model):
    u = model(x_res, t_res)   # (n, 1)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    return (u_t + 0.5 * u * u_x) ** 2

def pinn_dfvm_region(x, model, c, range_value):
    x_res, t_res, X_inL, X_inR = x
    ind1 = t_res < 4 
    ind21 = (1 + c*t_res/2 - range_value) <= x_res
    ind22 = x_res <= (1 + c*t_res/2 + range_value)
    ind2 = ind21 & ind22
    ind31 = (torch.sqrt(2*c*t_res) - range_value) <= x_res
    ind32 = x_res <= (torch.sqrt(2*c*t_res) + range_value)
    ind3 = ind31 & ind32
    ind = (ind1 & ind2) | (~ind1 & ind3)


    loss_res = torch.mean(torch.where(ind, pde_burgers1D((x_res, t_res, X_inL, X_inR), model, 0.001)**2, 
                                pinn_burgers1D(x_res, t_res, model)))
    
    return loss_res

def pinn_dfvm_all(x, model, c, range_value):
    x_res, t_res, X_inL, X_inR = x
    ind1 = t_res < 4 
    ind21 = (1 + c*t_res/2 - range_value) <= x_res
    ind22 = x_res <= (1 + c*t_res/2 + range_value)
    ind2 = ind21 & ind22
    ind31 = (torch.sqrt(2*c*t_res) - range_value) <= x_res
    ind32 = x_res <= (torch.sqrt(2*c*t_res) + range_value)
    ind3 = ind31 & ind32
    ind = (ind1 & ind2) | (~ind1 & ind3)


    loss_res = torch.mean(torch.where(ind, pde_burgers1D((x_res, t_res, X_inL, X_inR), model, 0.001)**2, 
                                pinn_burgers1D(x_res, t_res, model) + pde_burgers1D((x_res, t_res, X_inL, X_inR), model, 0.001)**2))
    
    return loss_res

def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

def main(args):
    epochs = args.epochs
    folder_name = str(epochs)
    acti = args.activate
    range_value = args.range
    model_name = args.model_name + '_' + str(epochs) + '_' + acti + '_' + str(range_value) + '_256_all'
    label_path='./burgers_c_0.5.mat'
    X_SIZE = 501
    T_SIZE = 501
    c = 0.5

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 1. get data
    # 几何区域：-1<=x<=1, 0<=t<=0.6
    # 均匀网格离散化：101*101
    # res: 所有散点坐标
    res, b_init, b_end, b_left, b_right = get_data([-1, 7], [10, 0], X_SIZE, T_SIZE) 
    res_test, _, _, _, _ = get_data([-1, 7], [10, 0], X_SIZE, T_SIZE) 

    res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
    b_init = torch.tensor(b_init, dtype=torch.float32, requires_grad=True).to(device)  # init
    b_end = torch.tensor(b_end, dtype=torch.float32, requires_grad=True).to(device)
    b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)  # right
    b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)  # left
    # 分别提取散点的x t坐标 --> 列表
    x_res, t_res = res[:,0:1], res[:,1:2]
    x_init, t_init = b_end[:,0:1], b_end[:,1:2]
    x_end, t_end = b_end[:,0:1], b_end[:,1:2]
    x_left, t_left = b_left[:,0:1], b_left[:,1:2]  # 去掉了边界约束
    x_right, t_right = b_right[:,0:1], b_right[:,1:2]
    # 有限体积采样边界点
    DFVM_solver = DFVMsolver(1, device)
    X_inL, X_inR = DFVM_solver.get_vol_data2(res)
    X_inL  = X_inL.requires_grad_(True).to(device)  # 内部点的CV边界采样点
    X_inR  = X_inR.requires_grad_(True).to(device)
    print('x_inL: ', X_inL.shape)
    print('res shape: ', res.shape, 'b_left shape: ', b_left.shape, 'b_right shape: ', b_right.shape, 'b_upper shape: ', b_init.shape, 'b_lower shape: ', b_end.shape)

    # Train PINNs -- MLP
    model = PINNs(in_dim=2, hidden_dim=256, out_dim=1, num_layer=6, acti=acti).to(device)
    model.apply(init_weights)
    # optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.9, verbose=True)

    print(model)
    print(get_n_params(model))  # 总的参数量

    loss_track = []
    model.train()
    step_size = 5000
    min_loss = torch.inf
    saved_path = os.path.join('/home/data2/pengguohang/HCL/Burgers_c_0.5', model_name)

    pde = lambda x, model: pde_burgers1D(x, model, 0.001)
    for i in tqdm(range(int(epochs))):
        optim.zero_grad()
        pred_init = model(x_init, t_init)  # init

        if args.model_name == 'DFVM':
            # print('DFVM')
            loss_res = torch.mean(pde((x_res, t_res, X_inL,X_inR), model)**2)
        elif args.model_name == 'PINNs':
            # print('PINNs')
            loss_res = torch.mean(pinn_burgers1D(x_res, t_res, model))
        elif args.model_name == 'MIX_region':
            # print('DFVM+PINNs')
            loss_res = pinn_dfvm_all((x_res, t_res, X_inL,X_inR), model, c, range_value)
        else:
            raise NotImplementedError
        
        ind1 = x_init < 1
        ind2 = x_init > 0
        ind = ind1 & ind2
        loss_ic = torch.mean(torch.where(ind, (pred_init-1) ** 2, pred_init ** 2))  # 初值：u(x, 0) = 1, x<=0; u(x, 0) = 0, x>0
        loss = loss_res + 100*loss_ic
        loss.backward()  # 反向传播
        optim.step()  # 参数更新

        if (i+1) % step_size == 0:
            scheduler.step()
        with torch.no_grad():
            loss_track.append([loss_res.item(), loss_ic.item()])
        if i % 5000 == 0:
            print('Iter %d, res: %.5e, Lossic: %.5e' % (i, loss_res.item(), loss_ic.item()))

        # save_model
        model_state_dict = model.state_dict()
        torch.save({"epoch": i+1, "loss": min_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optim.state_dict(),
            "history": loss_track
            }, saved_path + "-latest.pt")
        if (i+1) % 100 == 0:
            if loss < min_loss:
                min_loss = loss
                ## save best
                torch.save({"epoch": i + 1, "loss": min_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict()
                    }, saved_path + "-best.pt")

    # test
    res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
    x_test, t_test = res_test[:,0:1], res_test[:,1:2]
    with torch.no_grad():
        checkpoint = torch.load(saved_path + "-best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        pred = model(x_test, t_test)[:,0:1]
        pred = pred.cpu().detach().numpy()
    pred = pred.reshape(X_SIZE, T_SIZE)

    mat = scipy.io.loadmat(label_path)
    u = mat['u'].reshape(X_SIZE, T_SIZE)

    save_fig(pred, u, folder_name, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True, help='30000')
    parser.add_argument('--model_name', type=str, required=True, help='DFVM or PINNs or MIX')
    parser.add_argument('--activate', type=str, required=True, help='Tanh, ReLU, Sigmoid')
    parser.add_argument('--range', type=float, required=True, help='0.1, 0.15, 0.2, for mix')
    args = parser.parse_args()
    main(args)