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

sys.path.append("./")
from my_utils import *

# h控制体大小
# 区域位置

X_SIZE = 201
T_SIZE = 201
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def sample_data(args):
    res, b_init, b_end, b_left, b_right = get_data([-1, 6], [7, 0], X_SIZE, T_SIZE) 
    res_test, _, _, _, _ = get_data([-1, 6], [7, 0], X_SIZE, T_SIZE) 
    if args.sample == 'QMC':
        res = sampleCubeQMC(2, [-1, 0], [6, 7], 15)  # 2^n - 1个点  

    res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
    b_init = torch.tensor(b_init, dtype=torch.float32, requires_grad=True).to(device)  # init
    b_end = torch.tensor(b_end, dtype=torch.float32, requires_grad=True).to(device)
    b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)  # right
    b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)  # left

    # pinnformer的数据需扩展为伪序列
    if args.model == 'PINNsformer':
        step_size = 1e-4
        # 有限体积采样边界点
        DFVM_solver = DFVMsolver(1, device)
        # print('RES: ', res.shape)
        X_inL, X_inR = DFVM_solver.get_vol_data2(res)
        # print('x_inL: ', X_inL.shape)
        X_inL = make_time_sequence(X_inL, num_step=5, step=step_size) # 内部点的CV边界采样点
        X_inR = make_time_sequence(X_inR, num_step=5, step=step_size)
        X_inL = torch.tensor(X_inL, dtype=torch.float32, requires_grad=True).to(device)
        X_inR = torch.tensor(X_inR, dtype=torch.float32, requires_grad=True).to(device)  # init
        x_inL, t_inL = X_inL[:,:,0:1], X_inL[:,:,1:2]
        x_inR, t_inR = X_inR[:,:,0:1], X_inR[:,:,1:2]

        res = make_time_sequence(res, num_step=5, step=step_size)
        b_init = make_time_sequence(b_init, num_step=5, step=step_size)
        res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
        b_init = torch.tensor(b_init, dtype=torch.float32, requires_grad=True).to(device)  # init
        x_res, t_res = res[:,:,0:1], res[:,:,1:2]
        x_init, t_init = b_init[:,:,0:1], b_init[:,:,1:2]
    else:
        # 有限体积采样边界点
        DFVM_solver = DFVMsolver(1, device)
        X_inL, X_inR = DFVM_solver.get_vol_data2(res)
        X_inL = X_inL.requires_grad_(True).to(device)  # 内部点的CV边界采样点
        X_inR = X_inR.requires_grad_(True).to(device)
        x_inL, t_inL = X_inL[:,0:1], X_inL[:,1:2]
        x_inR, t_inR = X_inR[:,0:1], X_inR[:,1:2]

        x_res, t_res = res[:,0:1], res[:,1:2]
        x_init, t_init = b_end[:,0:1], b_end[:,1:2]
        x_end, t_end = b_end[:,0:1], b_end[:,1:2]
        x_left, t_left = b_left[:,0:1], b_left[:,1:2]  # 没用到边界点
        x_right, t_right = b_right[:,0:1], b_right[:,1:2]

    
    print('x_inL: ', X_inL.shape)
    print('res shape: ', res.shape, 'b_left shape: ', b_left.shape, 'b_right shape: ', b_right.shape, 'b_upper shape: ', b_init.shape, 'b_lower shape: ', b_end.shape)

    return x_res, t_res, x_init, t_init,  x_inL, t_inL, x_inR, t_inR, res_test

def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = src.unsqueeze(1).repeat(1, dim, 1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src

def main(args):
    epochs = args.epochs
    acti = args.activate
    range_value = args.range
    # image path:  image_folder + model_name(模型名称_epochs_激活函数_ragne_模型大小_modeltype_losstype_采样_优化器)
    image_folder = str(epochs) 
    # saved model path: saved_path + model_name
    model_name = args.model_name + '_' + acti + '_' + str(range_value) + '_' + str(args.layer) + '_' + str(args.param) + '_' + args.model + '_' + args.sample
    saved_path = os.path.join('/home/data2/pengguohang/HCL/Burgers_c_0.5', model_name)
    
    # label path
    label_path='./burgers_c_0.5.mat'
    c = 0.5

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 1. get data
    # 几何区域：-1<=x<=1, 0<=t<=0.6
    # 均匀网格离散化：101*101
    # res: 所有散点坐标

    x_res, t_res, x_init, t_init,  x_inL, t_inL, x_inR, t_inR, res_test = sample_data(args)
    x_val = torch.rand(1000) 
    print('data: ', x_res.shape, t_res.shape, x_init.shape, t_init.shape, x_inL.shape, t_inR.shape)
    # 分别提取散点的x t坐标 --> 列表
    
    # get model
    model = get_model(device, args)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.9)
    print(model)
    print(get_n_params(model))  # 总的参数量

    start_epoch = 0
    loss_track = []
    model.train()
    step_size = 5000
    min_val = torch.inf
    pde = lambda x, model: pde_burgers1D(x, model, 0.001)

    # continute training
    if args.continue_training:
        checkpoint = torch.load(saved_path + "-latest.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading latest checkpoint from ", saved_path + "-latest.pt")
        optimizer = getattr(torch.optim, 'Adam')(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint['epoch']
        min_val = checkpoint['loss']
        loss_track  = checkpoint["history"]

    # test data
    res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
    x_test, t_test = res_test[:,0:1], res_test[:,1:2]
    mat = scipy.io.loadmat(label_path)
    u = mat['u'].reshape(X_SIZE, T_SIZE)

    # start training
    print('start training--------------------------------------------------------------')
    for i in tqdm(range(start_epoch, int(start_epoch + epochs))):
        optim.zero_grad()
        loss_res, loss_ic = compute_loss(model, pde, x_init, t_init, x_res, t_res, x_inL, t_inL, x_inR, t_inR, c, range_value, args)
        loss = loss_res + 100*loss_ic
        loss.backward()  # 反向传播
        optim.step()  # 参数更新

        if (i+1) % step_size == 0:
            scheduler.step()
        with torch.no_grad():
            loss_track.append([loss_res.item(), loss_ic.item()])
        if i % 5000 == 0:
            learning_rate = scheduler.get_last_lr()[0]
            print('Iter %d, res: %.5e, Lossic: %.5e, lr: %.5e' % (i, loss_res.item(), loss_ic.item(), learning_rate))

        if (i+1) % 100 == 0:
            # val
            with torch.no_grad():
                pred = model(x_test, t_test).reshape(X_SIZE, T_SIZE)
                pred = pred.cpu().detach().numpy()
            val = np.mean((pred - u) ** 2)
            if val < min_val:
                min_val = val
                ## save best
                torch.save({"epoch": i + 1, "loss": min_val,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict()
                    }, saved_path + "-best.pt")

    # save latest model
    model_state_dict = model.state_dict()
    torch.save({"epoch": i+1, "loss": min_val,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optim.state_dict(),
        "history": loss_track
        }, saved_path + "-latest.pt")
    
    with torch.no_grad():
        checkpoint = torch.load(saved_path + "-best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        pred = model(x_test, t_test)[:,0:1]
        pred = pred.cpu().detach().numpy()
    pred = pred.reshape(X_SIZE, T_SIZE)

    save_fig(pred, u, image_folder, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='30000')
    parser.add_argument('--model_name', type=str,  default='MIX_all', help='DFVM or PINNs or MIX_all or MIX_region or GA')
    parser.add_argument('--activate', type=str,  default='Tanh', help='Tanh, ReLU, Sigmoid')
    parser.add_argument('--range', type=float,  default=0.1, help='0.1, 0.15, 0.2, for mix')
    parser.add_argument('--param', type=int,  default=128, help='128, 256, 512')
    parser.add_argument('--layer', type=int,  default=4, help='4, 6')
    parser.add_argument('--model', type=str,  default='MLP', help='MLP, ResNet')
    parser.add_argument('--sample', type=str,  default='QMC', help='QMC')
    parser.add_argument("--continue_training", action='store_true', help='continue training')
    args = parser.parse_args()
    print(args)
    main(args)