import numpy as np
import torch.nn as nn
import copy
import torch
import os
import h5py
from torch.utils.data import Dataset
from scipy.stats import qmc
# def get_data(x_range, y_range, x_num, y_num):
#     # 在[0,X]*[0,T]区域采样
#     x = np.linspace(x_range[0], x_range[1], x_num)
#     t = np.linspace(y_range[0], y_range[1], y_num)

#     x_mesh, t_mesh = np.meshgrid(x,t)
#     data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    
#     b_left = data[0,:,:] 
#     b_right = data[-1,:,:]
#     b_upper = data[:,-1,:]
#     b_lower = data[:,0,:]
#     res = data.reshape(-1,2)

#     return res, b_left, b_right, b_upper, b_lower

def get_data(x_range, y_range, x_num, y_num):
    # 在[x_range]*[y_range]区域采样
    x = np.linspace(x_range[0], x_range[1], x_num)  # [x_range0, x_range1]
    t = np.linspace(y_range[0], y_range[1], y_num)  # [y_range0, y_range1]
    
    x_mesh, t_mesh = np.meshgrid(x,t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    
    b_init = data[0,:,:]  # 初始时刻
    b_end = data[-1,:,:]
    b_left = data[:,0,:]
    b_right = data[:,-1,:]
    res = data.reshape(-1,2)

    return res, b_init, b_end, b_left, b_right


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src


def get_clones(module, N):
    # copy.deepcopy(module) 会创建 module 的一个完全独立的副本
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_data_3d(x_range, y_range, t_range, x_num, y_num, t_num):

    step_x = (x_range[1] - x_range[0]) / float(x_num-1)
    step_y = (y_range[1] - y_range[0]) / float(y_num-1)
    step_t = (t_range[1] - t_range[0]) / float(t_num-1)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]

    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    res = data.reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[0]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_left = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[1]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_right = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[0]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_lower = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[1]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_upper = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    return res, b_left, b_right, b_upper, b_lower



class PINNDataset1Dpde(Dataset):
    def __init__(self, filename, root_path='./datasets/', val_batch_idx=-1, vol_size=1, vol_ep=1.e-3):
        """
        :param filename: filename that contains the dataset
        :type filename: STR
        """

        self.vol_size = vol_size   
        self.vol_ep = vol_ep

        # load data file
        data_path = os.path.join(root_path, filename)
        h5_file = h5py.File(data_path, "r")

        # build input data from individual dimensions
        # dim x = [x]
        self.data_grid_x = torch.tensor(h5_file["x-coordinate"], dtype=torch.float)
        self.dx = self.data_grid_x[1] - self.data_grid_x[0]
        self.xL = self.data_grid_x[0] - 0.5 * self.dx
        self.xR = self.data_grid_x[-1] + 0.5 * self.dx
        self.xdim = self.data_grid_x.size(0)
        # # dim t = [t]
        self.data_grid_t = torch.tensor(h5_file["t-coordinate"], dtype=torch.float)

        # main data
        keys = list(h5_file.keys())
        keys.sort()
        print(keys)
        if 'tensor' in keys:
            self.data_output = torch.tensor(np.array(h5_file["tensor"][val_batch_idx]),
                                            dtype=torch.float)
            # permute from [t, x] -> [x, t]
            self.data_output = self.data_output.T

            # for init/boundary conditions
            self.init_data = self.data_output[..., 0, None]
            self.bd_data_L = self.data_output[0, :, None]
            self.bd_data_R = self.data_output[-1, :, None]

        else:
            _data1 = np.array(h5_file["density"][val_batch_idx])
            _data2 = np.array(h5_file["Vx"][val_batch_idx])
            _data3 = np.array(h5_file["pressure"][val_batch_idx])
            _data = np.concatenate([_data1[...,None], _data2[...,None], _data3[...,None]], axis=-1)
            # permute from [t, x] -> [x, t]
            _data = np.transpose(_data, (1, 0, 2))

            self.data_output = torch.tensor(_data, dtype=torch.float)
            del(_data, _data1, _data2, _data3)

            # for init/boundary conditions
            self.init_data = self.data_output[:, 0]
            self.bd_data_L = self.data_output[0]
            self.bd_data_R = self.data_output[-1]

        self.tdim = self.data_output.size(1)
        self.data_grid_t = self.data_grid_t[:self.tdim]

        XX, TT = torch.meshgrid(
            [self.data_grid_x, self.data_grid_t],
            indexing="ij",
        )

        self.data_input = torch.vstack([XX.ravel(), TT.ravel()]).T

        h5_file.close()
        if 'tensor' in keys:
            self.data_output = self.data_output.reshape(-1, 1)
        else:
            self.data_output = self.data_output.reshape(-1, 3)
        

    def get_initial_condition(self):
        # return (self.data_grid_x[:, None], self.init_data)
        return (self.data_input[::self.tdim, :], self.init_data)

    def get_boundary_condition(self):
        # return (self.data_grid_t[:self.nt, None], self.bd_data_L, self.bd_data_R)
        return (self.data_input[:self.tdim, :], self.bd_data_L, self.data_input[-1-self.tdim:-1, :], self.bd_data_R)

    def get_test_data(self, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        n_t = len(self.data_grid_t)

        # start_idx = n_x * n_y * (n_t - n_last_time_steps)
        test_input_x = self.data_input[:, 0].reshape((n_x, n_t))
        test_input_t = self.data_input[:, 1].reshape((n_x, n_t))
        test_output = self.data_output.reshape((n_x, n_t, n_components))

        # extract last n time steps
        test_input_x = test_input_x[:, -n_last_time_steps:]
        test_input_t = test_input_t[:, -n_last_time_steps:]
        test_output = test_output[:, -n_last_time_steps:, :]

        test_input = torch.vstack([test_input_x.ravel(), test_input_t.ravel()]).T

        # stack depending on number of output components
        test_output_stacked = test_output[..., 0].ravel()
        if n_components > 1:
            for i in range(1, n_components):
                test_output_stacked = torch.vstack(
                    [test_output_stacked, test_output[..., i].ravel()]
                )
        else:
            test_output_stacked = test_output_stacked.unsqueeze(1)

        test_output = test_output_stacked

        return test_input, test_output

    def unravel_tensor(self, raveled_tensor, n_last_time_steps, n_components=1):
        n_x = len(self.data_grid_x)
        return raveled_tensor.reshape((1, n_x, n_last_time_steps, n_components))

    def generate_plot_input(self, time=1.0):
        x_space = np.linspace(self.xL, self.xR, self.xdim)
        # xx, yy = np.meshgrid(x_space, y_space)

        tt = np.ones_like(x_space) * time
        val_input = np.vstack((x_space, tt)).T
        return val_input

    def __len__(self):
        return len(self.data_output)

    def __getitem__(self, idx):
        return self.data_input[idx, :], self.data_output[idx]


def sampleCubeQMC(dim, l_bounds, u_bounds, expon=100):
    '''Quasi-Monte Carlo Sampling

    Get the sampling points by quasi-Monte Carlo Sobol sequences in dim-dimensional space. 

    Args:
        dim:      The dimension of space
        l_bounds: The lower boundary
        u_bounds: The upper boundary
        expon:    The number of sample points will be 2^expon

    Returns:
        numpy.array: An array of sample points
    '''
    sampler = qmc.Sobol(d=dim, scramble=False)
    sample = sampler.random_base2(expon)
    data = qmc.scale(sample, l_bounds, u_bounds)
    data = np.array(data)
    return data[1:]

class DFVMsolver():
    '''
    控制体积采样
    '''
    def __init__(self, dim, device) -> None:
        self.vol_size = 1  # 采样数量
        self.vol_ep   = 1.e-4  # CV大小
        self.device = device
        l_bounds = [-1 for _ in range(dim)]  # 采样上下界
        u_bounds = [ 1 for _ in range(dim)]
        self.Int = sampleCubeQMC(dim, l_bounds, u_bounds, self.vol_size) * self.vol_ep  # CV采样 []

    def get_vol_data(self, x):
        # 每个CV内部采样
        # input: 内部离散点(控制体积)  [bs, dim]
        # output: 每个控制体积进行采样，vol_size个  [bs, vol_size, dim]
        x_shape = x.shape
        x = x.cpu().numpy()
        x = np.expand_dims(x, axis=1)
        x = np.broadcast_to(x, (x_shape[0], len(self.Int), x_shape[1]))
        x = x + self.Int  
        return torch.from_numpy(x).to(self.device).requires_grad_(True)
    
    def get_vol_data2(self, x):
        # 每个CV边界采样
        # input: 内部离散点(控制体积)  [bs, dim]
        # output: 每个控制体积进行采样，两边一边一个  [bs, dim]
        bound = torch.tensor([self.vol_ep, 0]).to(self.device)
        xL = x - bound
        xR = x + bound 
        return xL.detach().requires_grad_(True), xR.detach().requires_grad_(True)

    def get_len(self):
        return self.vol_ep*2  # 控制体积大小(直径)
    

    