import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import inspect

# regularizer

class DifferentialEntropyRegularization(torch.nn.Module):

    def __init__(self, eps=1e-8):
        super(DifferentialEntropyRegularization, self).__init__()
        self.eps = eps
        self.pdist = torch.nn.PairwiseDistance(2)

    def forward(self, x):

        with torch.no_grad():
            dots = torch.mm(x, x.t())
            n = x.shape[0]
            dots.view(-1)[::(n + 1)].fill_(-1)  # trick to fill diagonal with -1
            _, I = torch.max(dots, 1)  # max inner prod -> min distance

        rho = self.pdist(x, x[I])

        # dist_matrix = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), p=2, dim=-1)
        # rho = dist_matrix.topk(k=2, largest=False)[0][:, 1]

        loss = -torch.log(rho + self.eps).mean()

        return loss

# ***** *****

def FixedImageStandard(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

# ***** *****

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
# ***** *****

class IdentityLayer(nn.Module):
    
    def __init__(self):
        super(IdentityLayer, self).__init__()
        
    def forward(self, x):
        return x
    
# ***** *****

# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/11  

import numbers
import numpy as np
from PIL import ImageFilter

class GaussianSmoothing(object):
    
    def __init__(self, radius):
        
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        
        elif isinstance(radius, list):
            
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))

    
# ***** *****

# URL : https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py
# URL : https://github.com/clovaai/tunit/blob/master/tools/ops.py#L69

import sys
import torch

# ***

def estimate_mi( x_out, x_tf_out, lamb = 1.0, eps = sys.float_info.epsilon ):

    # ***
    
    p_i_j = estimate_joint(x_out, x_tf_out)
    
    _, k = x_out.size()
    assert (p_i_j.size() == (k, k))

    # *** 
    
    # p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    # p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric
    
    p_i = p_i_j.sum(dim=1)
    p_j = p_i_j.sum(dim=0)
    
    # Avoid NaN losses.     
    p_i_j = torch.clamp(p_i_j, min=eps)
    p_j = torch.clamp(p_j, min=eps)
    p_i = torch.clamp(p_i, min=eps)
    
    # *** 
    
    e_pi = ( p_i * torch.log(p_i) ).sum() * -1
    e_pj = ( p_j * torch.log(p_j) ).sum() * -1
    
    # *** 
    
    p_i = p_i.view(k, 1).expand(k, k)
    p_j = p_j.view(1, k).expand(k, k) 
    
    # print(loss.size()) : C x C
    loss = p_i_j * ( torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i) ) 
    loss = loss.sum() 
    
    '''
    loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                            - torch.log(p_j) \
                            - torch.log(p_i))

    loss_no_lamb = loss_no_lamb.sum()
    
    '''
    
    # ***
    
    loss = ( 2 * loss ) / (e_pi + e_pj)

    # *** 
    
    del p_i_j, p_i, p_j, e_pi, e_pj
    
    return loss # , loss_no_lamb

# ***

# produces variable that requires grad (since args require grad)
def estimate_joint(x_out, x_tf_out):
  
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)                            # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.                    # symmetrise
    p_i_j = p_i_j / p_i_j.sum()                         # normalise

    return p_i_j

# ***** *****

def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]