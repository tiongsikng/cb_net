import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import math
from scipy.linalg import hadamard  # direct import hadamard matrix from scipy
import numpy as np


# *** *** *** *** ***

class PolyLoss(nn.Module):
    def __init__(self, epsilon = 2):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.softmax_ = torch.nn.Softmax(dim=-1)
    
    def forward(self, ce_loss, preds, labels):
        softmax_preds = self.softmax_(preds)
        pt_face = torch.sum((F.one_hot(labels, num_classes=1054) * softmax_preds), dim = 1)
        poly_loss = ce_loss + (self.epsilon * (1 - pt_face))

        return poly_loss

    def forward(self, ce_loss, preds, labels):
        softmax_preds = self.softmax_(preds)
        pt_face = torch.sum((F.one_hot(labels, num_classes=1054) * softmax_preds), dim = 1)
        poly_loss = ce_loss + (self.epsilon * (1 - pt_face))

        return poly_loss
    
class PolyLossLS(nn.Module):
    def __init__(self, epsilon = 2, alpha=0.1):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.softmax_ = torch.nn.Softmax(dim=-1)
    
    def forward(self, ce_loss, preds, labels):
        softmax_preds = self.softmax_(preds)
        pt_face = torch.sum((F.one_hot(labels, num_classes=1054) * softmax_preds), dim = 1)
        poly_loss = ce_loss + (self.epsilon * (1 - pt_face))

        return poly_loss

# *** *** *** *** ***

# CurricularFace
# https://github.com/JDAI-CV/FaceX-Zoo/blob/fd517dc5bb53638af086676a8d51347e88cb9060/head/CurricularFace.py

class CurricularFace(nn.Module):
    
    """Implementation for "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition".
    """
    
    # in_features : fea_dim
    # out_features : num_class
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        # nn.init.normal_(self.kernel, std=0.01)
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.kernel, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, feats.size(0)), labels].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2) + 1e-10)
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output

    
# *** *** *** *** ***
# Center Loss

class CenterLoss(nn.Module):
    
    def __init__(self, in_features, out_features):
        
        super(CenterLoss, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.centers = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def forward(self, input, label = None):

        cosine = self.cosine_sim(input, self.centers)
        print(cosine)
        print(cosine.size())
        # *** ***
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        print(one_hot.size())
        print(torch.sum(one_hot, dim = 1))
        # print(one_hot[:, label])
        # print(cosine[:, label])
        # *** ***
        loss = cosine * one_hot

        loss = loss.clamp(min=1e-12, max=1e+12).sum() / input.size(0)

        return loss

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.centers)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', num_classes = ' + str(self.out_features) + ')'
    
    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)
        

# ***** CrossEntropy *****

class CrossEntropy(nn.Module):
    
    def __init__(self, in_features, out_features):
        
        super(CrossEntropy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
   
    def forward(self, x):
        
        x = F.linear(F.normalize(x, p=2, dim=1), self.weight)
        # x = F.linear(F.normalize(x, p=2, dim=1), \
        #             F.normalize(self.weight, p=2, dim=1))

        return x
       
    def __repr__(self):

        return self.__class__.__name__ + '(' \
           + 'in_features = ' + str(self.in_features) \
           + ', out_features = ' + str(self.out_features) + ')'
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

        
# ***** KLDivergence *****


class KLDivergence(nn.Module):
    
    def __init__(self, in_features, out_features, t = 20, bias = True):
        
        super(KLDivergence, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.t = t
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def forward(self, hr_pred, lr_pred):
        
        # output = F.kl_div(torch.log_softmax((F.linear(lr_emb, self.weight, self.bias)/self.t), dim = 1), \
        #                  torch.softmax((hr_pred/self.t), dim = 1), reduction = "batchmean")
        
        output = F.kl_div(torch.log_softmax(hr_pred/self.t, dim = 1), \
                          torch.softmax(lr_pred/self.t, dim = 1), reduction = "batchmean")
        
        return output
        
    def reset_parameters(self):
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    
# **************** 

# URL-1 : https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# URL-2 : https://github.com/MuggleWang/CosFace_pytorch/blob/master/main.py
#    Args:
#        in_features: size of each input sample
#        out_features: size of each output sample
#        s: norm of input feature
#        m: margin

class CosFace(nn.Module):
    
    def __init__(self, in_features, out_features, s = 30.0, m = 0.40):
        
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def forward(self, input, label = None):
        
        # cosine = self.cosine_sim(input, self.weight).clamp(-1,1)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1,1)
        
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)
        
        return output# , F.normalize(self.weight, p=2, dim=1), (cosine * one_hot)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'
    
    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)
    
# ****************

# URL : https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# Args:
#    in_features: size of each input sample
#    out_features: size of each output sample
#    s: norm of input feature
#    m: margin
#    cos(theta + m)

class ArcFace(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device = 'cuda:0'):
        
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # ***
        self.weight = Parameter(torch.FloatTensor(out_features, in_features)) 
        self.reset_parameters() 
        self.device = device
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1,1)
        
        # nan issues: https://github.com/ronghuaiyang/arcface-pytorch/issues/32
        # sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)),1e-9,1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        
        return output #, self.weight[label,:]
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'
    

# ****************

# URL : https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# Args:
#    in_features: size of each input sample
#    out_features: size of each output sample
#    m: margin
#    cos(m*theta)

class SphereFace(nn.Module):
   
    def __init__(self, in_features, out_features, m=4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'

# ****************
# SphereFace2
# URL: https://github.com/niliusha123/Margin-based-Softmax/blob/main/sphereface2.py

class SphereFace2(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, lamb = 0.7, r = 30, m = 0.4, t = 3, b = 0.25, device = 'cuda:0'):
        super(SphereFace2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lamb = lamb
        self.r = r
        self.m = m
        self.t = t
        self.b = b
        self.device = device
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: 2 * ((x + 1) / 2) ** self.t - 1,
        ]

    def forward(self, input, label):

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.r * (self.mlambda[0](cos_theta) - self.m) + self.b
        cos_m_theta1 = self.r * (self.mlambda[0](cos_theta) + self.m) + self.b
        cos_p_theta = (self.lamb / self.r) * torch.log(1 + torch.exp(-cos_m_theta))

        cos_n_theta = ((1 - self.lamb) / self.r) * torch.log(1 + torch.exp(cos_m_theta1))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        # one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot = one_hot.to(self.device) if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Calculate output ---------------------------
        loss = (one_hot * cos_p_theta) + (1 - one_hot) * cos_n_theta
        loss = loss.sum(dim=1)
        
        return loss.mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'

# ****************