import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math

# *** *** *** *** ***
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

    
# **************** 

# URL-1 : https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# URL-2 : https://github.com/MuggleWang/CosFace_pytorch/blob/master/main.py
#    Args:
#        in_features: size of each input sample
#        out_features: size of each output sample
#        s: norm of input feature
#        m: margin

class CosFace(nn.Module):
    
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def forward(self, input, label=None):
        
        # cosine = self.cosine_sim(input, self.weight).clamp(-1,1)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1,1)
        
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i={x_i if condition_i else y_i) -------------
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

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device='cuda:0'):
        
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
        
        # -------------torch.where(out_i={x_i if condition_i else y_i) -------------
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