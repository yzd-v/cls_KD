import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import DISTILL_LOSSES

@DISTILL_LOSSES.register_module()
class NKDLoss(nn.Module):

    """ PyTorch version of NKD """

    def __init__(self,
                 name,
                 use_this,
                 temp=1.0,
                 gamma=1.5,
                 ):
        super(NKDLoss, self).__init__()

        self.temp = temp
        self.gamma = gamma

    def forward(self, logit_s, logit_t, gt_label):
        
        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        s_i = F.softmax(logit_s, dim=1)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        mask = torch.zeros_like(logit_s).scatter_(1, label, 1).bool()
        logit_s = logit_s - 1000 * mask
        logit_t = logit_t - 1000 * mask
        
        # N*class
        T_i = F.softmax(logit_t/self.temp, dim=1)
        S_i = F.softmax(logit_s/self.temp, dim=1)
        # N*1
        T_t = torch.gather(T_i, 1, label)
        S_t = torch.gather(S_i, 1, label)
        # N*class 
        nt_i = T_i/(1-T_t)
        ns_i = S_i/(1-S_t)
        nt_i[T_i==T_t] = 0
        ns_i[T_i==T_t] = 1

        loss_t = - (t_t * torch.log(s_t)).mean()

        loss_non =  (nt_i * torch.log(ns_i)).sum(dim=1).mean()
        loss_non = - self.gamma * (self.temp**2) * loss_non

        return loss_t + loss_non 