import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

@MODELS.register_module()
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
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logit_s, logit_t, gt_label):
        
        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        N, c = logit_s.shape
        s_i = self.log_softmax(logit_s)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        loss_t = - (t_t * s_t).mean()

        mask = torch.ones_like(logit_s).scatter_(1, label, 1).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)
        
        # N*class
        S_i = self.log_softmax(logit_s/self.temp)
        T_i = F.softmax(logit_t/self.temp, dim=1)     

        loss_non =  (T_i * S_i).sum(dim=1).mean()
        loss_non = - self.gamma * (self.temp**2) * loss_non

        return loss_t + loss_non 