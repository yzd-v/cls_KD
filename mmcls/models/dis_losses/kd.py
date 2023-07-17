import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

@MODELS.register_module()
class KDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 temp=4.0,
                 alpha=0.5,
                 ):
        super(KDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha

    def forward(self, logit_s, logit_t):
        # N*class
        S_i = F.softmax(logit_s/self.temp, dim=1)
        T_i = F.softmax(logit_t/self.temp, dim=1)


        kd_loss = - self.alpha*(self.temp**2)*(T_i*torch.log(S_i)).sum(dim=1).mean()
        return  1 - self.alpha, kd_loss

        