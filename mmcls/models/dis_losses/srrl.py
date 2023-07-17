import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

@MODELS.register_module()
class SRRLLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 student_channels,
                 teacher_channels,
                 alpha=1.0,
                 beta=5.0,
                 ):
        super(SRRLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.Connectors = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(teacher_channels), nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fea_s, fea_t, logit_st, logit_t):
        x = fea_s
        y = fea_t
        x = x.view(x.size(0),x.size(1),-1)
        y = y.view(y.size(0),y.size(1),-1)
        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        fm_loss = (x_mean-y_mean).pow(2).mean(1)
        sr_loss = F.mse_loss(logit_st, logit_t)

        return self.alpha * fm_loss.mean() + self.beta * sr_loss


        