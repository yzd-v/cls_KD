import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

@MODELS.register_module()
class WSLDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 temp = 4.0,
                 alpha=2.25,
                 num_classes=100,
                 ):
        super(WSLDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logit_s, logit_t, gt_label):
        s_input_for_softmax = logit_s / self.temp
        t_input_for_softmax = logit_t / self.temp
        t_soft_label = F.softmax(t_input_for_softmax, dim=1)
        logsoftmax = nn.LogSoftmax()
        softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)
        fc_s_auto = logit_s.detach()
        fc_t_auto = logit_t.detach()
        log_softmax_s = logsoftmax(fc_s_auto)
        log_softmax_t = logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(gt_label, num_classes=self.num_classes).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)
        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss
        wsld_loss = (self.temp ** 2) * torch.mean(softmax_loss)

        return self.alpha * wsld_loss

        