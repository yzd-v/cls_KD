import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

@MODELS.register_module()
class USKDLoss(nn.Module):

    """ PyTorch version of USKD """

    def __init__(self,
                 name,
                 use_this,
                 channel=None,
                 alpha=1.0,
                 beta=0.1,
                 mu=0.005,
                 num_classes=1000,
                 ):
        super(USKDLoss, self).__init__()

        self.channel = channel
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.fc = nn.Linear(channel, num_classes)
        self.zipf = 1 / torch.arange(1, num_classes + 1).cuda()

    def forward(self, fea_mid, logit_s, gt_label):

        if len(gt_label.size()) > 1:
            value, label = torch.sort(gt_label, descending=True, dim=-1)
            value = value[:,:2]
            label = label[:,:2]
        else:
            label = gt_label.view(len(gt_label), 1)
            value = torch.ones_like(label)

        N,c = logit_s.shape

        # final logit
        s_i = F.softmax(logit_s, dim=1)
        s_t = torch.gather(s_i, 1, label)

        # soft target label
        p_t = s_t**2
        p_t = p_t + value - p_t.mean(0, True)
        p_t[value==0] = 0
        p_t = p_t.detach()

        s_i = self.log_softmax(logit_s)
        s_t = torch.gather(s_i, 1, label)
        loss_t = - (p_t * s_t).sum(dim=1).mean()

        # weak supervision
        if len(gt_label.size()) > 1:
            target = gt_label * 0.9 + 0.1 * torch.ones_like(logit_s) / c
        else:
            target = torch.zeros_like(logit_s).scatter_(1, label, 0.9) + 0.1 * torch.ones_like(logit_s) / c
        
        # weak logit
        w_fc = self.fc(fea_mid)
        w_i = self.log_softmax(w_fc)
        loss_weak = - (self.mu * target * w_i).sum(dim=1).mean()

        # N*class
        w_i = F.softmax(w_fc, dim=1)
        w_t = torch.gather(w_i, 1, label)

        # rank
        rank = w_i / (1 - w_t.sum(1, True) + 1e-6) + s_i / (1 - s_t.sum(1, True) + 1e-6)

        # soft non-target labels
        _, rank = torch.sort(rank, descending=True, dim=-1)
        z_i = self.zipf.repeat(N, 1)
        ids_sort = torch.argsort(rank)
        z_i = torch.gather(z_i, dim=1, index=ids_sort)

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()

        logit_s = logit_s[mask].reshape(N, -1)
        ns_i = self.log_softmax(logit_s)

        nz_i = z_i[mask].reshape(N, -1)
        nz_i = nz_i/nz_i.sum(dim=1, keepdim=True)

        nz_i = nz_i.detach()
        loss_non = - (nz_i *ns_i).sum(dim=1).mean()

        # overall
        loss_uskd = self.alpha * loss_t + self.beta * loss_non + loss_weak

        return loss_uskd