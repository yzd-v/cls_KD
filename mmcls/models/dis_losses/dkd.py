import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

def _get_gt_mask(logit, gt_label):
    gt_label = gt_label.reshape(-1)
    mask = torch.zeros_like(logit).scatter_(1, gt_label.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logit, gt_label):
    gt_label = gt_label.reshape(-1)
    mask = torch.ones_like(logit).scatter_(1, gt_label.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

@MODELS.register_module()
class DKDLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 temp=4.0,
                 alpha=1.0,
                 beta=6.0,
                 ):
        super(DKDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.beta = beta

    def forward(self, logit_s, logit_t, gt_label):
        gt_mask = _get_gt_mask(logit_s, gt_label)
        other_mask = _get_other_mask(logit_s, gt_label)
        pred_student = F.softmax(logit_s / self.temp, dim=1)
        pred_teacher = F.softmax(logit_t / self.temp, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (self.temp**2)
            / gt_label.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logit_t / self.temp - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logit_s / self.temp - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (self.temp**2)
            / gt_label.shape[0]
        )

        return self.alpha * tckd_loss + self.beta * nckd_loss

        