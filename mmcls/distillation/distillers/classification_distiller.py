import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.models import build_classifier
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict


@DISTILLER.register_module()
class ClassificationDistiller(BaseClassifier):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg = None,
                 teacher_pretrained = None,
                 use_logit = False,
                 is_vit = False,
                 sd = False):

        super(ClassificationDistiller, self).__init__()
        
        if not sd:
            self.teacher = build_classifier(teacher_cfg.model)
            if teacher_pretrained:
                self.init_weights_teacher(teacher_pretrained)
            self.teacher.eval()

        self.use_logit = use_logit
        self.is_vit = is_vit
        self.sd = sd

        self.student= build_classifier(student_cfg.model)
        self.student.init_weights()
            
        self.distill_cfg = distill_cfg   
        self.distill_losses = nn.ModuleDict()
        if self.distill_cfg is not None:  
            for item_loc in distill_cfg:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    use_this = item_loss.use_this
                    if use_this:
                        self.distill_losses[loss_name] = build_distill_loss(item_loss)
    
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def forward_train(self, 
                      img, 
                      gt_label, 
                      **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """

        if self.student.augments is not None:
            img, gt_label = self.student.augments(img, gt_label)

        fea_s = self.student.extract_feat(img, stage='backbone')
        x = fea_s
        if self.student.with_neck:
            x = self.student.neck(x)
        if self.student.with_head and hasattr(self.student.head, 'pre_logits'):
            x = self.student.head.pre_logits(x)
        
        if self.is_vit:
            logit_s = self.student.head.layers.head(x)
        else:
            logit_s = self.student.head.fc(x)
        loss = self.student.head.loss(logit_s, gt_label)

        s_loss = dict()
        for key in loss.keys():
            s_loss['ori_'+key] = loss[key]

        """ KD """
        if not self.sd:
            with torch.no_grad():
                fea_t = self.teacher.extract_feat(img, stage='backbone')
                if self.use_logit:
                    if self.is_vit:
                        logit_t = self.teacher.head.layers.head(self.teacher.head.pre_logits(fea_t))
                    else:
                        logit_t = self.teacher.head.fc(self.teacher.head.pre_logits(self.teacher.neck(fea_t)))

            all_keys = self.distill_losses.keys()
            if 'loss_vitkd' in all_keys:
                loss_name = 'loss_vitkd'
                s_loss[loss_name] = self.distill_losses[loss_name](fea_s[-1][0], fea_t[-1][0]) 

            if ('loss_srrl' in all_keys) and self.use_logit:
                loss_name = 'loss_srrl'
                fea_s_align = self.distill_losses[loss_name].Connectors(fea_s[-1])
                logit_st = self.teacher.head.fc(self.teacher.head.pre_logits(self.teacher.neck(fea_s_align)))         
                s_loss[loss_name] = self.distill_losses[loss_name](fea_s_align, fea_t[-1], logit_st,logit_t)

            if 'loss_mgd' in all_keys:
                loss_name = 'loss_mgd'
                s_loss[loss_name] = self.distill_losses[loss_name](fea_s[-1], fea_t[-1])

            if ('loss_wsld' in all_keys) and self.use_logit:
                loss_name = 'loss_wsld'
                s_loss[loss_name] = self.distill_losses[loss_name](logit_s, logit_t, gt_label)

            if ('loss_dkd' in all_keys) and self.use_logit:
                loss_name = 'loss_dkd'
                s_loss[loss_name] = self.distill_losses[loss_name](logit_s, logit_t, gt_label)

            if ('loss_kd' in all_keys) and self.use_logit:
                loss_name = 'loss_kd'
                ori_alpha, s_loss[loss_name] = self.distill_losses[loss_name](logit_s, logit_t)
                s_loss['ori_loss'] = ori_alpha * s_loss['ori_loss']

            if ('loss_nkd' in all_keys) and self.use_logit:
                loss_name = 'loss_nkd'
                s_loss[loss_name] = self.distill_losses[loss_name](logit_s, logit_t, gt_label)

        """ Self-KD """
        if self.sd:
            all_keys = self.distill_losses.keys()
            if 'loss_uskd' in all_keys:
                loss_name = 'loss_uskd'
                if self.is_vit:
                    fea_mid = fea_s[-1][0][2]
                else:
                    fea_mid = self.student.head.pre_logits(self.student.neck(fea_s[-2]))
                s_loss[loss_name] = self.distill_losses[loss_name](fea_mid, logit_s, gt_label)

        return s_loss
    
    def simple_test(self, img, img_metas=None, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def extract_feat(self, imgs, stage='neck'):
        """Extract features from images.
          'backbone', 'neck', 'pre_logits'
        """
        return self.student.extract_feat(imgs, stage)


