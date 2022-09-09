_base_ = [
    '../../resnet/resnet18_8xb16_cifar100.py'
]
# model settings
find_unused_parameters=True
use_logit = True
srrl = False
mgd = False
wsld = False
dkd = False
kd = False
nkd = True
tf_nkd = False
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'work_dirs/resnet34_8xb16_cifar100/latest.pth',
    use_logit = use_logit,
    tf_nkd = tf_nkd,
    distill_cfg = [ dict(methods=[dict(type='SRRLLoss',
                                       name='loss_srrl',
                                       use_this = srrl,
                                       student_channels = 512,
                                       teacher_channels = 512,
                                       alpha=1.0,
                                       beta=5.0,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='MGDLoss',
                                       name='loss_mgd',
                                       use_this = mgd,
                                       student_channels = 512,
                                       teacher_channels = 512,
                                       alpha_mgd=0.0012,
                                       lambda_mgd=0.15,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='WSLDLoss',
                                       name='loss_wsld',
                                       use_this = wsld,
                                       temp=4.0,
                                       alpha=2.25,
                                       num_classes=100,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='DKDLoss',
                                       name='loss_dkd',
                                       use_this = dkd,
                                       temp=4.0,
                                       alpha=1.0,
                                       beta=8.0,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='NKDLoss',
                                       name='loss_nkd',
                                       use_this = nkd,
                                       temp=3.0,
                                       alpha=3.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_kd',
                                       use_this = kd,
                                       temp=3.0,
                                       alpha=0.5,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/resnet/resnet18_8xb16_cifar100.py'
teacher_cfg = 'configs/resnet/resnet34_8xb16_cifar100.py'
