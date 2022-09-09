_base_ = [
    '../../deit/deit-base_pt-16xb64_in1k.py'
]
# model settings
find_unused_parameters=True
use_logit = True
vitkd = True
wsld = False
dkd = False
kd = False
nkd = True
tf_nkd = False
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'deit-large_pt3.pth',
    use_logit = use_logit,
    vitkd = vitkd,
    tf_nkd = tf_nkd,
    distill_cfg = [ dict(methods=[dict(type='ViTKDLoss',
                                       name='loss_vitkd',
                                       use_this = vitkd,
                                       student_dims = 768,
                                       teacher_dims = 1024,
                                       alpha_vitkd=0.00003,
                                       beta_vitkd=0.000003,
                                       lambda_vitkd=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='WSLDLoss',
                                       name='loss_wsld',
                                       use_this = wsld,
                                       temp=2.0,
                                       alpha=2.5,
                                       num_classes=1000,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='DKDLoss',
                                       name='loss_dkd',
                                       use_this = dkd,
                                       temp=1.0,
                                       alpha=1.0,
                                       beta=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='NKDLoss',
                                       name='loss_nkd',
                                       use_this = nkd,
                                       temp=1.0,
                                       alpha=1.0,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_kd',
                                       use_this = kd,
                                       temp=1.0,
                                       alpha=0.5,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/deit/deit-base_pt-16xb64_in1k.py'
teacher_cfg = 'configs/deit/deit-large3_pt-16xb64_in1k.py'
