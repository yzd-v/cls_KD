_base_ = [
    '../../mobilenet_v1/mobilenet_v1.py'
]
# model settings
find_unused_parameters = False

# distillation settings
sd = True

# config settings
uskd = True

# method details
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = None,
    sd = sd,
    distill_cfg = [dict(methods=[dict(type='USKDLoss',
                                       name='loss_uskd',
                                       use_this=uskd,
                                       channel=512,
                                       alpha=1,
                                       beta=0.1,
                                       mu=0.005,
                                       )
                                ]
                        ),
                    ]
    )

student_cfg = 'configs/mobilenet_v1/mobilenet_v1.py'
teacher_cfg = 'configs/mobilenet_v1/mobilenet_v1.py'