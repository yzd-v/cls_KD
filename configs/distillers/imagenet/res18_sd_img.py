_base_ = [
    '../../resnet/resnet18_b32x8_imagenet.py'
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
                                       channel=256,
                                       alpha=1,
                                       beta=0.1,
                                       mu=0.005,
                                       )
                                ]
                        ),
                    ]
    )

student_cfg = 'configs/resnet/resnet18_b32x8_imagenet.py'
teacher_cfg = 'configs/resnet/resnet34_b32x8_imagenet.py'
