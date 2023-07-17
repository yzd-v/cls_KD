_base_ = [
    '../../mobilenet_v2/mobilenet-v2_8xb32_in1k.py'
]
# model settings
find_unused_parameters = False

# distillation settings
sd = True

# config settings
uskd = True

# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    sd = sd,
    teacher_pretrained = None,
    teacher_cfg = 'configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py',
    student_cfg = 'configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py',
    distill_cfg = [dict(methods=[dict(type='USKDLoss',
                                       name='loss_uskd',
                                       use_this=uskd,
                                       channel=320,
                                       alpha=1,
                                       beta=0.1,
                                       mu=0.005,
                                       )
                                ]
                        ),
                    ]
    )
