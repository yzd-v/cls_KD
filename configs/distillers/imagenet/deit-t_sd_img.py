_base_ = [
    '../../deit/deit-tiny_pt-4xb256_in1k.py'
]
# model settings
find_unused_parameters = False

# distillation settings
sd = True
is_vit = True

# config settings
uskd = True

# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    sd = sd,
    is_vit = is_vit,
    teacher_pretrained = None,
    teacher_cfg = 'configs/deit/deit-tiny_pt-4xb256_in1k.py',
    student_cfg = 'configs/deit/deit-tiny_pt-4xb256_in1k.py',
    distill_cfg = [dict(methods=[dict(type='USKDLoss',
                                       name='loss_uskd',
                                       use_this=uskd,
                                       channel=192,
                                       alpha=1,
                                       beta=0.1,
                                       mu=0.005,
                                       )
                                ]
                        ),
                    ]
    )

