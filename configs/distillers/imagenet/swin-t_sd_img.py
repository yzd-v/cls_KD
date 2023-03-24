_base_ = [
    '../../swin_transformer/swin-tiny_16xb64_in1k.py'
]
# model settings
find_unused_parameters = True

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
                                       channel=384,
                                       alpha=1,
                                       beta=0.1,
                                       mu=0.005,
                                       )
                                ]
                        ),
                    ]
    )

student_cfg = 'configs/swin_transformer/swin-tiny_16xb64_in1k.py'
teacher_cfg = 'configs/swin_transformer/swin-tiny_16xb64_in1k.py'
