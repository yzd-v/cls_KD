_base_ = [
    '../../swin_transformer/swin-tiny_16xb64_in1k.py'
]
# model settings
find_unused_parameters=True
tf_nkd = True
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth',
    tf_nkd = tf_nkd,
    )

student_cfg = 'configs/swin_transformer/swin-tiny_16xb64_in1k.py'
teacher_cfg = 'configs/swin_transformer/swin-tiny_16xb64_in1k.py'
