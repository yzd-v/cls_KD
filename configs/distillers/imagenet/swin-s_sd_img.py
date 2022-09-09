_base_ = [
    '../../swin_transformer/swin-small_16xb64_in1k.py'
]
# model settings
find_unused_parameters=True
tf_nkd = True
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth',
    tf_nkd = tf_nkd,
    )

student_cfg = 'configs/swin_transformer/swin-small_16xb64_in1k.py'
teacher_cfg = 'configs/swin_transformer/swin-small_16xb64_in1k.py'
