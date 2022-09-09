_base_ = [
    '../../shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py'
]
# model settings
find_unused_parameters=True
tf_nkd = True
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth',
    tf_nkd = tf_nkd,
    )

student_cfg = 'configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py'
teacher_cfg = 'configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py'