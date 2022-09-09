_base_ = [
    '../../mobilenet_v2/mobilenet-v2_8xb32_in1k.py'
]
# model settings
find_unused_parameters=True
tf_nkd = True
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
    tf_nkd = tf_nkd,
    )

student_cfg = 'configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py'
teacher_cfg = 'configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py'