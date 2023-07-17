_base_ = [
    '../_base_/models/resnet18_cifar.py',
    '../_base_/datasets/cifar100_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        out_indices=(2,3)),
    head=dict(
    num_classes=100,
    ))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
