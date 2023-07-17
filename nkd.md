# NKD and USKD
ICCV 2023 Paper: [From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels](https://arxiv.org/abs/2303.13005)

![architecture](imgs/architecture.jpg)

## Train

```
#single GPU
python tools/train.py configs/distillers/imagenet/res18_sd_img.py

#multi GPU
bash tools/dist_train.sh configs/distillers/imagenet/res34_distill_res18_img.py 8
```

## Transfer
```
# Tansfer the Distillation model into mmcls model
python pth_transfer.py --dis_path $dis_ckpt --output_path $new_mmcls_ckpt
```
## Test

```
#single GPU
python tools/test.py configs/resnet/resnet18_8xb32_in1k.py $new_mmcls_ckpt --metrics accuracy

#multi GPU
bash tools/dist_test.sh configs/resnet/resnet18_8xb32_in1k.py $new_mmcls_ckpt 8 --metrics accuracy
```

## Results
### NKD
|  Model   | Teacher  | Baseline(Top-1 Acc) | +NKD(Top-1 Acc) |                            dis_config                            | weight |
| :------: | :-------: | :----------------: | :------------: | :----------------------------------------------------------: | :--: |
|   ResNet18   | ResNet34 |        69.90        |      71.96 (+2.06)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/res34_distill_res18_img.py) | [baidu](https://pan.baidu.com/s/1u82mk5SWYLxin6AKv9fPPw?pwd=sodb)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnWB-ra7Zwe1T7SNO?e=iQhdde) |
| MobileNet | ResNet50 |        69.21        |      72.58 (+3.37)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/res50_distill_mv1_img.py) | [baidu](https://pan.baidu.com/s/1uENiLmj5HpYyLY0dTkeeMg?pwd=paak)/[one drive](https://1drv.ms/u/s!Ah7OVljahSArnVoW0JxXFAZXVoOf?e=yKaAba) |
| DeiT-Tiny | DeiT III-Small |        74.42        |      76.68 (+2.26)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/deit-s3_distill_deit-t_img.py) |  |
| DeiT-Base | DeiT III-Large |        81.76        |      84.96 (+3.20)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/deit-s3_distill_deit-t_img.py) |  |

### USKD
|  Model   | Baseline(Top-1 Acc) | +tf-NKD(Top-1 Acc) |                            dis_config                            |
| :------: | :----------------: | :------------: | :----------------------------------------------------------: |
| MobileNet |        69.21        |      70.38 (+1.17)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/mv1_sd_img.py) |
| MobileNetV2 |        71.86        |      72.41 (+0.55)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/mv2_sd_img.py) |
| ShuffleNetV2 |        69.55        |      70.30 (+0.75)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/mv1_sd_img.py) |
|   ResNet18   |        69.90        |      70.79 (+0.89)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/res18_sd_img.py) |
|   ResNet50   |        76.55        |      77.07 (+0.52)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/res50_sd_img.py) |
|   ResNet101   |        77.97        |      78.54 (+0.57)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/res101_sd_img.py) |
|   RegNetX-1.6GF   |        76.84        |      77.30 (+0.46)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/res101_sd_img.py) |
|   Swin-Tiny   |        81.18        |      81.49 (+0.31)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/swin-t_sd_img.py) |
|   DeiT-Tiny   |        74.42        |      74.97 (+0.55)      | [config](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/imagenet/deit-t_sd_img.py) |

## Citation
```
@article{yang2023knowledge,
  title={From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels},
  author={Yang, Zhendong and Zeng, Ailing and Li, Zhe and Zhang, Tianke and Yuan, Chun and Li, Yu},
  journal={arXiv preprint arXiv:2303.13005},
  year={2023}
}
```