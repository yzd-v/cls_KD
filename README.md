# Knowledge Distillation for Image Classification (coming soon~)
This repository includes official implementation for the following papers:

* [NKD and tf-NKD](https://github.com/yzd-v/cls_KD/blob/master/nkd.md): Rethinking Knowledge Distillation via Cross-Entropy

* [ViTKD](https://github.com/yzd-v/cls_KD/blob/master/vitkd.md): ViTKD: Practical Guidelines for ViT feature knowledge distillation

It also provides unofficial implementation for the following papers:
* [KD](https://arxiv.org/abs/1503.02531), [DKD](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html), [WSLD](https://arxiv.org/abs/2102.00650)
* [MGD](https://arxiv.org/abs/2205.01529), [SRRL](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/70425/Tzimiropoulos%20Knowledge%20distillation%20via%202021%20Accepted.pdf?sequence=2)

If this repository is helpful, please give us a star ⭐ and cite relevant papers.

## Install
  - You can set the environment according to the folder [requirements](https://github.com/yzd-v/cls_KD/blob/master/requirements/).
  - Our codes are based on [MMClassification](https://github.com/open-mmlab/mmclassification). Please make sure you can run it successfully.
  - This repo uses mmcls = 0.23.2. If you want to use lower mmcls version for distillation, you can refer [MGD](https://github.com/yzd-v/MGD) to change the codes.

## Run
  - The implementation details of different methods can be seen in the folder [distillation](https://github.com/yzd-v/cls_KD/blob/master/mmcls/distillation/).
  - Please refer [nkd.md](https://github.com/yzd-v/cls_KD/blob/master/nkd.md) and [vitkd.md](https://github.com/yzd-v/cls_KD/blob/master/vitkd.md) to train the student.
  - You can modify the [configs](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/) to choose different distillation methods and pairs.

## Acknowledgement

Our code is based on the project [MMClassification](https://github.com/open-mmlab/mmclassification).
