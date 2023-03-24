# Knowledge Distillation for Image Classification
This repository includes official implementation for the following papers:

* [NKD and USKD](https://github.com/yzd-v/cls_KD/blob/master/nkd.md): From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels

* [ViTKD](https://github.com/yzd-v/cls_KD/blob/master/vitkd.md): ViTKD: Practical Guidelines for ViT feature knowledge distillation

It also provides unofficial implementation for the following papers:
* [KD](https://arxiv.org/abs/1503.02531), [DKD](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html), [WSLD](https://arxiv.org/abs/2102.00650)
* [MGD](https://arxiv.org/abs/2205.01529), [SRRL](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/70425/Tzimiropoulos%20Knowledge%20distillation%20via%202021%20Accepted.pdf?sequence=2)

If this repository is helpful, please give us a star ⭐ and cite relevant papers.

## Install
  - Prepare the dataset in data/imagenet
  - ```
    # Set environment
    pip install -r requirements.txt
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```
  - This repo uses mmcls = 0.23.2. If you want to use lower mmcls version for distillation, you can refer [MGD](https://github.com/yzd-v/MGD) to change the codes.

## Run
  - Please refer [nkd.md](https://github.com/yzd-v/cls_KD/blob/master/nkd.md) and [vitkd.md](https://github.com/yzd-v/cls_KD/blob/master/vitkd.md) to train the student and get the weight.
  - You can modify the [configs](https://github.com/yzd-v/cls_KD/blob/master/configs/distillers/) to choose different distillation methods and pairs.
  - The implementation details of different methods can be seen in the folder [distillation](https://github.com/yzd-v/cls_KD/blob/master/mmcls/distillation/).

## Citing NKD and USKD
```
@article{yang2023knowledge,
  title={From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels},
  author={Yang, Zhendong and Zeng, Ailing and Li, Zhe and Zhang, Tianke and Yuan, Chun and Li, Yu},
  journal={arXiv preprint arXiv:2303.13005},
  year={2023}
}
```

## Citing ViTKD
```
@article{yang2022vitkd,
  title={ViTKD: Practical Guidelines for ViT feature knowledge distillation},
  author={Yang, Zhendong and Li, Zhe and Zeng, Ailing and Li, Zexian and Yuan, Chun and Li, Yu},
  journal={arXiv preprint arXiv:2209.02432},
  year={2022}
}
```

## Acknowledgement

Our code is based on the project [MMClassification](https://github.com/open-mmlab/mmclassification).
