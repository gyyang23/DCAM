# DCAM
The official code of JVCIR 2023 paper ([DCAM: Disturbed Class Activation Maps for Weakly Supervised Semantic Segmentation](https://doi.org/10.1016/j.jvcir.2023.103852)).

## Citation
```
inproceedings{2023dcam,
  title={DCAM: Disturbed class activation maps for weakly supervised semantic segmentation},
  author={Lei, Jie and Yang, Guoyu and Wang, Shuaiwei and Feng, Zunlei and Liang, Ronghua},
  journal={Journal of Visual Communication and Image Representation},
  pages={103852},
  year={2023},
  publisher={Elsevier}
}
```

## Prerequisite
- Python 3.6, PyTorch 1.9.0, and others in environment.yml
- You can create the environment from environment.yml file
```
conda env create -f environment.yml
```
## Usage (PASCAL VOC)
### Step 1. Prepare dataset.
- Download PASCAL VOC 2012 devkit from [official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit). [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). 
- You need to specify the path ('voc12_root') of your downloaded devkit in the following steps.
### Step 2. Train DCAM and generate seeds.
- Please specify a workspace to save the model and logs.
```
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --train_cam_pass True --train_dcam_pass True --train_dcam_sce_pass True --make_dcam_sce_pass True --eval_cam_pass True 
```
### Step 3. Train IRN and generate pseudo masks.
```
CUDA_VISIBLE_DEVICES=0 python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True 
```
### Step 4. Train semantic segmentation network.
To train DeepLab-v2, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). 
To train DeepLab-v3+, we refer to [deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch). 
Please replace the groundtruth masks with generated pseudo masks.

## Acknowledgment
This code is borrowed from [ReCAM](https://github.com/zhaozhengChen/ReCAM), thanks Zhaozheng Chen.
