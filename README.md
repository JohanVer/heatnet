# HeatNet: Bridging the Day-Night Domain Gap in Semantic Segmentation with Thermal Images

<p align="center"> <img src='docs/teaser.png' align="center" height="230px"> </p>

## Abstract

The majority of learning-based semantic segmentation methods are optimized for daytime scenarios and favorable lighting conditions.
Real-world driving scenarios, however, entail adverse environmental conditions such as nighttime illumination or glare which remain
a challenge for existing approaches. In this work, we propose a multimodal semantic segmentation model that can be applied during
daytime and nighttime. To this end, besides RGB images, we leverage thermal images, making our network significantly more robust.
We avoid the expensive annotation of nighttime images by leveraging an existing daytime RGB-dataset and propose a teacher-student
training approach that transfers the dataset's knowledge to the nighttime domain. We further employ a domain adaptation method to
align the learned feature spaces across the domains and propose a novel two-stage training scheme. Furthermore, due to a lack of
thermal data for autonomous driving, we present a new dataset comprising over 20,000 time-synchronized and
aligned RGB-thermal image pairs. In this context, we also present a novel target-less calibration method that allows
for automatic robust extrinsic and intrinsic thermal camera calibration. Among others, we employ our new dataset to
show state-of-the-art results for nighttime semantic segmentation.


# Dataset preparation

1. Download the dataset

Train:
wget http://aisdatasets.informatik.uni-freiburg.de/freiburg-thermal-segmentation/train.zip

Test:
wget http://aisdatasets.informatik.uni-freiburg.de/freiburg-thermal-segmentation/test.zip


2. Process the dataset

Unzip the data wherever you want it:
unzip train.zip
unzip test.zip


# Environment

Best: create a conda env with requirements.txt

# Train

All model parameters can be turned on or off using the provided run arguments

The configuration with the best overall mIoU can be trained with the following command

```
python train.py --batch_size 8 --iter_initial_critic_phase 1 --iter_seg_phase 1 --iter_critic_phase 1 --lr 0.0001 --modalities rgb_ir --checkpointname my_experiment_0 --gpus 0 1 2 3 --arch pspnet --night_supervision_model checkpoint_1_best.pth --weight_ir_sup --conf_weight 0.01 --num_critics 1 --late_fusion --multidir --dataroot /path/to/train/dir/ --testroot_day /path/to/test/dir/day/ --testroot_night /path/to/test/dir/night/
```

# Evaluate

You can evaluate the model with

```
python evaluate.py --cuda --weights /path/to/your/weights --testroot_day /path/to/testdata-day/ --testroot_night /path/to/testdata-night/
```

## Pretrained weights

The pretrained weights are available here: #TODO
