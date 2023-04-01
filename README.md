# Pose detection

An Tensorflow implementation of PersonLab for Multi-Person Pose Estimation. Identify every person instance, localize its facial and body keypoints, and estimate its instance segmentation mask.

Introduction
Code repo for reproducing 2018 ECCV paper PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model
***


The augmentation code  and data iterator code is heavily borrowed from [this fork](https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation) of the Keras implementation of CMU's "Realtime Multi-Person Pose Estimation".

The model loss function and modelling are from [this fork](https://github.com/scnuhealthy/Tensorflow_PersonLab) of the Keras implementation of PersonLab.

### Citation

```
@inproceedings{papandreou2018personlab,
  title={PersonLab: Person pose estimation and instance segmentation with a bottom-up, part-based, geometric embedding model},
  author={Papandreou, George and Zhu, Tyler and Chen, Liang-Chieh and Gidaris, Spyros and Tompson, Jonathan and Murphy, Kevin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={269--286},
  year={2018}
}
```