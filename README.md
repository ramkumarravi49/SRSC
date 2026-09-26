# SRSC: Spike Regularization with Spike Curriculum

Code accompanying an anonymous submission on direct training of Spiking Neural
Networks (SNNs) for energy-efficient inference. SRSC applies a time-weighted
penalty on every spiking layer's output, held negligible until validation
accuracy crosses a threshold, then ramped geometrically to a ceiling. This
repository builds on the official codebase for
[Temporal Efficient Training (TET)](https://openreview.net/forum?id=_XNtisL32jv).

## Prerequisites

* Python >= 3.5
* PyTorch >= 1.9.0
* CUDA >= 10.2

## Repository structure

**Training entry points**
* `1_Train.py` — baseline direct training (TET or SDT loss), no spike regularization ("No SR").
* `2_Train_sr_proxy.py` — differentiable proxy spike-regularization variant.
* `3.1_Train_sr_physical.py` / `3.2_Train_sr_physical_new.py` — physical spike regularization at a fixed strength from epoch 0 (the collapse ablation); `3.2` adds checkpoint-resume support.
* `3.3_Train_sr_SpikeCurriculum.py` — SRSC: the accuracy-gated curriculum with a uniform per-layer coefficient.
* `3.32_Train_sr_SpikeCurriculum_EpochGate.py` — ablation: a fixed-epoch gate in place of the accuracy gate.
* `3.4_Train_sr_SpikeCurriculum_GradRank.py` — per-layer allocation variants (GR Binary Split, OWL GradProxy).
* `main_training_distribute.py` — multi-GPU training entry point.
* `Tune_Parllel.py` — training entry point used for hyperparameter sweeps.

**Checkpoint / log analysis**
* `4_spikes_TET_layer.py` — layer-wise spike and threshold inference on a trained checkpoint.
* `aux_inference_T.py` — layer-wise inference swept across multiple values of `T`.
* `aux2_parse_logs.py` — parses training logs for best test accuracy and NAS, with CSV export.
* `aux3_spike_sparsity.py` — computes spike sparsity (fraction of always-silent neurons) from a checkpoint.
* `aux4_wt_sparsity.py` — computes weight sparsity from a checkpoint.

**Data and models**
* `data_loaders.py` / `data_loaders_qcfs.py` — CIFAR-10/CIFAR-100/DVS-CIFAR10 loaders.
* `data_agument.py` — Cutout augmentation.
* `functions.py` — shared utilities (TET/SDT loss, seeding, logging).
* `models/` — network architectures (VGG, ResNet) and the LIF spiking layer.

## DVS-CIFAR10 preprocessing

* Download the CIFAR10-DVS dataset.
* Convert `.aedat` to `.mat` with `preprocess/test_dvs.m` in MATLAB.
* Build the train/test split with `preprocess/dvscifar_dataloader.py` (adapted from [this repo](https://github.com/aa-samad/conv_snn)).
* Pre-processed data: [7:3 split](https://drive.google.com/file/d/1s2csG5eagX3ZMfFpZCd5d7g8zqJxht4U/view?usp=drive_link) or [9:1 split](https://drive.google.com/file/d/1JGKL5avOrY3rgjHAamWGeP9RBvXqkBzm/view?usp=sharing).

## Notes

* The surrogate gradient (`ZIF`, triangle-like) and the spiking layer are in `models/layers.py`.
* Snn convolution layers follow `[batch, time, ...]` ordering for the first two tensor dimensions.

## Citation

This work builds on Temporal Efficient Training:
```
@inproceedings{
deng2022temporal,
title={Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting},
author={Shikuang Deng and Yuhang Li and Shanghang Zhang and Shi Gu},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=_XNtisL32jv}
}
```
