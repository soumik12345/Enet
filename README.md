# Enet

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/soumik12345/Enet/master)

Pytorch Implementation of ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation (https://arxiv.org/abs/1606.02147).

![Training Results](./assets/enet_camvid_gif-1.gif)

### Training Notebooks

- [Enet with PReLU Encoder on CamVid](https://github.com/soumik12345/Enet/blob/master/Enet_CamVid.ipynb)
- [Enet with Mish Encoder on CamVid](https://github.com/soumik12345/Enet/blob/master/Enet_CamVid_Mish.ipynb)

## Training Results on CamVid

### Inference on Training Data

![Training Inference Results](./assets/image_train.png)

### Inference on Validation Data

![Validation Inference Results](./assets/image_val.png)

## TODO

- [x] Implement Vanilla Enet Architecture
- [x] Encorporate Custom Activations for Codebase
- [x] Train Enet on CamVid
- [x] Train Enet with Mish Encoder on CamVid
- [ ] Experiment to find best Mish Version of Enet
- [ ] Repeat Same experiments for Cityscapes Dataset
- [ ] Repeat Same experiments for SUN RGB-D Dataset


## References

- [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)
- [Tensorflow Enet](https://github.com/kwotsin/TensorFlow-ENet)
- [ENet on TowardsDataScience](https://towardsdatascience.com/enet-a-deep-neural-architecture-for-real-time-semantic-segmentation-2baa59cf97e9)
- [Mish: Self Regularized Non-Monotonic Activation Function](https://github.com/digantamisra98/Mish)
- [Mish on Arxiv](https://arxiv.org/abs/1908.08681)
