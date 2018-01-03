---
title: CNN-经典神经网络模型
date: 2018-01-02 19:08:03
tags: [CNN, Deep learning]
---


本文涉及以下CNN经典模型附paper链接，LeNet、AlexNet、VGGNet、GoogleNet、ResNet、RCNN、Fast R-CNN、Faster R-CNN、YOLO。

<!--more-->

- LeNet (1998) [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- AlexNet (2012) [Imagenet classification with deep convolutional neural networks](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf)
- VGGNet (2014) [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)
- GoogleNet (2014) [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- ResNet (2015) [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- RCNN (2014) [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)
- Fast R-CNN (2015) [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- Faster R-CNN (2016) [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- YOLO (2016) [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

## LeNet
参考[Cafe LeNet的配置文件](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_train_test.prototxt)，由两个卷积层，两个池化层，两个全连接层组成。
- kernel size：5*5
- stride: 1
- pooling: MAX
- Architecture: 2 conv, 2 pooling, 2 FC

下图是类似的网络结构，和cafe的不完全一样，可以帮忙理解结构。

![Arch of LeNet-5](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FLeNet.png?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-02T12%3A31%3A07Z%2F-1%2Fhost%2Fdab6ce8ce33f1bb2c06afd5fc62fc991bcc814a71a7957b73490fdc31dd02184)

## AlexNet
2012年ImageNet比赛冠军的model，AlexNet以第一作者Alex命名。[cafe的网络配置在此处](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt)。
废话不说，先上一张经典的AlexNet网络结构的半截图（仔细看下，图的上半部分好像被截断了）。

![Arch of AlexNet](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FAlexNet.png?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-02T12%3A51%3A36Z%2F-1%2Fhost%2F4b029f787bdf5d1bd15da5f6652247e3d38d9c870e1078b0f3bf83a7d352fdf8)
