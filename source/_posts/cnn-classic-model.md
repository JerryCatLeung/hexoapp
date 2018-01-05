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

本文要介绍的前几个模型都是ILSVRC竞赛历年的佼佼者，这里先来总览比较AlexNet、VGG、GoogLeNet、ResNet四个模型。见下表：

|模型| AlexNet | VGG | GoogleNet | ResNet|
|---|---|---|---|---|
|时间 |2012 |2014 |2014 |2015 |
|层数 |8 |19 |22 |152 |
|错误率 |16.4% |7.3% |6.7% |3.57% |
|Data Augmentation |+ |+ |+ |+ |
|Inception(NIN) |- |- |+ |- |
|卷积层数 |5 |16 |21 |151 |
|卷积核大小 |11,5,3 |3 |7,1,3,5 |7,1,3,5 |
|全连接层数 |3 |3 |1 |1 |
|全连接层大小 |4096,4096,1000 |4096,4096,1000 |1000 |1000 |
|Dropout |+ |+ |+ |+ |
|LRN |+ |- |+ |- |
|Batch Normalization |- |- |- |+ |
## LeNet
参考[Cafe LeNet的配置文件](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_train_test.prototxt)，由两个卷积层，两个池化层，两个全连接层组成。

- Input: 32*32 
- kernel size：5*5
- stride: 1
- pooling: MAX
- Architecture: 2 conv, 2 pooling, 2 FC
- Output: 10 classes (0-9 probability)

下图是类似的网络结构，和cafe的不完全一样，可以帮忙理解结构。

![Arch of LeNet-5](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FLeNet.png?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-02T12%3A31%3A07Z%2F-1%2Fhost%2Fdab6ce8ce33f1bb2c06afd5fc62fc991bcc814a71a7957b73490fdc31dd02184)

## AlexNet
2012年ImageNet比赛冠军的model，AlexNet以第一作者Alex命名。[cafe的网络配置在此处](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt)。
废话不说，先上一张经典的AlexNet网络结构的半截图（仔细看下，图的上半部分好像被截断了）。

![Arch of AlexNet](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FAlexNet.png?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-02T12%3A51%3A36Z%2F-1%2Fhost%2F4b029f787bdf5d1bd15da5f6652247e3d38d9c870e1078b0f3bf83a7d352fdf8)
AlexNet以Top-5的错误率为16.4%赢得ILSVRC 2012年的比赛。它做出了如下创新：

- 首次使用ReLU作为CNN激活函数，解决了Sigmod激活函数的梯度弥散问题，关于激活函数可参考[上一篇激活函数部分](http://www.laphiler.com/2017/12/20/CNN_startup/)
- 使用Dropout随机丢弃部分神经元，可以避免模型的过拟合。AlexNet的最后几个全连接层使用了Dropout。
- 使用重叠的最大池化，之前使用的都是平均池化。最大池化可以避免平均池化的模糊效果。同时，步长比卷积核的尺寸小，这样池化层的输出之间会有重叠，提升了特征的丰富性。
- 提出LRN层（局部相应一体化），对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对较大，并抑制其他反馈较小的神经元，增强模型泛化能力。
- 使用CUDA加速深度卷积网络的训练，使用了GPU的并行计算能力。
- 数据增强，随机从256x256的原始图中截取224x224大小的区域，再做水平翻转，相当于增加了 (256−224)2×2=2048(256−224)2×2=2048 倍的数据量。仅靠原始的数据量，参数众多的CNN会陷入过拟合。预测时，取图片四个角和中间共5个位置，再加上翻转，共10个位置，对它们的预测结果求均值。

**AlexNet结构图**（换个视角）
![](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FAlexNet_arch.png?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-03T08%3A43%3A20Z%2F-1%2Fhost%2Fd88f9621d1f6238555637b5f10122dc7b08c987c26e8888c99ed083bf6a517f2)
卷积神经网络的结构并不是各个层的简单组合，它是由一个个“模块”有机组成的，在模块内部，各个层的排列是有讲究的。比如AlexNet的结构图，它是由八个模块组成的。
**模块1和模块2**是CNN的前面部分
	
	卷积-激活函数-池化-标准化 

构成了一个基本计算模块，也可以说是一个卷积过程的**标配**，CNN的结构就是这样，从宏观的角度来看，就是一层卷积，一层降采样这样循环的，中间适当地插入一些函数来控制数值的范围，以便后续的循环计算。

**模块3和模块4**也是两个卷积过程，差别是少了降采样，原因就跟输入的尺寸有关，特征的数据量已经比较小了，所以没有降采样，这个都没有关系啦。

**模块5**也是一个卷积过程，和模块1，2一样。

**模块6和模块7**就是所谓的全连接层，全连接层就和人工神经网络的结构一样的，结点数超级多，连接线也超多，所以这儿引出了一个dropout层，来去除一部分没有足够激活的层。

**模块8**就是一个输出的结果，结合上softmax做出分类。有几类，输出几个结点，每个结点保存的是属于该类别的概率值。

## VGGNet

VGGNet可以看做是加深版的AlexNet，都是Conv Layer加FC Layer。
VGGNet探索卷积神经网络**深度与性能**的关系，通过反复堆叠3x3的小型卷积核和2x2的最大池化层，VGGNet成功构筑了16～19层深的卷积神经网络，VGGNet取得ILSVRC 2014比赛分类项目的第2名和定位项目的第1名。同时VGGNet的拓展性很强，迁移到其他图片数据上的泛化性非常好。
VGGNet论文中全部使用3x3的卷积核，通过不断加深网络来提升性能，表1表示为不同的**网络结构图**，表2表示每一级别的**参数量**。从11层的网络一直到19层的网络都有详尽的性能测试。

![](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FVGGNet.jpg?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-05T08%3A52%3A50Z%2F-1%2Fhost%2F19399a9e990c262d586d187cc435ac9ac2529083d5b20b747865d75b4237a797)

**VGGNet特性**

- 虽然从A到E每一级网络逐渐变深，但是网络的参数量并没有增长很多，这是因为参数量主要都消耗在最后3个全连接层。前面的卷积部分虽然很深，但是消耗的参数量不大。
- D、E也就是我们常说的VGGNet-16和VGGNet-19。C很有意思，相比B多了几个1x1的卷积层，1x1卷积的意义主要在于线性变换，而输入通道数和输出通道数不变，没有发生降维。
- 两个3x3的卷积层串联相当于1个5x5的卷积层，即一个像素会跟周围5x5的像素产生关联，可以说感受野大小为5x5。而3个3x3的卷积层串联的效果则相当于1个7x7的卷积层。除此之外，3个串联的3x3的卷积层，拥有比1个7x7的卷积层更少的参数量，只有后者的(3x3x3)/(7x7)=55%。最重要的是，3个3x3的卷积层拥有比1个7x7的卷积层更多的非线性变换（前者可以使用三次ReLU激活函数，而后者只有一次），使得CNN对特征的学习能力更强。
- 先训练级别A的简单网络，再复用A网络的权重来初始化后面的几个复杂模型，这样训练收敛的速度更快。

**与AlexNet比较**
比AlexNet收敛的要快一些，原因为：（1）使用小卷积核和更深的网络进行的正则化；（2）在特定的层使用了预训练得到的数据进行参数的初始化。
对于较浅的网络，如网络A，可以直接使用随机数进行随机初始化，而对于比较深的网络，则使用前面已经训练好的较浅的网络中的参数值对其前几层的卷积层和最后的全连接层进行初始化。

## GoogLeNet
[caffe的GoogleNet网络配置](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt)，直观上看GoogleNet是非常深的神经网络模型，本文介绍关于GoogleNet第一篇正式论文，习惯称为inception v1，GoogleNet的主要贡献：

1. 提出Inception Architecture并对其优化
2. 取消全连层
3. 运用auxiliary classifiers加速网络converge

**Inception Architecture**

作者发现传统提高网络精度或性能的方法是一条邪路（P.S.传统方法指的是**扩大网络模型**或**增大训练数据集**），而想从本质上提高网络性能，就得用sparsely connected architectures，即“稀疏连接结构”。
对IA，可以理解为用尽可能的“小”、“分散”的可堆叠的网络结构，去学习复杂的分类任务，如下图：
![](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FInception.png?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-05T12%3A29%3A51Z%2F-1%2Fhost%2F55336fc76f309bdb9da55bb513fd1cd792b1480b62319ace6a3b177a18714104)
原来造神经网络，都是一条线下来，我们可以回想一下AlexNet、VGG等著名网络，而IA是“分叉-汇聚”型网络，也就是说在一层网络中存在多个不同尺度的kernels，卷积完毕后再汇聚，为了更好理解，“汇聚”的tensorflow代码写出来是这样的：
	
	net = tf.concat(3, [branch1x1, branch5x5, branch3x3, branch_pool])

这种网络结构会带来**参数爆炸**问题，所以在原机构基础上加入了kernels数量控制方式，就是那些1×1的卷积层，如下图：
![](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FInception2.png?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-05T12%3A36%3A38Z%2F-1%2Fhost%2F6c630dd782ce5050a1d2277c04a2214c8af9956fcdb457547f738b594e67980f)

> IA之所以能提高网络精度，可能就是归功于它拥有多个不同尺度的kernels，每一个尺度的kernel会学习不同的特征，把这些不同kernels学习到的特征汇聚给下一层，能够更好的实现全方位的深度学习！

**取消FC全连层**

为什么VGG网络的参数那么多？就是因为它在最后有两个4096的全连层！Szegedy吸取了教训，为了压缩GoogLeNet的网络参数，他把全连层取消了！
![](http://xiaoluban.bj.bcebos.com/laphiler%2FCNN_classic_model%2FGoogleNet_FC.png?authorization=bce-auth-v1%2F94767b1b37b14a259abca0d493cefafa%2F2018-01-05T12%3A40%3A35Z%2F-1%2Fhost%2F808e0e37a5ca8637c504df1b52e3bf9ee02c6a912e4bfb33eac397c8de31c6ca)

从上图就可以看出，网络的最后几层是avg pool、dropout、linear和softmax，没有看到fully connect的影子。现在取消全连层貌似是个大趋势，近两年的优秀大型神经网络都没有全连层，可能是全连层参数太多，网络深度增加了以后，难以接受吧

**Auxiliary classifiers**