---
title: CNN经典神经网络模型
date: 2018-01-02 19:08:03
tags: [CNN, Deep learning]
---


本文涉及以下CNN经典模型，LeNet、AlexNet、VGGNet、GoogleNet、ResNet。下一篇涉及RCNN、Fast R-CNN、Faster R-CNN、YOLO。

<!--more-->

- [LeNet - Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (1998)
- [AlexNet - Imagenet classification with deep convolutional neural networks](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf) (2012)
- [VGGNet - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556) (2014)
- [GoogleNet - Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) (2014)
- [ResNet - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2015)


本文要介绍的几个模型都是ILSVRC竞赛历年的佼佼者，这里先来总览比较AlexNet、VGG、GoogLeNet、ResNet四个模型。见下表：

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

![Arch of LeNet-5](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FLeNet.png@!laphiler)

## AlexNet
2012年ImageNet比赛冠军的model，AlexNet以第一作者Alex命名。[cafe的网络配置在此处](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt)。
废话不说，先上一张经典的AlexNet网络结构的半截图（仔细看下，图的上半部分好像被截断了）。

![Arch of AlexNet](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FAlexNet.png@!laphiler)
AlexNet以Top-5的错误率为16.4%赢得ILSVRC 2012年的比赛。它做出了如下创新：

- 首次使用ReLU作为CNN激活函数，解决了Sigmod激活函数的梯度弥散问题，关于激活函数可参考[上一篇激活函数部分](http://www.laphiler.com/2017/12/20/CNN_startup/)
- 使用Dropout随机丢弃部分神经元，可以避免模型的过拟合。AlexNet的最后几个全连接层使用了Dropout。
- 使用重叠的最大池化，之前使用的都是平均池化。最大池化可以避免平均池化的模糊效果。同时，步长比卷积核的尺寸小，这样池化层的输出之间会有重叠，提升了特征的丰富性。
- 提出LRN层（局部相应一体化），对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对较大，并抑制其他反馈较小的神经元，增强模型泛化能力。
- 使用CUDA加速深度卷积网络的训练，使用了GPU的并行计算能力。
- 数据增强，随机从256x256的原始图中截取224x224大小的区域，再做水平翻转，相当于增加了 (256−224)2×2=2048(256−224)2×2=2048 倍的数据量。仅靠原始的数据量，参数众多的CNN会陷入过拟合。预测时，取图片四个角和中间共5个位置，再加上翻转，共10个位置，对它们的预测结果求均值。

**AlexNet结构图**（换个视角）
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FAlexNet_arch.png@!laphiler)
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

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FVGGNet.jpg@!laphiler)

**VGGNet特性**

- 虽然从A到E每一级网络逐渐变深，但是网络的参数量并没有增长很多，这是因为参数量主要都消耗在最后3个全连接层。前面的卷积部分虽然很深，但是消耗的参数量不大。
- D、E也就是我们常说的VGGNet-16和VGGNet-19。C很有意思，相比B多了几个1x1的卷积层，1x1卷积的意义主要在于线性变换，而输入通道数和输出通道数不变，没有发生降维。
- 两个3x3的卷积层串联相当于1个5x5的卷积层，即一个像素会跟周围5x5的像素产生关联，可以说感受野大小为5x5。而3个3x3的卷积层串联的效果则相当于1个7x7的卷积层。除此之外，3个串联的3x3的卷积层，拥有比1个7x7的卷积层更少的参数量，只有后者的(3x3x3)/(7x7)=55%。最重要的是，3个3x3的卷积层拥有比1个7x7的卷积层更多的非线性变换（前者可以使用三次ReLU激活函数，而后者只有一次），使得CNN对特征的学习能力更强。
- 先训练级别A的简单网络，再复用A网络的权重来初始化后面的几个复杂模型，这样训练收敛的速度更快。

**与AlexNet比较**
比AlexNet收敛的要快一些，原因为：（1）使用小卷积核和更深的网络进行的正则化；（2）在特定的层使用了预训练得到的数据进行参数的初始化。
对于较浅的网络，如网络A，可以直接使用随机数进行随机初始化，而对于比较深的网络，则使用前面已经训练好的较浅的网络中的参数值对其前几层的卷积层和最后的全连接层进行初始化。

## GoogLeNet
[caffe的GoogleNet网络配置](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt)，直观上看GoogleNet是非常深的神经网络模型，本文介绍关于GoogleNet的主要贡献：

1. 提出Inception Architecture并对其优化
2. 取消全连层
3. 运用auxiliary classifiers加速网络converge
4. 提出Batch Normalization

**Inception Architecture**

作者发现传统提高网络精度或性能的方法是一条邪路（P.S.传统方法指的是**扩大网络模型**或**增大训练数据集**），而想从本质上提高网络性能，就得用sparsely connected architectures，即“稀疏连接结构”。
对IA，可以理解为用尽可能的“小”、“分散”的可堆叠的网络结构，去学习复杂的分类任务，如下图：
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FInception.png@!laphiler)
原来造神经网络，都是一条线下来，我们可以回想一下AlexNet、VGG等著名网络，而IA是“分叉-汇聚”型网络，也就是说在一层网络中存在多个不同尺度的kernels，卷积完毕后再汇聚，为了更好理解，“汇聚”的tensorflow代码写出来是这样的：
	
	net = tf.concat(3, [branch1x1, branch5x5, branch3x3, branch_pool])

这种网络结构会带来**参数爆炸**问题，所以在原机构基础上加入了kernels数量控制方式，就是那些1×1的卷积层，如下图：
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FInception2.png@!laphiler)

> IA之所以能提高网络精度，可能就是归功于它拥有多个不同尺度的kernels，每一个尺度的kernel会学习不同的特征，把这些不同kernels学习到的特征汇聚给下一层，能够更好的实现全方位的深度学习！

**取消FC全连层**

为什么VGG网络的参数那么多？就是因为它在最后有两个4096的全连层！Szegedy吸取了教训，为了压缩GoogLeNet的网络参数，他把全连层取消了！
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FGoogleNet_FC.png@!laphiler)

从上图就可以看出，网络的最后几层是avg pool、dropout、linear和softmax，没有看到fully connect的影子。现在取消全连层貌似是个大趋势，近两年的优秀大型神经网络都没有全连层，可能是全连层参数太多，网络深度增加了以后，难以接受吧

**Auxiliary classifiers**

梯度消散是所有深层网络的通病，往往训练到最后，网络最开始的几层就“训不动了”！于是Szegedy加入了auxiliary classifiers（简称AC），用于辅助训练，加速网络converge，如下图画红框部分：
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FGoogleNet_arch.jpeg@!laphiler)
可以看到，作者在网络中间层加入了两个AC，这两个AC在训练的时候也跟着学习，同时把自己学习到的梯度反馈给网络，算上网络最后一层的梯度反馈，GoogleNet一共有3个“梯度提供商”，先不说这么做有没有问题，它确实提高了网络收敛的速度，因为梯度大了嘛。另外，GoogleNet在做inference的时候AC是要被摘掉的。

**Batch Normalization**

BN有很多神奇的特性，比如BN可以带来如下好处

- 选择比较大的初始学习率，也可以选择较小的学习率一样都能得到比较快的收敛速度；
- 忽略drop out、L2正则参数，采用BN算法后，你可以移除这两项参数。
- 不需要LRN归一化层（局部响应归一化层由AlexNet中首次提出），因为BN本身就是一个归一化网络层。

归一化的好处有哪些呢？一方面，神经网络学习的**本质**在于学习数据的分布，一旦训练数据分布和测试数据分布不同，那么网络的繁华能力大大降低；另外，我们知道每批训练数据的分布各不相同，网络的每次迭代都需要学习适应不同的分布，无形中影响了训练速度。

**BN的本质原理**就是：在网络的每一层输入的时候，又插入了一个归一化层，也就是先做一个归一化处理，然后再进入网络的下一层。比如网络第三层输入数据X3(X3表示网络第三层的输入数据)把它归一化至：均值0、方差为1，然后再输入第三层计算，这样我们就可以解决前面所提到的“Internal Covariate Shift”的问题了。[keras代码实现：](http://keras-cn.readthedocs.io/en/latest/)

	m = K.mean(X, axis=-1, keepdims=True)#计算均值  
	std = K.std(X, axis=-1, keepdims=True)#计算标准差  
	X_normed = (X - m) / (std + self.epsilon)#归一化  
	out = self.gamma * X_normed + self.beta#重构变换  

## ResNet
ResNet在2015年大放异彩，在ImageNet的classification、detection、localization以及COCO的detection和segmentation上均斩获了第一名的成绩。

ResNet最根本的动机就是解决所谓的“退化”问题，即当模型的层次加深时，错误率却提高了，如下图：
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FResNet_plain.png@!laphiler)
我们知道，在计算机视觉里，特征的“等级”随增网络深度的加深而变高，研究表明，网络的深度是实现好的效果的重要因素。然而梯度弥散/爆炸成为训练深层次的网络的障碍，导致无法收敛。
有一些方法可以弥补，如归一初始化，各层输入归一化，使得可以收敛的网络的深度提升为原来的十倍。然而，虽然收敛了，但网络却开始退化了，即增加网络层数却导致更大的误差。

而这个“退化”问题产生的原因归结于优化难题，当模型变复杂时，SGD的优化变得更加困难，导致了模型达不到好的学习效果。针对这个问题，作者提出了一个**残差块**(Residual Unit)的结构：
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FResNet_block.png@!laphiler)
的确，通过在一个浅层网络基础上叠加y=x的层（称identity mappings，恒等映射），可以让网络随深度增加而不退化。这反映了多层非线性网络无法逼近恒等映射网络。

但是，不退化不是我们的目的，我们希望有更好性能的网络。ResNet学习的是残差函数F(x) = H(x) - x, 这里如果F(x) = 0, 那么就是上面提到的恒等映射。事实上，ResNet是“shortcut connections”的在connections是在恒等映射下的特殊情况，它没有引入额外的参数和计算复杂度。 假如优化目标函数是逼近一个恒等映射, 而不是0映射， 那么学习找到对恒等映射的扰动会比重新学习一个映射函数要容易。ResNet相当于将学习目标改变了，不再是学习一个完整的输出H(x)，只是输出和输入的差别H(x)−x即残差。

传统卷积层或全连接层在信息传递时，或多或少会存在信息丢失、损耗等问题。ResNet在某种程度上解决了这个问题，通过直接将输入信息绕道到输出，保护信息的完整性，整个网络则只需要学习输入、输出差别的那一部分，简化学习目标和难度。

[参考](https://www.jianshu.com/p/e502e4b43e6d)Ryan Dahl的[tensorflow-resnet](https://github.com/ry/tensorflow-resnet)程序源码，按照Ryan Dahl实现的ResNet，画出了残差块内部网络的具体实现，这个是全网络中第一个残差块的前三层，输入的image大小为[batch_size,56,56,64]，输出大小为[batch_size,56,56,256]，如下图
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FResNet_RU.png@!laphiler)

##### 由于篇幅问题，有关[RCNN、Fast R-CNN、Faster R-CNN、YOLO的几个模型另开一篇](http://www.laphiler.com/2018/01/08/cnn-classic-model-objectdetection/)。




