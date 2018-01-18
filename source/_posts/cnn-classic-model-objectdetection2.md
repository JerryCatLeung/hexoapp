---
title: CNN经典神经网络模型 | 目标检测(续)
date: 2018-01-15 21:44:09
tags: [AI, Deep learning, CNN, YoLo, SSD, Object detection]
---
目标检测是计算机视觉领域一个经典问题，[上一篇介绍了基于region proposal的深度学习系列模型](http://www.laphiler.com/2018/01/08/cnn-classic-model-objectdetection/)，本篇介绍基于回归方法的深度学习目标检测模型YoLo，YoLov2，SSD。

<!-- more -->

paper:

- [YoLo - You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
- [YoLov2 - Fast YOLO: A Fast You Only Look Once System for Real-time Embedded Object Detection in Video
](https://arxiv.org/abs/1709.05943)
- [YOLO9000 - YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
- [SSD: Single Shot MultiBox Detector
](https://arxiv.org/abs/1512.02325) (2016)

code:

- [caffe实现：caffe-yolo9000](https://github.com/choasUp/caffe-yolo9000)
- [caffe实现：ssd](https://github.com/weiliu89/caffe/tree/ssd)

## YoLo

### 创新

YOLO将物体检测作为回归问题求解。基于一个单独的end-to-end网络，完成从原始图像的输入到物体位置和类别的输出。从网络设计上，YOLO与rcnn、fast rcnn及faster rcnn的区别如下：

- **YOLO训练和检测均是在一个单独网络中进行，实现真正意义上的端到端的模型。**YOLO没有显示地求取region proposal的过程。而rcnn/fast rcnn 采用分离的模块（独立于网络之外的selective search方法）求取候选框（可能会包含物体的矩形区域），训练过程因此也是分成多个模块进行。Faster rcnn使用RPN（region proposal network）卷积网络替代rcnn/fast rcnn的selective
search模块，将RPN集成到fast rcnn检测网络中，得到一个统一的检测网络。尽管RPN与fast rcnn共享卷积层，但是在模型训练过程中，需要反复训练RPN网络和fast rcnn网络（注意这两个网络核心卷积层是参数共享的）

- **YOLO将物体检测作为一个回归问题进行求解**，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。而rcnn/fast rcnn/faster rcnn将检测结果分为两部分求解：物体类别（分类问题），物体位置即bounding box（回归问题）。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fmodel_compare.png@!laphiler)

### 网络结构

- YOLO网络借鉴了GoogLeNet分类网络结构。不同的是，YOLO未使用Inception Module，而是使用1x1卷积层（此处1x1卷积层的存在是为了跨通道信息整合）+ 3x3卷积层简单替代。
- Fast YOLO使用9个卷积层代替YOLO的24个，网络更轻快，速度从YOLO的45fps提升到155fps，但同时损失了检测准确率。
- 使用全图作为 Context 信息，背景错误（把背景错认为物体）比较少。

**流程**

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FYoLo_detection_sys.png@!laphiler)

1. Resize成448x448，图片分割得到7x7网格(cell)
2. CNN提取特征和预测：卷积不忿负责提特征。全链接部分负责预测：
 - a) 7x7x2=98个bounding box(bbox) 的坐标x_{center},y_{center},w,h 以及是否有物体的conﬁdence共5个值。
 - b) 7x7=49个cell所属20个物体的概率。
3. 过滤bbox（通过nms）

YOLO检测网络包括24个卷积层和2个全连接层，如下图所示。其中，**卷积层用来提取图像特征**，**全连接层用来预测图像位置和类别概率值**。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FYoLov1-arch.png@!laphiler)

### 训练

1. **预训练分类网络**： 在 ImageNet 1000-class competition dataset上预训练一个分类网络，这个网络是上图中的前20个卷机网络+average-pooling layer+ fully connected layer （此时网络输入是224x224）。
2. 训练检测网络：转换模型去执行检测任务，《Object detection networks on convolutional feature maps》提到说在预训练网络中增加卷积和全链接层可以改善性能。在他们例子基础上添加4个卷积层和2个全链接层，随机初始化权重。检测要求细粒度的视觉信息，所以把网络输入也又224x224变成448x448。见上图。 

#### 具体过程

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FYoLo_inference.png@!laphiler)

[详细过程参考Google doc](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_4011)

- 一幅图片分成 {% math %} 7*7 {% endmath %} 个网格(grid cell)，某个物体的中心落在这个网格中此网格就负责预测这个物体。
- 最后一层输出为 {% math %} (7 * 7) * 30 {% endmath %} 的维度。每个 {% math %} 1 * 1 * 30 {% endmath %} 的维度对应原图 {% math %} 7 * 7 {% endmath %} 个cell中的一个，{% math %} 1 * 1 * 30 {% endmath %} 中含有类别预测和bbox坐标预测。
	
	1. a) 每个网格（ {% math %} 1 * 1 * 30 {% endmath %} 维度对应原图中的cell）要预测**2个bounding box** （下图中**黄色实线框**）的坐标 {% math %} （x_{center},y_{center},w,h）{% endmath %} ，其中：中心坐标的 {% math %} x_{center},y_{center} {% endmath %} 相对于对应的网格归一化到0-1之间，w,h用图像的width和height归一化到0-1之间。 每个bounding box除了要回归自身的位置之外，还要附带预测一个 {% math %} confidence {% endmath %} 值。 这个**confidence**代表了所预测的box中**含有object的置信度**和**这个box预测的有多准两重信息**： {% math %} confidence = Pr(Object) * IOU^{truth}_{pred} {% endmath %} 其中如果有ground true box(人工标记的物体)落在一个grid cell里，第一项取1，否则取0。 第二项是预测的bounding box和实际的ground truth box之间的IoU值。即：每个bounding box要预测 {% math %} x_{center},y_{center},w,h,confidence {% endmath %} 共5个值 ，2个bounding box共10个值，对应 {% math %} 1 * 1 * 30 {% endmath %} 维度特征中的前10个。

	![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FYoLo_2_boxes.png@!laphiler)
	
	2. b) 每个网格还要预测**类别信息**，论文中有20类。{% math %} 7 * 7 {% endmath %} 的网格，每个网格要预测2个 bounding box 和 20个类别概率，输出就是 {% math %} 7 * 7 * (5 * 2 + 20) {% endmath %} 。 (通用公式： SxS个网格，每个网格要预测B个bounding box还要预测C个categories，输出就是{% math %} S * S * (5 * B + C) {% endmath %} 的一个tensor。 注意：class信息是针对每个网格的，confidence信息是针对每个bounding box的）

#### 损失函数

损失函数的定义如下，损失函数的设计目标就是让坐标，置信度和类别这个三个方面达到很好的平衡。简单的全部采用了Sum-Squared Error Loss来做这件事会有以下不足：① 8维的Localization Error和20维的Classification Error同等重要显然是不合理的；② 如果一个网格中没有Object（一幅图中这种网格很多），那么就会将这些网格中的Box的Confidence Push到0，相比于较少的有Object的网格，这种做法是Overpowering的，这会导致网络不稳定甚至发散。 解决方案如下。[参考原文](http://lanbing510.info/2017/08/28/YOLO-SSD.html)

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FYoLo-loss.png@!laphiler)

1. 更重视8维的坐标预测，给这些损失前面赋予更大的Loss Weight, 记为 {% math %} λ_{coord} {% endmath %} ,在Pascal VOC训练中取5。（上图蓝色框）。
2. 对没有Object的Bbox的Confidence Loss，赋予小的Loss Weight，记为 
{% math %} λ_{noobj} {% endmath %} 在Pascal VOC训练中取0.5上图橙色框）。
3. 有Object的Bbox的Confidence Loss（上图红色框）和类别的Loss（上图紫色框）的Loss Weight正常取1。
4. 对不同大小的Bbox预测中，相比于大Bbox预测偏一点，小Bbox预测偏一点更不能忍受。而Sum-Square Error Loss中对同样的偏移Loss是一样。为了缓和这个问题，将Bbox的Width和Height取平方根代替原本的Height和Width。 如下图：Small Bbox的横轴值较小，发生偏移时，反应到y轴上的Loss（下图绿色）比Big Bbox（下图红色）要大。
5. 一个网格预测多个Bbox，在训练时我们希望每个Object（Ground True box）只有一个Bbox专门负责（一个Object 一个Bbox）。具体做法是与Ground True Box（Object）的IOU最大的Bbox 负责该Ground True Box（Object）的预测。这种做法称作Bbox Predictor的Specialization（专职化）。每个预测器会对特定（Sizes,Aspect Ratio or Classed of Object）的Ground True Box预测的越来越好。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FYoLo-loss-1.png@!laphiler)

### 测试
计算每个Bbox的Class-Specific Confidence Score：每个网格预测的Class信息(
{% math %} Pr(Class_i|Object)) {% endmath %} 和Bbox预测的Confidence信息 {% math %} (Pr(Object) ∗ IOU^{truth}_{pred}) {% endmath %} 相乘，就得到每个Bbox的Class-Specific Confidence Score。
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FYoLo-test-1.png@!laphiler)

2 进行Non-Maximum Suppression（NMS）：得到每个Bbox的Class-Specific Confidence Score以后，设置阈值，滤掉得分低的Bboxes，对保留的Bboxes进行NMS处理，就得到最终的检测结果。更为直观详细的流程可参见[5]。
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FYoLo-test-2.png@!laphiler)


### 缺点

- YOLO对相互靠的很近的物体（挨在一起且中点都落在同一个格子上的情况），还有很小的群体 检测效果不好，这是因为一个网格中只预测了两个框，并且只属于一类。
- 测试图像中，当同一类物体出现的不常见的长宽比和其他情况时泛化能力偏弱。
- 由于损失函数的问题，定位误差是影响检测效果的主要原因，尤其是大小物体的处理上，还有待加强。

## SSD

上半部分介绍的YoLo有一些缺陷：每个网格只预测一个物体，容易造成漏检；对于物体的尺度相对比较敏感，对于尺度变化较大的物体泛化能力较差。针对这两个方面SSD都有所改进，同时兼顾了 mAP 和实时性的要求（见本文开始的图1，几个模型对比）。在满足实时性的条件下，接近state of art的结果。作者的思路就是Faster R-CNN+YOLO，利用YOLO的思路和Faster R-CNN的anchor box的思想。[参考](http://blog.csdn.net/u010167269/article/details/52563573)

### 网络结构

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2Fssd-model.png@!laphiler)

该论文采用 VGG16 的基础网络结构(作者用的是D这个模型进行修改的)，使用前面的前 5 层，然后利用 astrous 算法将 fc6 和 fc7 层转化成两个卷积层。再格外增加了 3 个卷积层，和一个 average pool层。不同层次的 feature map 分别用于 default box 的偏移以及不同类别得分的预测（惯用思路：使用通用的结构(如前5个conv等)作为基础网络，然后在这个基础上增加其他的层），最后通过nms得到最终的检测结果。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2FVGG_model.png@!laphiler)

**总结一下**，SSD 模型的最开始部分，本文称作 base network，是用于图像分类的标准架构。在 base network 之后，本文添加了额外辅助的网络结构：

- **Multi-scale feature maps for detection**
在基础网络结构后，添加了额外的卷积层，这些卷积层的大小是逐层递减的，可以在多尺度下进行 predictions。
- **Convolutional predictors for detection**
每一个添加的特征层（或者在基础网络结构中的特征层），可以使用一系列 convolutional filters，去产生一系列固定大小的 **predictions**，具体见 Fig.2。对于一个大小为 {% math %} m×n {% endmath %}，具有 {% math %} p {% endmath %} 通道的feature map层，使用的 convolutional filters 就是 {% math %} 3×3×p {% endmath %} 的 kernels。产生的 predictions，那么就是归属类别的一个得分，要么就是相对于 default box coordinate 的 shape offsets。 
在每一个 {% math %} m×n {% endmath %} 的特征图位置上，使用上面的 3×3 的 kernel，会产生一个输出值。**bounding box offset** 值是输出的 default box 与此时 feature map location 之间的相对距离（YOLO 架构则是用一个全连接层来代替这里的卷积层）。
- **Default boxes and aspect ratios**
每一个 box 相对于与其对应的 feature map cell 的位置是固定的。 在每一个 feature map cell 中，我们要 predict 得到的 box 与 default box 之间的 offsets，以及每一个 box 中包含物体的 score（每一个类别概率都要计算出）。 
因此，对于一个位置上的 {% math %} k {% endmath %} 个boxes 中的每一个 box，我们需要计算出 c 个类，每一个类的 score，还有这个 box 相对于 它的默认 box 的 {% math %} 4 {% endmath %} 个偏移值（offsets）。于是，在 feature map 中的每一个 feature map cell 上，就需要有 {% math %} (c+4)×k {% endmath %} 个 filters。对于一张 {% math %} m×n {% endmath %} 大小的 feature map，即会产生 {% math %} (c+4)×k×m×n {% endmath %} 个输出结果。

#### default boxes & feature map cell

- **feature map cell**就是将**feature map**切分成 8×8 或者 4×4 之后的一个个格子；
- 而**default box**就是每一个格子上，一系列固定大小的 box，即图中虚线所形成的一系列 boxes。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2Fssd_box.png@!laphiler)

#### 模型flow

这些增加的卷积层的 feature map 的大小变化比较大，允许能够检测出不同尺度下的物体： 在低层的feature map,感受野比较小，高层的感受野比较大，在不同的feature map进行卷积，可以达到多尺度的目的。观察YoLo，后面存在两个全连接层，全连接层以后，每一个输出都会观察到整幅图像，并不是很合理。但是SSD去掉了全连接层，每一个输出只会感受到目标周围的信息，包括上下文。这样来做就增加了合理性。并且不同的feature map,预测不同宽高比的图像，这样比YOLO增加了预测更多的比例的box。（下图横向的流程）

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2Fssd-model-detail.png@!laphiler)

### 训练

#### 损失函数

这个与Faster R-CNN中的RPN是一样的，不过RPN是预测box里面有object或者没有，所以，没有分类，SSD直接用的softmax分类。location的损失，还是一样，都是用predict box和default box/Anchor的差 与 ground truth box和default box/Anchor的差 进行对比，求损失。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2Fssd_loss.png@!laphiler)

#### 训练策略

监督学习的训练关键是人工标注的label。对于包含default box(在Faster R-CNN中叫做anchor)的网络模型（如： YOLO,Faster R-CNN, MultiBox）关键点就是如何把 标注信息(ground true box,ground true category)映射到（default box上）

- 正负样本： 给定输入图像以及每个物体的 ground truth,首先找到每个ground true box对应的default box中IOU最大的作为（与该ground true box相关的匹配）正样本。然后，在剩下的default box中找到那些与任意一个ground truth box 的 IOU 大于 0.5的default box作为（与该ground true box相关的匹配）正样本。 一个 ground truth 可能对应多个 正样本default box 而不再像MultiBox那样只取一个IOU最大的default box。其他的作为负样本（每个default box要么是正样本box要么是负样本box）。下图的例子是：给定输入图像及 ground truth，分别在两种不同尺度(feature map 的大小为 8 * 8，4 * 4)下的匹配情况。有两个 default box 与猫匹配（8 * 8），一个 default box 与狗匹配（4 * 4）。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2Fssd-positive-sample.jpg@!laphiler)

该论文是在 ImageNet 分类和定位问题上的已经训练好的 VGG16 模型中 fine-tuning 得到，使用 SGD，初始学习率为 {% math %} 10^{-3} {% endmath %}, 冲量为 0.9，权重衰减为 0.0005，batchsize 为 32。不同数据集的学习率改变策略不同。新增加的卷积网络采用 xavier 的方式进行初始化

- Default Box 的生成：
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_object_detection2%2Fssd-default-box.png@!laphiler)

