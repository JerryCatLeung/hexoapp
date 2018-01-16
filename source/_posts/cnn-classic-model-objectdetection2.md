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



