---
title: CNN经典神经网络模型 | 目标检测
date: 2018-01-08 17:21:46
tags: [AI, Deep learning, CNN, RCNN, Fast R-CNN, Faster R-CNN, YOLO, Object detection]
---
目标检测是计算机视觉领域一个经典问题，本文主要涉及该领域的经典模型：RCNN, Fast R-CNN, Faster R-CNN, YOLO，SSD。

<!--more-->

- [RCNN - Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) (2014) 
- [Fast R-CNN](https://arxiv.org/abs/1504.08083) (2015)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) (2016)
- [YoLo - You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2016)
- [SSD: Single Shot MultiBox Detector
](https://arxiv.org/abs/1512.02325) (2016)

从下图可以大概看到当前目标检测领域的研究路径，目标分类 -> 目标分类+定位 -> 目标检测 -> 实例分隔，由简到难，由浅到深。当前的目标检测总的发展方向主要就是两条线，一条线是提高检测的精度、另一条线是提高检测的速度（[参考](https://www.zhihu.com/question/34223049/answer/160336559)）。
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fobjectdetection.png@!laphiler)

**传统的目标检测方法**一般分为**三个阶段**（其实DL的方法也大概是这三阶段）：首先在给定的图像上选择一些**候选区域**，然后对这些区域**特征提取**，最后使用训练的分类器进行**目标分类**。

**1.候选区域**：利用不同尺度的滑动窗口在图像上进行区域搜索，定位候选区域。这种策略虽然可以检测到所有可能出现的位置，但是时间复杂度太高，产生的冗余窗口太多，严重影响后续特征的提取和分类速度的性能。
**2.特征提取**：对候选区域进行特征提取，如**SIFT**（尺度不变特征变换 ，Scale-invariant feature transform）和**HOG**（ 方向梯度直方图特征，Histogram of Oriented Gradient）等。
**3.目标分类**：利用分类器进行分类识别，SVM，Adaboost等

## 传统方法-DPM

**DPM算法思想**：输入一幅图像，对图像提取图像特征，针对某个物件制作出相应的激励模板，在原始的图像滑动计算，得到该激励效果图，根据激励的分布，确定目标位置。

制作激励模板就相当于人为地设计一个卷积核，一个比较复杂的卷积核，拿这个卷积核与原图像进行卷积运算得到一幅特征图。比如拿一个静止站立的人的HOG特征形成的卷积核，与原图像的梯度图像进行一个卷积运算，那么目标区域就会被加密。如下图所示：
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fdpm1.png@!laphiler)

由于目标可能会形变，之前模型不能很好的适应复杂场景的探测，所以一个改进是各个部分单独考虑，对物体的不同部分单独进行学习，所以DPM把物体看成了多个组成部件(比如说人脸的鼻子，嘴巴等)，用部件间的关系来描述物体，这个特点非常符合自然界许多物体的非刚性特征。基本思路如下:

1. 产生多个模板，整体模板(root filter)以及不同的局部模板(root filter)；
2. 拿这些不同的模板同输入图像“卷积”产生特征图；
3. 将这些特征图组合形成融合特征；
4. 对融合特征进行传统分类，回归得到目标位置。
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fdpm2.jpg@!laphiler)

**DPM算法缺点**：

1. 性能一般
2. 激励特征人为设计，工作量大；这种方法不具有普适性，因为用来检测人的激励模板不能拿去检测小猫或者小狗，所以在每做一种物件的探测的时候，都需要人工来设计激励模板，为了获得比较好的探测效果，需要花大量时间去做一些设计，工作量很大。
3. 无法适应大幅度的旋转，稳定性很差；

**总结：**
传统目标检测存在的两个主要问题：一个是基于滑动窗口的区域选择策略没有针对性，时间复杂度高，窗口冗余；二是手工设计的特征对于多样性的变化并没有很好的鲁棒性。

先用[一张图概览](http://shartoo.github.io/RCNN-series/)R-CNN系列模型
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Frcnn-all.png@!laphiler)

## R-CNN (CVPR2014, TPAMI2015)
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Frcnn-arch.jpg@!laphiler)
### 训练流程：

1. 输入测试图像
2. 利用selective search算法在图像中提取1-2k个左右的region proposal（候选区）。
3. 将每个region proposal缩放（warp）成227×227的大小并输入到CNN，将CNN的fc7层的输出作为特征。
4. 将每个region proposal提取到的CNN特征输入到SVM进行分类。
5. 边框回归（bounding-box regression)，边框回归是对region proposal进行纠正的线性回归算法，为了让region proposal提取到的窗口跟目标真实窗口更吻合。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Frcnn-mAP.jpg@!laphiler)

### 评估标准

- mAP(mean average precision)
  **precision** = （你选对了的）/（总共选的）
  **recall** =（你能包住的正样本数）/（总共的正样本数）
  ![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FmAP.png@!laphiler)
  
- IoU，真值和预测值的重叠部分与两者的并集的比值，一般大于 0.5 就认为是正确的
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FIoU.jpg@!laphiler)

### 小结
R-CNN在PASCAL VOC2007上的检测结果从DPM HSC的34.3%直接提升到了66%(mAP)。如此大的提升使我们看到了region proposal+CNN的巨大优势。 但是R-CNN框架也存在着很多问题:

1. 训练分为多个阶段，步骤繁琐: 微调网络+训练SVM+训练边框回归器
2. 训练耗时，占用磁盘空间大：5000张图像产生几百G的特征文件
3. 速度慢: 使用GPU, VGG16模型处理一张图像需要47s。








