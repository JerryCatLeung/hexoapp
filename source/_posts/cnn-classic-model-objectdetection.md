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

### DPM算法思想
输入一幅图像，对图像提取图像特征，针对某个物件制作出相应的激励模板，在原始的图像滑动计算，得到该激励效果图，根据激励的分布，确定目标位置。

制作激励模板就相当于人为地设计一个卷积核，一个比较复杂的卷积核，拿这个卷积核与原图像进行卷积运算得到一幅特征图。比如拿一个静止站立的人的HOG特征形成的卷积核，与原图像的梯度图像进行一个卷积运算，那么目标区域就会被加密。如下图所示：
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fdpm1.png@!laphiler)

由于目标可能会形变，之前模型不能很好的适应复杂场景的探测，所以一个改进是各个部分单独考虑，对物体的不同部分单独进行学习，所以DPM把物体看成了多个组成部件(比如说人脸的鼻子，嘴巴等)，用部件间的关系来描述物体，这个特点非常符合自然界许多物体的非刚性特征。基本思路如下:

1. 产生多个模板，整体模板(root filter)以及不同的局部模板(root filter)；
2. 拿这些不同的模板同输入图像“卷积”产生特征图；
3. 将这些特征图组合形成融合特征；
4. 对融合特征进行传统分类，回归得到目标位置。
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fdpm2.jpg@!laphiler)

### DPM算法缺点

1. 性能一般
2. 激励特征人为设计，工作量大；这种方法不具有普适性，因为用来检测人的激励模板不能拿去检测小猫或者小狗，所以在每做一种物件的探测的时候，都需要人工来设计激励模板，为了获得比较好的探测效果，需要花大量时间去做一些设计，工作量很大。
3. 无法适应大幅度的旋转，稳定性很差；

### 总结
传统目标检测存在的两个主要问题：一个是基于滑动窗口的区域选择策略没有针对性，时间复杂度高，窗口冗余；二是手工设计的特征对于多样性的变化并没有很好的鲁棒性。

先用[一张图概览](http://shartoo.github.io/RCNN-series/)R-CNN系列模型
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Frcnn-all.png@!laphiler)

## R-CNN (CVPR2014, TPAMI2015)
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Frcnn-arch.png@!laphiler)
### 训练流程：

1. 输入测试图像
2. 利用**selective search**算法在图像中提取1-2k个左右的region proposal（候选区）。
3. 将每个region proposal缩放（warp）成227×227的大小并输入到CNN，将CNN的fc7层的输出作为特征。
4. 将每个region proposal提取到的CNN特征输入到SVM进行分类。
5. **边框回归**（bounding-box regression)，边框回归是对region proposal进行纠正的线性回归算法，为了让region proposal提取到的窗口跟目标真实窗口更吻合。

### selective search
总体思路:假设现在图像上有n个预分割的区域,表示为R={R1, R2, ..., Rn}, 计算每个region与它相邻region(注意是相邻的区域)的相似度,这样会得到一个n*n的相似度矩阵(同一个区域之间和一个区域与不相邻区域之间的相似度可设为NaN),从矩阵中找出最大相似度值对应的两个区域,将这两个区域合二为一,这时候图像上还剩下n-1个区域; 重复上面的过程(只需要计算新的区域与它相邻区域的新相似度,其他的不用重复计算),重复一次,区域的总数目就少1,知道最后所有的区域都合并称为了同一个区域(即此过程进行了n-1次,区域总数目最后变成了1).算法的流程图如下图所示:

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fselective_search_ch.png@!laphiler)

### 边框回归（bounding-box regression)
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fbboxing-r.png@!laphiler)
对于上图，绿色的框表示Ground Truth, 红色的框为Selective Search提取的Region Proposal。那么即便红色的框被分类器识别为飞机，但是由于红色的框定位不准(IoU<0.5)， 那么这张图相当于没有正确的检测出飞机。 如果我们能对红色的框进行微调， 使得经过微调后的窗口跟Ground Truth 更接近， 这样岂不是定位会更准确。 确实，Bounding-box regression 就是用来微调这个窗口的。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fbboxing-r2.png@!laphiler)
对于窗口一般使用四维向量$$(x,y,w,h)$$来表示， 分别表示窗口的中心点坐标和宽高。 对于图 2, 红色的框 P 代表原始的Proposal, 绿色的框 G 代表目标的 Ground Truth， 我们的目标是寻找一种关系使得输入原始的窗口 P 经过映射得到一个跟真实窗口 G 更接近的回归窗口Ĝ 。
**边框回归的目的既是**：给定(Px,Py,Pw,Ph)寻找一种映射f， 使得$$f(Px,Py,Pw,Ph)=(Gx^,Gy^,Gw^,Gh^)$$ 并且$$(Gx^,Gy^,Gw^,Gh^)≈(Gx,Gy,Gw,Gh)$$
[详细原理参考caffe中文社区](http://caffecn.cn/?/question/160)

### 评估标准

- mAP(mean average precision)
  **precision** = （你选对了的）/（总共选的）
  **recall** =（你能包住的正样本数）/（总共的正样本数）
  ![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FmAP.png@!laphiler)
  
- IoU，真值和预测值的重叠部分与两者的并集的比值，一般大于 0.5 就认为是正确的
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2FIoU.jpg)

### 网络设计
网络架构我们有两个可选方案：第一选择经典的Alexnet；第二选择VGG16。经过测试Alexnet精度为58.5%，VGG16精度为66%。VGG这个模型的特点是选择比较小的卷积核、选择较小的跨步，这个网络的精度高，不过计算量是Alexnet的7倍。后面为了简单起见，我们就直接选用Alexnet，并进行讲解；Alexnet特征提取部分包含了5个卷积层、2个全连接层，在Alexnet中p5层神经元个数为9216、 f6、f7的神经元个数都是4096，通过这个网络训练完毕后，最后提取特征每个输入候选框图片都能得到一个4096维的特征向量。

#### 网络初始化
- 直接用Alexnet的网络，然后连参数也是直接采用它的参数，作为初始的参数值，然后再fine-tuning训练。
- 网络优化求解：采用随机梯度下降法，学习速率大小为0.001；

#### fine-tuning阶段
我们接着采用selective search 搜索出来的候选框，然后处理到指定大小图片，继续对上面预训练的cnn模型进行fine-tuning训练。假设要检测的物体类别有N类，那么我们就需要把上面预训练阶段的CNN模型的最后一层给替换掉，替换成N+1个输出的神经元(加1，表示还有一个背景)，然后这一层直接采用参数随机初始化的方法，其它网络层的参数不变；接着就可以开始继续SGD训练了。开始的时候，SGD学习率选择0.001，在每次训练的时候，我们batch size大小选择128，其中32个正样本、96个负样本。

### 小结
![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Frcnn-mAP.jpg@!laphiler)

R-CNN在PASCAL VOC2007上的检测结果从DPM HSC的34.3%直接提升到了66%(mAP)。如此大的提升使我们看到了region proposal+CNN的巨大优势。 但是R-CNN框架也存在着很多问题:

1. 训练分为多个阶段，步骤繁琐: 微调网络+训练SVM+训练边框回归器
2. 训练耗时，占用磁盘空间大：5000张图像产生几百G的特征文件
3. 速度慢: 使用GPU, VGG16模型处理一张图像需要47s。

针对速度慢的这个问题，SPP-NET给出了很好的解决方案。

## SPP-NET (ECCV2014, TPAMI2015)
(Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition)
**R-CNN**：检测速度慢，对图像提完region proposal（2k左右）之后将每个proposal当成一张图像进行后续处理(CNN提特征+SVM分类)，实际上对一张图像进行了2000次提特征和分类的过程！
对图像提一次卷积层特征，然后将region proposal在原图的位置映射到卷积层特征图上，这样对于一张图像我们只需要提一次卷积层特征，然后将每个region proposal的卷积层特征输入到全连接层做后续操作。
**小结**：使用SPP-NET相比于R-CNN可以大大加快目标检测的速度，但是依然存在着很多问题：

训练分为多个阶段，步骤繁琐: 微调网络+训练SVM+训练边框回归器
SPP-NET在微调网络的时候固定了卷积层，只对全连接层进行微调，而对于一个新的任务，有必要对卷积层也进行微调。（分类的模型提取的特征更注重高层语义，而目标检测任务除了语义信息还需要目标的位置信息）
针对这两个问题，RBG又提出Fast R-CNN, 一个精简而快速的目标检测框架。

## Fast R-CNN(ICCV2015)

### Fast R-CNN的三个进步

- 最后一个卷积层后加了一个ROI pooling layer。ROI pooling layer首先可以将image中的ROI定位到feature map，然后是用一个单层的SPP layer将这个feature map patch池化为固定大小的feature之后再传入全连接层。ROI pooling layer实际上是SPP-NET的一个精简版。
- 损失函数使用了多任务损失函数(multi-task loss)，将边框回归直接加入到CNN网络中训练。
- 将深度网络和后面的SVM分类两个阶段整合到一起，使用一个新的网络直接做分类和回归。用softmax 代替 svm 分类，用多目标损失函数加入候选框回归，除 region proposal 提取外实现了 end-to-end

### 模型

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Ffastrcnn-arch.png@!laphiler)

图中省略了通过ss获得proposal的过程，第一张图中红框里的内容即为通过ss提取到的proposal，中间的一块是经过深度卷积之后得到的conv feature map，图中灰色的部分就是我们红框中的proposal对应于conv feature map中的位置，之后对这个特征经过ROI pooling layer处理，之后进行全连接。在这里得到的ROI feature vector最终被分享，一个进行全连接之后用来做softmax回归，用来进行分类，另一个经过全连接之后用来做bbox回归。

注意： 对中间的Conv feature map进行特征提取。每一个区域经过RoI pooling layer和FC layers得到一个 固定长度 的feature vector(这里需要注意的是，输入到后面RoI pooling layer的feature map是在Conv feature map上提取的，故整个特征提取过程，只计算了一次卷积。虽然在最开始也提取出了大量的RoI，但他们还是作为整体输入进卷积网络的，最开始提取出的RoI区域只是为了最后的Bounding box 回归时使用，用来输出原图中的位置)。

#### 位置 + 类别联合学习

图片 => cnn feature map计算 => proposal应用 => feature map相应区域做 region pooling 得到固定大小的 feature map => classification & regression
用 softmax 代替 svm 分类，使用多任务损失函数(multi-task loss)，将候选框回归直接加入到 cnn 网络中训练，除去 region proposal 的提取阶段，这样的训练过程是**端到端**的(end-to-end)，整个网络的训练和测试十分方便

### 性能提升

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Ffastrcnn-perfermence-1.jpg@!laphiler)

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Ffastrcnn-per-2.jpg@!laphiler)

上图是不考虑候选区域生成，下图包括候选区域生成，region proposal 的提取使用 selective search，目标检测时间大多消耗在这上面(提region proposal 2~3s，而提特征分类只需0.32s)，无法满足实时应用，那么，怎么解决候选区域的计算呢？一个方法是也靠神经网络。

## Faster R-CNN(NIPS2015)

Faster R-CNN统一的网络结构如下图所示，可以简单看作RPN网络+Fast R-CNN网络。












