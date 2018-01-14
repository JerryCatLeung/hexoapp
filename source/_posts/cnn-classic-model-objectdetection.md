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

1. 输入测试图像；
2. 利用**selective search**算法在图像中提取1-2k个左右的region proposal（候选区）；
3. 将每个region proposal缩放（warp）成**227×227**的固定大小并输入到**CNN(AlexNet Input)**，将CNN的fc7层的输出作为特征，**得到每个region proposal的feature map**；
4. 将每个region proposal提取到的CNN特征输入到SVM进行分类；
5. **边框回归**（bounding-box regression)，边框回归是对region proposal进行纠正的线性回归算法，为了让region proposal提取到的窗口跟目标真实窗口更吻合。

### selective search
**总体思路**:假设现在图像上有n个预分割的区域,表示为R={R1, R2, ..., Rn}, 计算每个region与它相邻region(注意是相邻的区域)的相似度,这样会得到一个n*n的相似度矩阵(同一个区域之间和一个区域与不相邻区域之间的相似度可设为NaN),从矩阵中找出最大相似度值对应的两个区域,将这两个区域合二为一,这时候图像上还剩下n-1个区域; 重复上面的过程(只需要计算新的区域与它相邻区域的新相似度,其他的不用重复计算),重复一次,区域的总数目就少1,知道最后所有的区域都合并称为了同一个区域(即此过程进行了n-1次,区域总数目最后变成了1).算法的流程图如下图。

**合并规则**:

- 颜色（颜色直方图）相近的 
- 纹理（梯度直方图）相近的 
- 合并后总面积小的： 保证合并操作的尺度较为均匀，避免一个大区域陆续“吃掉”其他小区域 （例：设有区域a-b-c-d-e-f-g-h。较好的合并方式是：ab-cd-ef-gh -> abcd-efgh -> abcdefgh。 不好的合并方法是：ab-c-d-e-f-g-h ->abcd-e-f-g-h ->abcdef-gh -> abcdefgh）
- 合并后，总面积在其BBOX中所占比例大的： 保证合并后形状规则。

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fselective_search_ch.png@!laphiler)

### 非极大值抑制（NMS）

RCNN会从一张图片中找出n个可能是物体的矩形框，然后为每个矩形框为做类别分类概率：

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Frcnn-rms.jpg@!laphiler)

就像上面的图片一样，定位一个车辆，最后算法就找出了一堆的方框，我们需要判别哪些矩形框是没用的。非极大值抑制的方法是：先假设有6个矩形框，根据分类器的类别分类概率做排序，假设从小到大属于车辆的概率 分别为A、B、C、D、E、F。

- 从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
- 假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
- 从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。

就这样一直重复，找到所有被保留下来的矩形框。

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

### 训练阶段

- **有监督预训练**： 作者使用caffe框架利用ILSVRC 2012的数据集（ImageNet）对网络模型进行了训练，使网络模型中的参数都是经过训练过的参数，而不是刚开始那样随机初始化的参数；
- **特定领域的fine-tuning**：为了适应不同场合的识别需要，如VOC，对网络继续使用从VOC图片集上对region proposals归一化后的图片进行训练。网络只需要将最后的1000类的分类层换成21类的分类层（20个VOC中的类别和1个背景类），其他都不需要变。为了保证训练只是对网络的微调而不是大幅度的变化，网络的学习率只设置成了0.001。计算每个region proposal与人工标注的框的IoU，IoU重叠阈值设为0.5，大于这个阈值的作为正样本，其他的作为负样本，然后在训练的每一次迭代中都使用32个正样本（包括所有类别）和96个背景样本组成的128张图片的batch进行训练（这么做的主要原因还是正样本图片太少了）；
- **特定类别的分类器**：对每个类都训练一个线性的SVM分类器，训练SVM需要正负样本文件，可以想象得到，刚好包含某一类物体的region proposal应该是正样本，完全不包含的region proposal应该是负样本，但是对于部分包含某一类物体的region proposal该如何训练呢，作者同样是使用IoU阈值的方法，这次的阈值为0.3，计算每一个region proposal与标准框的IoU，大于这个阈值的作为正样本，小于的作为负样本。由于训练样本比较大，作者使用了standard hard negative mining method（具体reference to hard negative mining of my blog）来训练分类器。作者表示在补充材料中讨论了为什么fine-tuning和训练SVM时所用的正负样本标准不一样（0.5和0.3），以及为什么不直接用卷积神经网络的输出来分类而要单独训练SVM来分类(see the folowing bolg, and it will make it done) 。

### 测试阶段

- 使用selective search的方法在测试图片上提取2000个region propasals ，将每个region proposals归一化到227x227，然后再CNN中正向传播，将最后一层得到的特征提取出来。然后对于每一个类别，使用为这一类训练的SVM分类器对提取的特征向量进行打分，得到测试图片中对于所有region proposals的对于这一类的分数，再使用贪心的非极大值抑制去除相交的多余的框。
- 非极大值抑制（NMS）先计算出每一个bounding box的面积，然后根据score进行排序，把score最大的bounding box作为选定的框，计算其余bounding box与当前最大score与box的IoU，去除IoU大于设定的阈值的bounding box。然后重复上面的过程，直至候选bounding box为空，然后再将score小于一定阈值的选定框删除得到一类的结果。作者提到花费在region propasals和提取特征的时间是13s/张-GPU和53s/张-CPU，可以看出时间还是很长的，不能够达到及时性。

### R-CNN小结

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Frcnn-mAP.jpg@!laphiler)

R-CNN在PASCAL VOC2007上的检测结果从DPM HSC的34.3%直接提升到了66%(mAP)。如此大的提升使我们看到了region proposal+CNN的巨大优势。 但是R-CNN框架也存在着很多问题；

因为rcnn首先需要在AlexNet上进行分类的训练model，得到AlexNet之后才能进行分类(Pretrained procedure->SoftMax2SVM)。分类之后在改一下AxlexNet model （fc: 1000->21）得到detection model（training）->(testing)；

然后在上面利用SVM进行二分类判断当前的region有没有包含我们需要的物体(对结果进行排序，取前面的IOU最大的那几个(nms)),在对这些进行canny边缘检测，才可以得到bounding-box(then B-BoxRegression)。总结一下：[参考](http://blog.csdn.net/u011534057/article/category/6178027)

1. 训练分为多个阶段，步骤繁琐: 微调网络+训练SVM+训练边框回归器
2. 训练耗时，占用磁盘空间大：5000张图像产生几百G的特征文件
3. 速度慢: 使用GPU, VGG16模型处理一张图像需要47s。

针对以上问题，RBG又提出Fast R-CNN, 一个精简而快速的目标检测框架。

## Fast R-CNN(ICCV2015)

### Fast R-CNN的三个进步

- 最后一个卷积层后加了一个**ROI**(Regions of Interest) pooling layer。ROI pooling layer首先可以将image中的ROI定位到feature map，然后是用一个单层的SPP layer将这个feature map patch池化为固定大小的feature vector之后再传入全连接层。ROI pooling layer实际上是SPP-NET的一个精简版。
- 损失函数使用了多任务损失函数(multi-task loss)，将边框回归直接加入到CNN网络中训练。
- 将深度网络和后面的SVM分类两个阶段整合到一起，使用一个新的网络直接做分类和回归。用softmax 代替 svm 分类，用多目标损失函数加入候选框回归，除 region proposal 提取外实现了 end-to-end

### ROI Pooling Layer

首先需要介绍RCNN系列里的一个核心算法模块，即**ROI Pooling**(Regions of Interest)。我们知道在ImageNet数据上做图片分类的网络，一般都是先把图片crop、resize到固定的大小（i.e. 224*224），然后输入网络提取特征再进行分类，而对于检测任务这个方法显然并不适合，因为原始图像如果缩小到224这种分辨率，那么感兴趣对象可能都会变的太小无法辨认。RCNN的数据输入和SPPNet有点类似，并不对图片大小限制，而实现这一点的关键所在，就是ROI Pooling网络层，它可以在任意大小的图片feature map上针对输入的每一个ROI区域提取出固定维度的特征表示，保证后续对每个区域的后续分类能够正常进行。如下GIF图是ROI pooling的过程：

![](https://blog.deepsense.ai/wp-content/uploads/2017/02/roi_pooling-1.gif)


### SPP Layer

CNN网络需要固定尺寸的图像输入，SPPNet将任意大小的图像池化生成固定长度的图像表示，提升R-CNN检测的速度24-102倍。[原文参考](http://blog.csdn.net/u011534057/article/details/51219959)
固定图像尺寸输入的问题，截取的区域未涵盖整个目标或者缩放带来图像的扭曲。事实上，CNN的卷积层不需要固定尺寸的图像，全连接层是需要固定大小输入的，因此提出了SPP层放到卷积层的后面，改进后的网络如下图所示：

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Fspp-crop-warp.jpg@!laphiler)

**SPP核心思想** 通过对feature map进行相应尺度的pooling，使得能pooling出4×4, 2×2, 1×1的feature map，再将这些feature map concat成列向量与下一层全链接层相连。这样就消除了输入尺度不一致的影响。


### 多任务损失函数(multi-task loss)

上面已经讲到, Fast R-CNN 的一个优点是 end-to-end 的训练, 这一特点是通过 Multi-task Loss 来实现的.
- 第一个 Loss 是用来训练bounding box 的类别的. 输出是一个离散的概率分布, 输出的节点个数是 {% math %} K+1 {% endmath %}, {% math %} p = (p_0,...,p_k) {% endmath %}，其中{% math %} K {% endmath %} 数据集中的类别数, {% math %} 1 {% endmath %} 是background。
- 第二个 Loss 是用来训练 bounding box regression offset 的, {% math %} t^k = (t_x^k,t_y^k,t_w^k,t_h^k) {% endmath %}。
- 每一个 RoI 都有两个 label, 一个是类别 {% math %} u {% endmath %}, 另外一个是 bounding box regression target {% math %} v {% endmath %}. 

**Multi-task loss** 定义为:
$$L(p,u,t^u,v) = L_{p,u} + \lambda[u≥1]L_loc(t^u,v)$$
其中，{% math %} L_{p,u} = −logp_u {% endmath %}是针对 classify 的loss；
{% math %} L_loc {% endmath %}是定义在一个四元组上面的 bounding box 的损失函数，对于类别 {% math %} u {% endmath %}, 其**ground truth** 的bounding box 为{% math %} v = (v_x,v_y,v_w,v_h) {% endmath %}, 其predict 得到的结果是{% math %} t^u = (t_x^u,t_y^u,t_w^u,t_h^u) {% endmath %}。
针对bounding box regression 的loss 定义是:

{% math %} L_loc(t^u,v)= \sum\limits_{i∈x,y,w,h}smooth_L1 (t_i^u-v_i) {% endmath %}

在这里 

{% math %}  \begin{eqnarray}\label{OQPSK} smooth_{L1}(x) =
\begin{cases}
0.5x^2, &if |x| < 1\cr
|x| - 0.5, &otherwise
\end{cases}
\end{eqnarray}  {% endmath %}
### 模型

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Ffastrcnn-arch.png@!laphiler)

图中省略了通过ss获得proposal的过程，第一张图中红框里的内容即为通过ss提取到的proposal，中间的一块是经过深度卷积之后得到的conv feature map，图中灰色的部分就是我们红框中的proposal对应于conv feature map中的位置，之后对这个特征经过ROI pooling layer处理，之后进行全连接。在这里得到的ROI feature vector最终被分享，一个进行全连接之后用来做softmax回归，用来进行分类，另一个经过全连接之后用来做bbox回归。

注意： 对中间的Conv feature map进行特征提取。每一个区域经过RoI pooling layer和FC layers得到一个 固定长度 的feature vector(这里需要注意的是，输入到后面RoI pooling layer的feature map是在Conv feature map上提取的，故整个特征提取过程，只计算了一次卷积。虽然在最开始也提取出了大量的RoI，但他们还是作为整体输入进卷积网络的，最开始提取出的RoI区域只是为了最后的Bounding box 回归时使用，用来输出原图中的位置)。

#### 位置 + 分类联合学习

图片 => cnn feature map计算 => proposal应用 => feature map相应区域做 region pooling 得到固定大小的 feature map => classification & regression
用 softmax 代替 svm 分类，使用多任务损失函数(multi-task loss)，将候选框回归直接加入到 cnn 网络中训练，除去 region proposal 的提取阶段，这样的训练过程是**端到端**的(end-to-end)，整个网络的训练和测试十分方便

### 性能提升

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Ffastrcnn-perfermence-1.jpg@!laphiler)

![](http://xiaoluban.cdn.bcebos.com/laphiler%2FCNN_classic_model%2Ffastrcnn-per-2.jpg@!laphiler)

上图是不考虑候选区域生成，下图包括候选区域生成，region proposal 的提取使用 selective search，目标检测时间大多消耗在这上面(提region proposal 2~3s，而提特征分类只需0.32s)，无法满足实时应用，那么，怎么解决候选区域的计算呢？一个方法是也靠神经网络。

## Faster R-CNN(NIPS2015)

Faster R-CNN统一的网络结构如下图所示，可以简单看作RPN网络+Fast R-CNN网络。












