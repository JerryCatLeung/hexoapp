---
title: 交通灯检测
date: 2018-02-24 17:09:30
tags: [CNN, Caffe]
---

### 介绍

Apollo 2.0提供基于camera的交通灯检测Traffic Light Perception，以下简称TLP。

一般的交通灯有三种状态：

* Red
* Yellow
* Green

<!--more-->

在现实场景中，有可能有的交通灯坏掉或者因为视角的原因，导致camera不能捕捉到交通灯的状态，所以TLP模块输出5种状态：

* Red
* Yellow
* Green
* Black 
* Unknown

TLP模块通过提供车辆当前location，在高清地图上查询车辆前方的交通灯位置信息，交通灯位置在高清地图上用4个点来表示出boundary。Apollo 2.0使用25mm长焦和6mm广角两个camera来增强识别的范围，如下图，上一张图是长焦camera采集的图片，下一张图是广角camera采集的图片。

![](https://github.com/ApolloAuto/apollo/raw/master/docs/specs/images/traffic_light/long.jpg)

![](https://github.com/ApolloAuto/apollo/raw/master/docs/specs/images/traffic_light/short.jpg)

### 感知流程

整体交通灯感知分为两个部分。

**预处理模块**

* Traffic light projection
* Camera selection
* Image and cached lights sync

**处理模块**

* Rectify — Provide the accurate traffic light bounding boxes
* Recognize — Provide the color of each bounding box
* Revise — Correct the color based on the time sequence

#### 预处理模块

现实场景中，交通灯的变化频率较低，所以为了节约计算资源，没有必要对图像的每一帧都检测。同时，两个camera检测到的图片几乎是同步的，预处理模块要做的是同一个时刻只需要合理选取一个图片进入预处理流程。

入口函数位置：[apollo/modules/perception/traffic_light/onboard/tl_preprocessor_subnode.cc](https://github.com/ApolloAuto/apollo/blob/9257dcf33e97a91a0e92922ab842dfff5c23e5f3/modules/perception/traffic_light/onboard/tl_preprocessor_subnode.cc)

	bool TLPreprocessorSubnode::InitInternal()

##### 预处理模块的Input/Output

预处理模块接受4路输入，Localization，高精地图，标定结果。

**Input**

- camera采集的图片
	-  /apollo/sensor/camera/traffic/image_long
	-  /apollo/sensor/camera/traffic/image_short

- Localization
	- ／tf
- 高精地图
- 标定结果

**Output**

- 从camera选定的图片
- 从世界坐标系到图片坐标中的交通灯边框

预处理模块的核心处理函数``void TLPreprocessorSubnode::SubCameraImage``，根据代码注释，分成3步对该模块进行描述。

**1. camera选择( which camera should be used )**

SubCameraImage主函数从25mm和6mm camera获得image信息并创建文件保存，接下来选择一个camera做后续操作(默认选长焦相机)。交通灯在3D世界坐标系统中通过一个唯一ID和4个边框点标记

	signal info:
	id {
	  id: "xxx"
	}
	boundary {
	  point { x: ...  y: ...  z: ...  }
	  point { x: ...  y: ...  z: ...  }
	  point { x: ...  y: ...  z: ...  }
	  point { x: ...  y: ...  z: ...  }
	}

选出来的camera的数据结构，会被缓存在queue中

	struct ImageLights {
	 CarPose pose;
	 CameraId camera_id;
	 double timestamp;
	 size_t num_signal;
	 ... other ...
	};

**2. image同步和发布( sync image and publish data )**

**3. 验证projection( verify lights projection based on image time )**

#### 处理模块

##### 处理模块的Input/Output

处理模块的输入是预处理模块的输出，处理模块将输出结果发布到TLP的topic中，供后续流程使用。

**Input**

- 从camera选定的图片
- 交通灯边框集合

**Output**

- 带有颜色标注的交通灯边框集合

##### 边框校正Rectify

交通灯识别的实现使用SSD模型，模型输入是ROI区域，输出是一个或多个交通灯的边框，如果CNN模型不能从ROI中检测到交通灯，那么后两步处理（Recognize，Revise）将会跳过。

##### 交通灯识别Recognize

交通灯识别使用典型的CNN分类任务，输入为交通灯边框集合，输出$4\times n$的vector，表示black, red, yellow, green的可能性。

##### 交通灯颜色修正Revise

当交通灯被识别为black或unknown时，Revise部分会通过查询保存的序列地图信息，来确认交通灯的颜色。

**参考来源**

[1] [Traffic Light Perception](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/traffic_light.md)

[2] [PPT资料 | Apollo 自动驾驶感知技术
](https://mp.weixin.qq.com/s/IIRQoAnEVgTbcmpTI2jo4g)


