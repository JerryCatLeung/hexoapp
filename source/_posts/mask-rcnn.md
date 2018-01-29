---
title: Mask R-CNN原理与Mask R-CNN2Go介绍
date: 2018-01-26 17:02:11
tags: [CNN, Mask R-CNN, Mask R-CNN2Go]
---


作为ICCV 2017的best paper，Kaiming He大神所在的[FAIR](https://research.fb.com/category/facebook-ai-research-fair/)又一次在CV领域引领方向，本篇分两部分，主要介绍Mask R-CNN原理，以及Facebook AI Camera Team刚刚(JANUARY 25, 2018)发布的[Mask R-CNN2Go](https://research.fb.com/enabling-full-body-ar-with-mask-r-cnn2go/)。

<!--more-->

**paper:**

- [Mask R-CNN](https://arxiv.org/abs/1703.06870) (2017)
- [Enabling full body AR with Mask R-CNN2Go](https://research.fb.com/enabling-full-body-ar-with-mask-r-cnn2go/) (2018)

**code:**

FAIR开源的项目，实现了一系列state-of-art物体检测算法，以RCNN系列算法为主，RCNN系列算法原理见前几篇文章。

- [GitHub:Detectron](https://github.com/facebookresearch/Detectron)

