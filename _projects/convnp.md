---
layout: page
title: Inpainting satellite images
description: Convolutional Neural Processes for Inpainting Satellite Images
img: assets/img/inpaint/landsat7.png
importance: 1
category: machine learning
---

Over most of the last year, I have led a team of other Imperial students on this research project, which takes a new approach to inpainting satellite images. It was first proposed by <a href="https://harrisonzhu508.github.io/">Harison Zhu</a> and I found out about it through the Imperial College Data Science Society (ICDSS). The first few months involved learning about the problem, collecting the satellite images and learning about the architecture of neural processes/meta-learning framework.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/inpaint/inpainted.png" title="team pic" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Inpainting scanlines of LANDSAT 7 satellite image
</div>

Inpainting is a computer vision problem involving filling in missing data like the pixels in a frame. This has seen massive improvements in recent years using deep learning techniques. We take an interesting approach to inpaint in a meta-learning setting. The architecture we use is called a neural process and the idea is that it treats each satellite image as its own regression task or function. The neural network is able to share information of the land over time as it is trained on these tasks and then can do something similar to "few-shot" learning, where it takes in a new set of pixels of the current satellite image and predicts the missing pixels. Do check out <a href="https://yanndubs.github.io/Neural-Process-Family">Yann Dubois' website</a>, which gives an in-depth intro to this architecture.

**Abstract**
The widespread availability of satellite images has allowed researchers to model complex systems such as disease dynamics. However, many satellite images have missing values due to measurement defects, which render them unusable without data imputation. For example, the scanline corrector for the LANDSAT 7 satellite broke down in 2003, resulting in a loss of around 20\% of its data. Inpainting involves predicting what is missing based on the known pixels and is an old problem in image processing, classically based on PDEs or interpolation methods, but recent deep learning approaches have shown promise. However, many of these methods do not explicitly take into account the inherent spatiotemporal structure of satellite images. In this work, we cast satellite image inpainting as a natural meta-learning problem, and propose using convolutional neural processes (ConvNPs) where we frame each satellite image as its own task or 2D regression problem. We show ConvNPs can outperform classical methods and state-of-the-art deep learning inpainting models on a scanline inpainting problem for LANDSAT 7 satellite images, assessed on a variety of in and out-of-distribution images.

See <a href="/assets/pdf/CSML_LANDSAT7_Inpainting.pdf">slides</a> of a recent talk I presented and our <a href="https://arxiv.org/abs/2205.12407">paper under review</a> to get further insight.

The current goal is to get our paper accepted to an ML conference. Feel free to reach out for insights or questions!
