---
layout: page
title: Repulsion in Stable Diffusion 2
description: Master's thesis at Imperial College London
img: assets/img/repulsion/thesis_thumbnail.png
importance: 1
category: machine learning
---

For my Master's thesis, I worked on improving the diversity of image samples generated using Stable Diffusion 2 (SD2) by integrating particle based methods into the denoising steps. This method introduces a repulsion force between particles inspired by [SVGD](https://arxiv.org/abs/1608.04471) and augments the denoising steps without requiring additional training of the diffusion model. While these models are often considered quite diverse already as they resolve issues like mode collapse, this method can easily be added just on the sampling method to improve the spread of samples in a certain space. This improves the control and flexibility that users have when exploring the image space.

The samples at each noise level of the diffusion process usually take a denoising step towards the score or towards more likely samples. This method adds repulsion terms between all particles (samples) to reduce redundancy. This repulsion term is equal to the gradient of the kernel similarity of the particles. This intuitively means that we are updating particles in the direction that decreases the similarity between particles as defined by the kernel.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/repulsion/particles.png" title="repulsion of particles" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Repulsion terms added to samples/particles to get a better spread across the distribution at each noise level.
</div>

Applying an RBF kernel directly on the 64x64x4 latent space of SD2 does not lead to very much repulsion due to the high dimensionality of this comparison. We therefore choose to compare two particles in a lower dimensional embedding space instead, such that the kernel similarity compares more semantic, high level information. This introduces a whole design space of embedding models that lead to different types of repulsions in different spaces. Several repulsion methods are introduced that repulse the average pixel intensities of images, the location of features, and the style.

Finally, we introduce several diversity metrics to evaluate the spread of samples generated using these methods and verify that the diversity does indeed improve while maintaining the quality of images. These metrics involve comparing image sets to a "maximum diversity" set, such that closer sets are said to be more diverse.

See <a href="/assets/pdf/masters_presentation.pdf" target="_blank">slides</a> for the final presentation slides with image samples and check out the <a href="https://github.com/alexpondaven/particlediffusion">code repo</a> to experiment with generating repulsed samples. Report will be made public soon.

