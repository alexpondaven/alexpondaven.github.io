---
layout: page
title: Music Genre Classification
description: Predicting the genre of music from raw audio
img: assets/img/genreclassify/freq.jpg
importance: 3
category: machine learning
---

<a href="https://github.com/alexpondaven/Music-Genre-Classification">GitHub</a>

<iframe width="560" height="315" src="https://www.youtube.com/embed/OqOR4L5_XtM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This project involved classifying the genre of piece of music just based on the raw audio. Genres include blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock. This was one of my first projects to help me learn more Tensorflow and get practice manipulating data for machine learning purposes. I analysed the spectra and coefficients called MFCC to understand how the audio differentiates between different genres of music. Different ML algorithms were used like K-Nearest Neighbours and CNNs. The CNN performed the best with 90% classification accuracy on the test set. All the analysis and models can be found in the <a href="https://github.com/alexpondaven/Music-Genre-Classification">GitHub</a>.

It was interesting to apply Convolutional Neural Networks, which are usually applied to image data, to spectrograms of music (plotting frequency against time). This could extract spatial patterns from another type of data and I liked how different ML algorithms from one field can be applied to many different fields.

The CNN model was used to classify music live while it was being played into a mic. The <a href="https://github.com/alexpondaven/Music-Genre-Classification/blob/main/genre_classify_mic.py">script</a> takes the last 3 seconds of recorded audio, runs through the CNN classifier to get a probability distribution over the music genres and plots results on a histogram. It is quite fun to play around with and see how it classifies genre just from a small section of audio, so please check it out!

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/genreclassify/musicbars.png" title="team pic" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Live music genre classifier
</div>

More info can be found on the <a href="https://github.com/alexpondaven/Music-Genre-Classification">GitHub</a>. This project was early on in my ML journey, so it can definitely be improved, but it was an interesting project.


