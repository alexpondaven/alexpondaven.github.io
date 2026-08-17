---
layout: page
permalink: /chase/
title: Chase
description: A hunter with no AI — it plans by imagining futures in a world model
nav: false
nav_order: 7
---
<!-- _pages/chase.md -->

Collect the orbs. Don't get caught. The red hunter chasing you has **no scripted AI at all**: every half-second it *imagines* nine possible futures — rolling candidate moves through the same two tiny neural nets that run this page's physics — scores them by whether they end with you caught, and takes the best one. That's model-predictive control, live in your browser, and **the glowing threads are its actual imagination**: every future it considered, the brightest being the one it chose.

You're faster than it is. It can only win by *prediction* — it chases where the model says you're going, not where you are. Cut corners, feint, use your speed. And you can borrow the same crystal ball: **hold Shift** (or the Premonition button) to see the model's dream of your own next second.

As on [/arena/](/arena/), there is no physics engine here — the ball you drive, the hunter, every wall bounce is the output of two ~2,000-parameter MLPs trained offline by hand-rolled backprop, with learned motion gates holding still things still ([chase_train.js](/assets/worldmodel/chase_train.js), no ML libraries). The hunter's cunning is not in the nets: it emerges from *searching* them.

{% include chase.html %}
