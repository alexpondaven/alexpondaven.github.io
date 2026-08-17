---
layout: page
permalink: /playground/
title: Playground
description: small games where everything that moves is a tiny neural net
nav: true
nav_order: 5
---
<!-- _pages/playground.md -->

<script type="importmap">
  { "imports": { "three": "{{ '/assets/js/vendor/three.module.min.js' | relative_url }}" } }
</script>

Everything that moves below is driven by neural nets with a few thousand parameters, trained with [hand-rolled backprop](https://github.com/alexpondaven/alexpondaven.github.io/tree/master/assets/worldmodel) in plain JavaScript. No physics engines, no scripted AI.

#### World model

A ball you can drive (WASD / arrows), simulated entirely by a 2.3k-param net. Toggle ground truth to see how far it drifts from real physics.

{% include worldmodel.html %}

#### Arena

Push crates into the ring. All the physics — collisions, bounces, friction — is two nets. Autopilot is a 460-param policy trained inside the world model.

{% include arena.html skip_map=true %}

#### Colony

Creatures that hatch from the paragraph's letters, eat more letters until they spell a word, then fly. Click the masked one when it steals.

{% include colony.html %}

#### Chase

The red thing has no AI — it plans by imagining futures through the same nets that run the game. The threads are its thoughts. Hold Shift to see yours.

{% include chase.html skip_map=true %}
