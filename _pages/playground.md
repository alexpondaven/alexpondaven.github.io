---
layout: page
permalink: /playground/
title: Playground
description: small games driven by tiny neural nets
nav: true
nav_order: 5
---
<!-- _pages/playground.md -->

<script type="importmap">
  { "imports": { "three": "{{ '/assets/js/vendor/three.module.min.js' | relative_url }}" } }
</script>

Small games. Everything that moves is pushed around by neural nets with a few thousand parameters each, trained in plain JavaScript ([code](https://github.com/alexpondaven/alexpondaven.github.io/tree/master/assets/worldmodel)). No physics engine, no game AI.

#### World model

Drive the ball with WASD or the arrows. A small net predicts all the physics. "Show ground truth" overlays the real simulation so you can see where the model drifts.

{% include worldmodel.html %}

#### Arena

Push the crates into the ring. The collisions are learned, not simulated. Autopilot plays it with a policy trained inside the model.

{% include arena.html skip_map=true %}

#### Colony

The paragraph is alive. Letters hatch, eat other letters, turn into words, fly for a while, then put themselves back.

{% include colony.html %}

#### Chase

The red one picks its moves by simulating futures with the same nets that run the game. The threads on the floor are the futures it considered. Hold shift to see your own.

{% include chase.html skip_map=true %}
