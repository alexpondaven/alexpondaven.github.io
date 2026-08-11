---
layout: page
permalink: /play/
title: Play
description: A tiny neural world model, running entirely in your browser
nav: true
nav_order: 4
---
<!-- _pages/play.md -->

A world model predicts what happens next from a state and an action. [ActionParty](https://action-party.github.io/) and [DiTFlow](https://ditflow.github.io/) do this for video frames. This one does it for four numbers: position and velocity — no pixels involved, much smaller problem.

It's a ~1,800-parameter network trained purely on simulated physics — never told the rules, only ever shown replayed transitions. Move the ball with **WASD** / arrow keys or the buttons below. Toggle **"show ground truth"** for a faint outline of what actually happens, and watch the model drift from it.

That drift is mostly the model eating its own predictions — small errors compound. Two things fight it here, both real techniques, not hidden tricks:

- [Self-forcing](https://arxiv.org/abs/2506.08009): alongside correct history, it also trains on states it reached by *its own* mistakes, learning to correct rather than spiral.
- Periodic grounding: every so often the model is fed the real state as a trusted correction — it was trained to snap toward this when told to. Real long-video models do the same with keyframes. Toggle **"Grounding"** off to see it drift unchecked.

Even grounded, it still wanders a little between corrections. That's the point — the same failure mode real world model research spends most of its effort on, just visible here in seconds instead of minutes.

{% include worldmodel.html %}

Training script (hand-rolled backprop, no ML libraries) is [here](/assets/worldmodel/train.js).
