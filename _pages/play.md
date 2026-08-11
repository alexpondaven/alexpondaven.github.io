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
- Learned guidance: the true state is fed in as an extra conditioning token, and the network is trained — with conditioning dropout, like long-video models re-conditioning on keyframes — to steer its own rollout gently toward that token, position *and* velocity. The pull you see is the network's output, not post-processing. Toggle **"Guidance"** off to zero the token and watch it run unaided.

(One design lesson baked in: exposing the pull *strength* as a confidence input fails at this scale — ~1,800 parameters learn an all-or-nothing snap, never a proportional dial. Fixing the rate during training works, because then the correction is just a linear function of the error. And since a memoryless net can only learn proportional-style control, a small steady-state offset remains — integral action would need memory.)

Even guided, the pull is weak enough that the model visibly wanders — the same failure mode real world model research spends most of its effort on, just visible here in seconds instead of minutes.

{% include worldmodel.html %}

Training script (hand-rolled backprop, no ML libraries) is [here](/assets/worldmodel/train.js).
