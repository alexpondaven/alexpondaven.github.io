---
layout: page
permalink: /play/
title: Play
description: A tiny neural world model, running entirely in your browser
nav: false
nav_order: 4
---
<!-- _pages/play.md -->

A world model predicts what happens next from a state and an action. [ActionParty](https://action-party.github.io/) and [DiTFlow](https://ditflow.github.io/) do this for video frames. This one does it for four numbers: position and velocity — no pixels involved, much smaller problem.

It's a ~2,300-parameter network trained purely on simulated physics — never told the rules, only ever shown replayed transitions. Besides the next state, it also predicts the next *action*: a small behavior prior learned from the same play data, so a ball can keep playing itself. Move the ball with **WASD** / arrow keys or the buttons below. Toggle **"show ground truth"** for a faint outline of what actually happens, and watch the model drift from it.

That drift is mostly the model eating its own predictions — small errors compound. Two things fight it here, both real techniques, not hidden tricks:

- [Self-forcing](https://arxiv.org/abs/2506.08009): alongside correct history, it also trains on states it reached by *its own* mistakes, learning to correct rather than spiral.
- Learned guidance: the true state is fed in as an extra conditioning token, and the network is trained — with conditioning dropout, like long-video models re-conditioning on keyframes — to steer its own rollout gently toward that token, position *and* velocity. The pull you see is the network's output, not post-processing. Toggle **"Guidance"** off to zero the token and watch it run unaided.

(One design lesson baked in: exposing the pull *strength* as a confidence input fails at this scale — a couple thousand parameters learn an all-or-nothing snap, never a proportional dial. Fixing the rate during training works, because then the correction is just a linear function of the error. And since a memoryless net can only learn proportional-style control, a small steady-state offset remains — integral action would need memory.)

Even guided, the pull is weak enough that the model visibly wanders — the same failure mode real world model research spends most of its effort on, just visible here in seconds instead of minutes.

{% include worldmodel.html %}

**Cycle mode** drops the ground truth entirely: three copies of the model in a ring, each gently following the next, and every ball choosing its own moves by sampling the model's predicted next action — the bright one included, until you press a key and take over (let go and it resumes playing itself). Everything on screen is the network: the motion is its dynamics head, the decisions are its behavior prior, and the only thing coupling the ring is the conditioning tokens. Two details doing quiet work: without the balls' own actions the ring collapses onto a single point, and the "gently" comes free from linearity — feeding a token only *partway* toward the target scales the learned pull down by the same fraction, no retraining. A tiny taste of multiplayer world models: several agents, one learned dynamics, coupled through conditioning.

Training script (hand-rolled backprop, no ML libraries) is [here](/assets/worldmodel/train.js).
