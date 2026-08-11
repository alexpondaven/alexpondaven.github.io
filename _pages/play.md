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
- Soft guidance: each step, the model's state gets nudged a few percent toward the true trajectory — position *and* velocity, so corrections carry the motion instead of teleporting the ball. A proportional pull alone stalls at a fixed offset where it exactly balances the model's bias (P-controller steady-state error), so there's also a tiny integral term that cancels the bias — at rest it settles *onto* the true trajectory. Toggle **"Guidance"** off to watch the rollout run unaided.

(I first tried making the *network* learn that correction — true state as an extra input, pull toward it in proportion to a confidence signal. At ~1,800 parameters it refuses: it learns an all-or-nothing snap, never a proportional pull. The gate is still in the weights; the training script tells the story.)

Even guided, the pull is weak enough that the model visibly wanders — the same failure mode real world model research spends most of its effort on, just visible here in seconds instead of minutes.

{% include worldmodel.html %}

Training script (hand-rolled backprop, no ML libraries) is [here](/assets/worldmodel/train.js).
