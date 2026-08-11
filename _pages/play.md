---
layout: page
permalink: /play/
title: Play
description: A tiny neural world model, running entirely in your browser
nav: true
nav_order: 4
---
<!-- _pages/play.md -->

A **world model** is a neural network that learns to predict what happens next, given the current state and an action — the same idea behind [ActionParty](https://action-party.github.io/) and [DiTFlow](https://ditflow.github.io/), just at a much bigger scale.

This is a *tiny* one: a ~900-parameter network trained on a couple of minutes of simulated ball-in-a-box physics. It has never been told the rules of physics — it only learned by replaying transitions. Move the ball with **WASD** / arrow keys, or the buttons below. Toggle **"show ground truth"** to see a faint outline of what *actually* happens physically, and watch the model's prediction drift away from it — the same failure mode world model research spends most of its time fighting, just easy to see here because the model is small enough to break in seconds instead of minutes.

{% include worldmodel.html %}

Curious how it works? The training script (hand-rolled backprop, no ML libraries) is [here](/assets/worldmodel/train.js) — it trains in about 10 seconds on a laptop CPU and exports straight to the ~18KB `weights.json` this page loads.
