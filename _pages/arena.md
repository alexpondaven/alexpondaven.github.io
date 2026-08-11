---
layout: page
permalink: /arena/
title: Arena
description: A 3D mini-game where every movement is neural-net output
nav: true
nav_order: 5
---
<!-- _pages/arena.md -->

Push the crates into the glowing ring. Simple — except there is **no physics engine on this page**. The ball you drive, the crates it shoves, every bounce and every collision: all of it is predicted, step by step, by two tiny neural networks (a ball net, and one crate net shared by every crate — entities conditioned on each other's relative state, the same decomposition idea as multi-subject world models). [three.js](https://threejs.org/) only draws what the nets decide.

Drive with **WASD** / arrow keys or the buttons. The nets were trained on a couple of minutes of simulated pushing — so the "physics" you feel is whatever they managed to learn from watching it. Each net also predicts its own *motion gate*: a learned "am I actually moving?" bit that holds resting objects still. (Without it, a residual bias far below the training-loss floor compounds step by step until every crate slowly conveyor-belts itself into a corner — regression can't learn *exactly zero*, but a tiny net is great at learning a binary switch.)

{% include arena.html %}

**Autopilot** hands the ball to a ~460-parameter policy net that plays the game itself. The meta bit: it learned *entirely inside the world model* — behavior-cloned from a scripted demonstrator driving these same two nets, then refined by evolution on dream rollouts, rewarded for crates delivered to the ring. It has never interacted with the real simulator (a pocket-sized version of the world-models / Dreamer idea). Press any key to take over mid-run; release to hand back.

Training scripts: [world model](/assets/worldmodel/arena_train.js), [policy](/assets/worldmodel/arena_agent_train.js) — same hand-rolled approach as the [/play/](/play/) demo, no ML libraries.
