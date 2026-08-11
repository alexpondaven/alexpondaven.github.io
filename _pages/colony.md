---
layout: page
permalink: /colony/
title: Colony
description: Tiny agents that live in the text and take it apart
nav: true
nav_order: 6
---
<!-- _pages/colony.md -->

The paragraph below is an ecosystem — and it is **learning while you watch**. Workers pull the text apart letter by letter, hoard it in a pile, then put every word back. Every so often a monster raids and eats letters; the lone slinger drives it off by throwing letters at it (three hits and it pops, coughing everything back up). Your cursor scares the small ones, and clicking the monster bonks it.

The crowned one is the **elder** — the critic. It watches every outcome and hands out the little +1/−1 verdicts you'll see floating up. The others *continually learn* from those rewards, right here in your browser:

- the **slinger's aim starts naive** — it throws straight at a moving monster and misses. Each hit pulls its learned lead toward what just worked and calms its exploration noise; each miss widens the search. Watch the hit rate in the status line climb.
- **workers learn which letters are worth it**: they score candidate jobs with a tiny value model (distance, how close the monster is, loose vs in-text) and update it online — +1 for a delivery, −1.5 when a letter they claimed gets eaten, −0.5 for a panicked drop. After a few raids they learn not to harvest next to a monster.

What they learn is saved locally, so *your* colony is better when you come back. Movement is still the frozen 434-parameter steering net shared by every caste ([critter_train.js](/assets/worldmodel/critter_train.js), hand-rolled like [/play/](/play/) and [/arena/](/arena/)); the learning lives in the decision layer on top.

{% include colony.html %}
