---
layout: page
permalink: /colony/
title: Colony
description: Tiny agents that live in the text and take it apart
nav: true
nav_order: 6
---
<!-- _pages/colony.md -->

The paragraph below is not safe. Seven critters treat it as their world: they wander it, pull it apart letter by letter, hoard the letters in a corner pile — and then, eventually, put every one back where it came from, forever. Your cursor is a predator; they'll scatter. Try to guard the words.

Their to-do list (which letter, where it goes) is a plain state machine, but **every movement is steered by a tiny neural net** — a 434-parameter policy trained offline to chase targets, ease into arrivals and flee the cursor, then [self-forced](/play/) so its own rollouts stay stable. Same hand-rolled training as the [/play/](/play/) and [/arena/](/arena/) demos: [critter_train.js](/assets/worldmodel/critter_train.js), no ML libraries.

{% include colony.html %}
