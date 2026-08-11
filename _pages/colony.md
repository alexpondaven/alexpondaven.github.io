---
layout: page
permalink: /colony/
title: Colony
description: Tiny agents that live in the text and take it apart
nav: true
nav_order: 6
---
<!-- _pages/colony.md -->

The paragraph below is an ecosystem. **Workers** pull it apart letter by letter, hoard the letters in a pile, then put every one back — forever. **Monsters** stalk in from the edges and *eat* letters; whatever they swallow is locked away, and the colony cannot finish its repairs until it gets them back. **Slingers** fight back: they grab letters (from the ground, the pile, or — desperate times — the text itself) and hurl them at monsters. Four hits pops a monster, scattering everything in its belly.

You're in it too: your cursor is a predator the small castes flee, and clicking a monster bonks it like a thrown letter. Help the slingers, or terrorize the workers — your call.

Every caste shares **one 434-parameter steering net**: the job list (which letter, where it goes, who to attack) is a plain state machine, but each approach, arrival, chase and escape is the net's output — behavior-cloned from scripted steering, patched and [self-forced](/play/), same hand-rolled training as [/play/](/play/) and [/arena/](/arena/): [critter_train.js](/assets/worldmodel/critter_train.js), no ML libraries. Monsters run the same brain with a heavier body (lower acceleration cap).

{% include colony.html %}
