---
layout: page
permalink: /colony/
title: Colony
description: Creatures whose whole life cycle is made of text
nav: false
nav_order: 6
---
<!-- _pages/colony.md -->

Everything below lives a life made entirely of the paragraph it lives in. An **egg** (the little pearl on a letter) hatches, and that letter pops out of the text as a **larva** — a caterpillar whose *body is its letters*. It crawls around hunting letters that extend it along the beginnings of real words (`ca` hunts a `t` or an `r` — its thought bubble shows exactly what it's looking for), shedding its tail in frustration when it dead-ends. The moment its body spells a real word, it curls into a **chrysalis**, pulses for a while, and emerges as a **word-butterfly** — the word itself on flapping wings, looping over the sentences it was born from. Longer words fly grander and longer.

And when the flight is over, the butterfly returns to the paragraph and **gives every letter back**: the holes in the text heal, an egg or two is left behind, and the cycle turns again. Hatchlings inherit their parent's appetite for word length, so lineages of long words breed ambitious children. The **family book** — every word that has ever lived on this page — persists in your browser, along with the bloodlines: this is *your* colony's history.

The masked **Snatcher** is the villain: it stalks caterpillars and steals the letter right off their tails for its corner hoard, setting their little lives back. Your cursor spooks everything — and **clicking the Snatcher** makes it spill its entire hoard, every stolen letter flying home at once.

Every creature — larvae, butterflies, the Snatcher, even letters mid-flight — moves on the same frozen **434-parameter steering net** trained for the [/play/](/play/) and [/arena/](/arena/) demos ([critter_train.js](/assets/worldmodel/critter_train.js), hand-rolled, no ML libraries). The life cycle is the choreography on top.

{% include colony.html %}
