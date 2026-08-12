---
layout: page
permalink: /colony/
title: Colony
description: Tiny agents that live in the text and take it apart
nav: true
nav_order: 6
---
<!-- _pages/colony.md -->

The paragraph below is a pantry of letters, and the three wordsmiths are **inventing words with it, live**. Each one proposes a word (watch its ghost appear in that worker's lane), shuttles the letters out of the text to spell it, and presents it to the crowned **elder** — who wanders over, inspects it for a solemn second, and floats its verdict over the word. Then every letter flies back home and the text heals.

The elder is a **novelty critic**. It remembers every word it has ever been shown (permanently — the archive is part of your browser's saved colony), and nothing bores it like a rerun: first showings score high, repeats decay fast. Its heritable *taste* also weighs length and flow (consonant–vowel alternation). So the only strategy that keeps working is **inventing new words forever** — novelty search, running in a paragraph.

And the wordsmiths do learn to invent: each carries a tiny letter-transition model updated by REINFORCE on the verdicts. Fresh colonies babble consonant mush; give them some generations and they drift toward flowing, wordish coinages — which you can literally read as they're spelled out. When one collapses onto a favorite word, the archive's boredom pushes it right back into exploring.

The dynasty on top survives from before: wordsmiths age, die, and hatch heirs with mutated brains; elders reign for two minutes and are succeeded along a fitness-selected lineage (fitness = new vocabulary coaxed out per minute); the **Ancestor** appears at each succession, crowning taste-mutated successors and tuning its mutation boldness to the dynasty's trend. Everything — brains, archive, ledger — persists in localStorage: *your* colony, *your* vocabulary. Movement stays the frozen 434-parameter steering net shared by all ([critter_train.js](/assets/worldmodel/critter_train.js), hand-rolled like [/play/](/play/) and [/arena/](/arena/)).

{% include colony.html %}
