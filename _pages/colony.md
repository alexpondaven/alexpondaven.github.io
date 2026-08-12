---
layout: page
permalink: /colony/
title: Colony
description: Tiny agents that live in the text and take it apart
nav: true
nav_order: 6
---
<!-- _pages/colony.md -->

The paragraph below is a pantry of letters, and the three wordsmiths are **inventing words with it, live** — under a real economy, with only local senses. Each one can see just the letters inside its faint colored **sensing ring**, so what it can spell depends on where it stands. Watch one scout (the dotted line points at the region it chose), look around, commit to a word, shuttle the letters into its lane, and present the result to the crowned **elder** — who wanders over, inspects for a solemn second, and floats a verdict. Then the letters fly home and the text heals.

The verdict is only *gross* income. **Net reward = verdict − haul** (every step spent carrying letters is paid for), and the net is what trains them. The elder is a **novelty critic** — it remembers every word it has ever been shown, first showings score high, reruns bore it — so exotic distant letters buy novelty but cost real legwork. And when the elder is pleased it *produces*: **morsels of food arc out of good verdicts**, and eating them is the only way wordsmiths refill the energy that hauling burns. Bad poets starve. The reward signal is literally the food chain.

What they learn, you can read off the panel below the stage: a letter-transition model and length taste (REINFORCE on net reward), and — the interesting part — each wordsmith's own **foraging map**: which region of the text has been paying. Two wordsmiths working the same patch steal each other's tiles and eat the loss, so the maps push them apart into **territories**. Fresh colonies babble and crowd; older ones spread out, specialize, and coin cheap flowing words from their own turf.

And there is an adversary. The masked one is the **Plagiarist**: it lurks translucent at the margins, learns *whose* lane pays best, and when a half-spelled word is left unguarded it sneaks in and steals it — presenting it to the elder as its own. The verdict, the morsels, even the novelty (the word is used up in the archive forever) all go to the thief; the victim eats the haul cost as pure loss. Its cowardice is the same steering net's flee channel — wordsmiths, the elder and your cursor all read as threats, so it strikes only when the coast is clear. Long words mean long exposure: predation is one more pressure the colony's learning has to answer. **Click the Plagiarist** to bonk it — it panics, spills the morsels it hoarded, and bolts.

The dynasty on top survives from before: wordsmiths age, die, and hatch heirs with mutated brains; elders reign for two minutes and are succeeded along a fitness-selected lineage (fitness = new vocabulary coaxed out per minute); the **Ancestor** appears at each succession, crowning taste-mutated successors and tuning its mutation boldness to the dynasty's trend. Everything — brains, archive, ledger — persists in localStorage: *your* colony, *your* vocabulary. Movement stays the frozen 434-parameter steering net shared by all ([critter_train.js](/assets/worldmodel/critter_train.js), hand-rolled like [/play/](/play/) and [/arena/](/arena/)).

{% include colony.html %}
