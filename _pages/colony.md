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

The crowned one is the **elder** — the critic. It watches every outcome and hands out the +1/−1 verdicts you'll see floating up, and the others *continually learn* from them, right here in your browser: the slinger's aim starts naive and improves reward by reward; workers learn which letters are worth the risk through a tiny value model updated online.

But who judges the judge? The colony is a **dynasty of critics, three learning loops deep**:

- **Actors** (seconds): workers and the slinger learn from the reigning elder's verdicts.
- **Elders** (reigns): an elder's verdicts are its *genome* — five reward magnitudes it was born with. Each elder reigns about ninety seconds, and its fitness is the colony's measured prosperity under its teaching (deliveries, letters lost, raids repelled). A mutant elder that rewards missing teaches the slinger to miss — and its line dies out for it.
- **The Ancestor** (generations): the critic of critics. It ends reigns (mercifully early, for truly bad teachers), keeps the ledger of ancestors, chooses which lineage continues — and adapts the one thing meta-evolution can: how *bold* each successor's mutation is, growing adventurous when the dynasty stagnates and careful while it improves. You'll see it appear, briefly, whenever a crown changes heads.

Workers and the slinger age and die too; each egg hatches an heir carrying mutated weights from the strongest living kin, so generation numbers climb. The whole dynasty — every brain, the ancestor ledger, the mutation temperature — persists in your browser's localStorage: *your* colony, *your* lineage, better every visit. Movement stays the frozen 434-parameter steering net shared by every caste ([critter_train.js](/assets/worldmodel/critter_train.js), hand-rolled like [/play/](/play/) and [/arena/](/arena/)); all the learning lives in the layers above it.

{% include colony.html %}
