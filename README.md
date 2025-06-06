# PufferAI (Forked & Modified)

This is a **fork** of the original [PufferAI repository](https://github.com/pufferai), with custom modifications.

## 🔧 Purpose of the Fork

The goal of this fork is to **extend the functionality** of PufferAI in order to:

> **Enable collapsing of the policy network for certain games using fiber symmetries.**

This allows more efficient training and inference by taking advantage of symmetries in the game state and action space, which can reduce redundancy in policy outputs.

## 🔄 Differences from Upstream

- Introduced support for **fiber symmetry-based policy collapsing**.
- Modified architecture components to allow group-invariant transformations.
- Custom preprocessing pipeline adjustments to accommodate the symmetry reductions.

> **Note:** These changes are experimental and tailored to research.

In order to reproduce the collapsed results, follow the instructions in the file: model_collapsing_guide.
The full documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support -- post here before opening issues.

