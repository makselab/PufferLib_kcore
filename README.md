# PufferAI (Forked & Modified)

This is a **fork** of the original [PufferAI repository](https://github.com/pufferai), with custom modifications.

## 🔧 Purpose of the Fork

The goal of this fork is to **extend the functionality** of PufferAI in order to:

> **Enable collapsing of the policy network for certain games using fibration symmetries and perform symmetry breaking.**

This enables more efficient training and inference by leveraging symmetries in the game state and action space, thereby reducing redundancy in policy outputs and improving performance.

## 🔄 Differences from Upstream

- Introduced support for **fiber symmetry-based policy collapsing**.
- Modified architecture components to allow group-invariant transformations.
- Custom preprocessing pipeline adjustments to accommodate the symmetry reductions and symmetry breaking.

In order to reproduce the collapsed results, follow the instructions in the file: [Model Collapsing Guide](model_collapsing_guide.MD)



The full documentation of the underlying games is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support -- post here before opening issues.

