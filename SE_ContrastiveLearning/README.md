# Stimulus Equivalence simulations with InfoNCE Loss in a Reinforcement Learning setting
## Project Overview

This repository contains Python code for a reinforcement learning experiment incorporating InfoNCE loss. The goal is to learn embeddings for stimuli and use them for train and test stimulus equivalence.

**Key Components:**
Data: The code expects a pandas DataFrame containing trial information with columns like st_sample, st_comp1, st_comp2, st_comp3, and option_answer.
Embeddings: A dictionary storing pre-computed embeddings for stimuli.
Q-Network: A simple Q-network based on dot product similarity between anchor and options.
InfoNCE Loss: Implemented for embedding learning.
Training Loop: Iteratively samples trials, computes Q-values, selects actions, updates embeddings, and calculates loss.

**Dependencies:**
NumPy
Pandas
