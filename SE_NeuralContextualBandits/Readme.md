# Simulating a stimulus equivalence (SE) experiment with a neural contextual bandit (NCB) agent.

This experiment employed a Matching-to-Sample (MTS) protocol to assess stimulus equivalence (SE) in a contextual bandit framework. Each experimental trial consisted of presenting the subject with one sample stimulus and three comparison stimuli.

All stimuli were one-hot encoded, assigning a unique feature vector representation to each stimulus. A trial was encoded by concatenating the one-hot encoded representations of the four stimuli in the order: sample, comparison 1, comparison 2, comparison 3.

During Training Process Each encoded trial was presented as context to the NCB agent. As reward function, the agent received a reward of 1 for selecting the correct comparison stimulus (matching the encoded action) and 0 for selecting an incorrect comparison. All encoded trials were used for training the NCB agent. Following training on baseline relations, the NCB agent's ability to demonstrate emergent properties of SE was evaluated. The agent was presented with encoded trials for reflexivity, symmetry, and transitivity relationships not explicitly trained upon. Performance on these evaluation trials was measured as the total reward received by the agent divided by the total number of trials for each subset (reflexivity, symmetry, transitivity). This provided a measure of the agent's ability to respond correctly to unseen relationships.

This experiment utilized the Neural Contextual Bandit (NCB) agent presented in [Neural Contextual Bandits with UCB-based Exploration](https://paperswithcode.com/paper/neural-contextual-bandits-with-upper). Experiment was adapted from [NeuralUCB repo](https://github.com/uclaml/NeuralUCB).
