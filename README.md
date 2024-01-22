# Simulating Equivalence relations
SE is an ability only seen in humans and is related with language; even pre-linguistic children tends to show poor performance in equivalence tests.
"Stimulus equivalence provides a solid framework for testing artificial systems' abilities to show human-like symbolic behavior" [Tovar, et al (2023)](https://onlinelibrary.wiley.com/doi/10.1002/jeab.829)

## What is Stimulus Equivalence (SE)?
In an experiment, during the training phase a participant is rewarded when: 
- an image of a cow is presented and the word “cow” is selected
- an image of a cow is presented and the equivalent word in Spanish “vaca” is selected
- an image of a cat is presented and the word “cat” is selected
- an image of a cat is presented and its equivalent in Spanish “gato” is selected.

After training, without reward, in the test phase: 
- the word “cow” is presented, and the participant selects the word “vaca”
- the word “gato” is presented, and the participant selects the image of a cat
- the word “vaca” is presented, and the participant selects the word “vaca”. 

![Stimulus Equivalence example](https://www.mdpi.com/mti/mti-07-00039/article_deploy/html/images/mti-07-00039-g001.png)

In this example, six stimuli were trained to form two classes (1 and 2) with three members (A, B, and C) each, in an MTS procedure. Stimulus Class 1 consists of the image of a cat (A1), the word cat (B1), and the Spanish word “gato” (C1), while Stimulus Class 2 consists of the image of a cow (A2), the word “cow” (B2), and the Spanish word “vaca” (C2). The two classes were formed by training the A1–B1, A1–C1, A2–B2, and A2–C2 relations. The test phase showed emergent relations between the stimuli within the class members without explicit training: transitivity (B2–C2), symmetry (C1–A1), and reflexivity (C2–C2).

## Evaluating a psychological phenomena on machine learning algorithms *
(* WORK IN PROGRESS)

The experiments are based on matching to sample procedures for evaluating relational derived response, a psychological phenomena related with complex behaviour such as cognition or language, from a behaviorist viewpoint (the psychological "school" that inspired reinforcement learning). The objective is to create a ML algorithm that be capable of pass equivalence tests.

There are related experiments and publications separated by folders.

### Publications
[Differences of Training Structures on Stimulus Class Formation in Computational Agents](https://www.mdpi.com/2414-4088/7/4/39)

## References
Sidman, M. (2000). Equivalence relations and the reinforcement contingency. Journal of the Experimental Analysis of Behavior, 74(1), 127–146. http://doi.org/10.1901/jeab.2000.74-127
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1284788/

Sidman, M. (2009). Equivalence Relations and Behavior: An Introductory Tutorial. The Analysis of Verbal Behavior, 25(1), 5–17.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2779070/

Tovar, Á.E., Torres-Chávez, Á., Mofrad, A.A., Arntzen, E. (2023) Computational models of stimulus equivalence: An intersection for the study of symbolic behavior. J. Exp. Anal. Behav. 119, 407–425. https://doi.org/10.1002/jeab.829
