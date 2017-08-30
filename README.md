# Equivalence relations
### Evaluating a psychological phenomena on machine learning algorithms *

* WORK IN PROGRESS

This python script contains arrays on the task of evaluating relational derived response as a psychological phenomena related with complex behaviors such as cognition or language from a behaviorist viewpoint.

The experiments are based on matching to sample procedures for evaluating relational response. Each trial consists on the presentation of 4 stimulus, a sample and 3 comparators. The participant must select the position of the comparator related to the sample according to the training. 

The data are arrays of two list of bits where the first one is the encoded ( one sample and 3 comparators) set of stimulus presented. The second is the (target) expected correct answer.


The stimulus encoding 

    stims={"A":[1,0,0,0,0],
           "B":[0,1,0,0,0],
           "C":[0,0,1,0,0],
           "D":[0,0,0,1,0],
           "E":[0,0,0,0,1]
           } 
The answer choices encoding 

    options={"O_1":[1,0,0],
             "O_2":[0,1,0],
             "O_3":[0,0,1],
             "O_0":[0,0,0]
             }

## References

Sidman, M. (2000). Equivalence relations and the reinforcement contingency. Journal of the Experimental Analysis of Behavior, 74(1), 127–146. http://doi.org/10.1901/jeab.2000.74-127
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1284788/

Sidman, M. (2009). Equivalence Relations and Behavior: An Introductory Tutorial. The Analysis of Verbal Behavior, 25(1), 5–17.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2779070/

http://www.txaba.org/files/pdfs/Conference2017/StimulusEquivalence_Vaidya.pdf

### also check:

Testing Response-Stimulus Equivalence Relations Using Differential Responses as a Sample
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1592360/
