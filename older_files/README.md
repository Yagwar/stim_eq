# Equivalence relations
### Evaluating a psychological phenomena on machine learning algorithms *
(* WORK IN PROGRESS)

The experiments are based on matching to sample procedures for evaluating relational derived response, a psychological phenomena related with complex behaviors such as cognition or language, from a behaviorist viewpoint (the psychological "school" that inspired reinforcement learning). Each trial consists on the presentation of 6 (out of 18 possible) stimuli: one sample and 5 comparators. The agent must select the position of the related comparator.

The experiment has 2 phases: training and evaluation. The objective is to create a ML algorithm that be capable of mark the related stimulus, evidencing emergent relations among the stimuli.

The data consist on .csv files,  3  for every task (training and 4 evaluations of emergent relations):
- "labels" file. It was made for informative purposes, showing the stimuli presented on the trial. 
- "values" file. Those are the stimuli encoded (108 bits), representing a trial for the task.
- "answer" file. The encoded answer (5 bit) marking the expected answer of the trial.


The stimuli encoding 


    stims={"A1":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "A2":[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "A3":[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "A4":[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "A5":[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "A6":[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "TX":[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "B1":[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
           "B2":[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
           "B3":[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
           "B4":[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
           "B5":[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
           "B6":[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
           "TY":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
           "C1":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
           "C2":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
           "C3":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
           "C4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
           "C5":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], # For explicitly train class emergency
           "C6":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], # For explicitly train class emergency
           "TZ":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1] # For explicitly train class emergency
          }

    options={"O_1":[1,0,0,0,0],
             "O_2":[0,1,0,0,0],
             "O_3":[0,0,1,0,0],
             "O_4":[0,0,0,1,0],
             "O_5":[0,0,0,0,1],
             "O_0":[0,0,0,0,0],
            }

## References

Sidman, M. (2000). Equivalence relations and the reinforcement contingency. Journal of the Experimental Analysis of Behavior, 74(1), 127–146. http://doi.org/10.1901/jeab.2000.74-127
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1284788/

Sidman, M. (2009). Equivalence Relations and Behavior: An Introductory Tutorial. The Analysis of Verbal Behavior, 25(1), 5–17.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2779070/


### also check:

Testing Response-Stimulus Equivalence Relations Using Differential Responses as a Sample
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1592360/
