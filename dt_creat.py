
# coding: utf-8

# # EQUIVALENCE CLASS:
# A stimulus class (usually produced through conditional discrimination in matching-to-sample) that includes all possible emergent relations among its members. The properties of an equivalence class are derived from the logical relations of reflexivity, symmetry, and transitivity. **Reflexivity** *refers to the matching of a sample to itself*, sometimes called identity matching (AA, BB, CC, in these examples, each letter pair represents a sample and its matching comparison stimulus). **Symmetry** *refers to the reversibility of a relation (if AB, then BA)*. **Transitivity** *refers to the transfer of the relation to new combinations through shared membership (if AB and BC, then AC)*. 
# If these properties are characteristics of a matching to-sample performance, then training AB and BC may produce AC, BA, CA, and CB as emergent relations (reflexivity provides the three other possible relations, AA, BB, and CC). Given AB and BC, for example, the combination of symmetry and transitivity implies the CA relation. The emergence of all possible stimulus relations after only AB and BC are trained through contingencies is the criterion for calling the three stimuli members of an equivalence class. The class can be extended by training new stimulus relations (e.g., if CD is learned, then AD, DA, BD, DB, and DC may be created as emergent relations). Stimuli that are members of an equivalence class are likely also to be functionally equivalent. It remains to be seen whether the logical properties of these classes are fully consistent with their behavioral ones. Cf. ** EQUIVALENCE RELATION**. ([source](http://www.scienceofbehavior.com/lms/mod/glossary/view.php?id=408&mode=letter&hook=E&sortkey=CREATION&sortorder=asc&fullsearch=0&page=3))
# 

# # Libraries

# In[1]:

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')
plt.style.use('seaborn')


# In[2]:

def view_trial(trial_labels,trial_values,trial_ans, n_trial):
    print (n_trial,len(trial_labels))
    print (np.array(trial_values[n_trial]).reshape((6,3,6)))
    print (trial_labels[n_trial])
    print (trial_ans[n_trial])

def create_trials(stims,pair):
    #Take a set of stimuli and a set of pairs, then find the mode (letter) of the comparatosr and combines them.
    filt_tr=[pair[1][0]==stim[0] for stim in stims.keys()]# filter the simulus of the mode of the comparator
    comprs=np.array(stims.keys())[filt_tr] # Get the set of comparators
    comprs_set_1=np.array([[p,q,r,s,t]for p in comprs for q in comprs for r in comprs for s in comprs for t in comprs])# all the combinations of the comparators
    comprs_filt=[(np.sum(cmpr_set==pair[1])==1) for cmpr_set in comprs_set_1]#==1 for the presence of the target comparator ## <2 for target comparator and no answer.
    comprs_set=comprs_set_1[comprs_filt]# filtered set of comparators with the sample presented just once.
    train_labels=np.insert(comprs_set,0, pair[0], axis=1)# train labels with sample and comparators
    train_answers=np.array([(tr_lbl==pair[1])*1 for tr_lbl in comprs_set]) # Encoded answers for the trials. 
    train_values=np.array([[stims[stml]for stml in stmls] for stmls in train_labels]).reshape(len(train_labels),(6*18))# create a list of the encoded values of the trial
    return train_labels,train_values,train_answers

def create_set(trials_pairs, stims):
    trialset=[create_trials(stims,pair) for pair in trials_pairs]
    labels=np.array([tr_lb for tr_pr in trialset for tr_lb in tr_pr[0]])
    values=np.array([tr_lb for tr_pr in trialset for tr_lb in tr_pr[1]])
    answer=np.array([tr_lb for tr_pr in trialset for tr_lb in tr_pr[2]])
    return labels,values,answer


# In[3]:

stims={"A1":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       "A2":[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       "A3":[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       "A4":[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
       "A5":[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
       "A6":[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
       "B1":[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
       "B2":[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
       "B3":[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
       "B4":[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
       "B5":[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
       "B6":[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
       "C1":[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
       "C2":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
       "C3":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
       "C4":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
       "C5":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
       "C6":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
      }

options={"O_1":[1,0,0,0,0],
         "O_2":[0,1,0,0,0],
         "O_3":[0,0,1,0,0],
         "O_4":[0,0,0,1,0],
         "O_5":[0,0,0,0,1],
         "O_0":[0,0,0,0,0],
        }


# # Trainning 
# ### Relation $A_{n}-B_{n}$ and $B_{n}-C_{n}$

# In[4]:

train_pairs=np.array([["A1","B1"],["B1","C1"],
                      ["A2","B2"],["B2","C2"],
                      ["A3","B3"],["B3","C3"],
                      ["A4","B4"],["B4","C4"],
                      ["A5","B5"],["B5","C5"],
                      ["A6","B6"],["B6","C6"]
                     ])


# In[5]:

train_labels,train_values,train_answer=create_set(train_pairs, stims)


# In[6]:

pd.DataFrame(train_labels, columns=["Sample",
                                    "Comparator_1",
                                    "Comparator_2",
                                    "Comparator_3",
                                    "Comparator_4",
                                    "Comparator_5"]).to_csv('train_labels.csv',index=False)#train_labels
pd.DataFrame(train_values).to_csv('train_values.csv',index=False)#train_values
pd.DataFrame(train_answer,columns=["Choice_1",
                        "Choice_2",
                        "Choice_3",
                        "Choice_4",
                        "Choice_5"]).to_csv('train_answer.csv',index=False)#train_answer


# In[7]:

view_trial(train_labels,train_values,train_answer,random.randrange(len(train_labels)))


# # Reflexivity evaluation
# 
# ### Given the sample stimulus $A_{n}$ the agent must select $A_{n}$ among the comparator stimuli

# In[8]:

reflexiv_pairs=np.array([[stm,stm]for stm in stims])
reflexivity_labels, reflexivity_values, reflexivity_answer =create_set(reflexiv_pairs, stims)


# In[9]:

pd.DataFrame(reflexivity_labels, columns=["Sample",
                                    "Comparator_1",
                                    "Comparator_2",
                                    "Comparator_3",
                                    "Comparator_4",
                                    "Comparator_5"]).to_csv('reflexivity_labels.csv',index=False)#reflexivity_labels
pd.DataFrame(reflexivity_values).to_csv('reflexivity_values.csv',index=False)#reflexivity_values
pd.DataFrame(reflexivity_answer,columns=["Choice_1",
                        "Choice_2",
                        "Choice_3",
                        "Choice_4",
                        "Choice_5"]).to_csv('reflexivity_answer.csv',index=False)#reflexivity_answer


# In[10]:

view_trial(reflexivity_labels,reflexivity_values,reflexivity_answer,random.randrange(len(reflexivity_labels)))


# # Symmetry evaluation
# ### Given the trainning pairs, the agent must select the comparator $A_{n}$ in presence of the sample $B_{n}$  and the comparator $B_{n}$ in presence of the sample $C_{n}$ 

# In[11]:

symmetry_pairs=np.array([[tr_pr[1],tr_pr[0]]for tr_pr in train_pairs])
symmetry_labels, symmetry_values, symmetry_answer =create_set(symmetry_pairs, stims)


# In[12]:

pd.DataFrame(symmetry_labels, columns=["Sample",
                                    "Comparator_1",
                                    "Comparator_2",
                                    "Comparator_3",
                                    "Comparator_4",
                                    "Comparator_5"]).to_csv('symmetry_labels.csv',index=False)#symmetry_labels
pd.DataFrame(symmetry_values).to_csv('symmetry_values.csv',index=False)#symmetry_values
pd.DataFrame(symmetry_answer,columns=["Choice_1",
                        "Choice_2",
                        "Choice_3",
                        "Choice_4",
                        "Choice_5"]).to_csv('symmetry_answer.csv',index=False)#symmetry_answer


# In[13]:

view_trial(symmetry_labels,symmetry_values,symmetry_answer,random.randrange(len(symmetry_labels)))


# # Transitivity
# ### Given the trainning pairs, the agent must select the comparator $C_{n}$ in presence of the sample $A_{n}$

# In[14]:

transitivity_pairs=np.array([["A1","C1"],
                             ["A2","C2"],
                             ["A3","C3"],
                             ["A4","C4"],
                             ["A5","C5"],
                             ["A6","C6"]
                            ])


# In[15]:

transitivity_labels, transitivity_values, transitivity_answer =create_set(transitivity_pairs, stims)


# In[16]:

pd.DataFrame(transitivity_labels, columns=["Sample",
                                    "Comparator_1",
                                    "Comparator_2",
                                    "Comparator_3",
                                    "Comparator_4",
                                    "Comparator_5"]).to_csv('transitivity_labels.csv',index=False)#transitivity_labels
pd.DataFrame(transitivity_values).to_csv('transitivity_values.csv',index=False)#transitivity_values
pd.DataFrame(transitivity_answer,columns=["Choice_1",
                        "Choice_2",
                        "Choice_3",
                        "Choice_4",
                        "Choice_5"]).to_csv('transitivity_answer.csv',index=False)#transitivity_answer


# In[17]:

view_trial(transitivity_labels,transitivity_values,transitivity_answer,random.randrange(len(transitivity_labels)))


# # Equivalence
# ### Given the trainning pairs, the agent must select the comparator $A_{n}$ in presence of the sample $C_{n}$

# In[18]:

equivalence_pairs=np.array([[tr_pr[1],tr_pr[0]]for tr_pr in transitivity_pairs])


# In[19]:

equivalence_labels, equivalence_values, equivalence_answer =create_set(equivalence_pairs, stims)


# In[20]:

pd.DataFrame(equivalence_labels, columns=["Sample",
                                    "Comparator_1",
                                    "Comparator_2",
                                    "Comparator_3",
                                    "Comparator_4",
                                    "Comparator_5"]).to_csv('equivalence_labels.csv',index=False)#equivalence_labels
pd.DataFrame(equivalence_values).to_csv('equivalence_values.csv',index=False)#equivalence_values
pd.DataFrame(equivalence_answer,columns=["Choice_1",
                        "Choice_2",
                        "Choice_3",
                        "Choice_4",
                        "Choice_5"]).to_csv('equivalence_answer.csv',index=False)#equivalence_answer


# In[21]:

view_trial(equivalence_labels,equivalence_values,equivalence_answer,random.randrange(len(equivalence_labels)))


# In[ ]:



