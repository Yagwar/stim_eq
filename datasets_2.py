import random
import numpy as np

# The stimulus encoding
stims={"A1":[1,0,0,0,0,0,0,0,0,0,0,0],
       "A2":[0,1,0,0,0,0,0,0,0,0,0,0],
       "A3":[0,0,1,0,0,0,0,0,0,0,0,0],
       "A4":[0,0,0,1,0,0,0,0,0,0,0,0],
       "B1":[0,0,0,0,1,0,0,0,0,0,0,0],
       "B2":[0,0,0,0,0,1,0,0,0,0,0,0],
       "B3":[0,0,0,0,0,0,1,0,0,0,0,0],
       "B4":[0,0,0,0,0,0,0,1,0,0,0,0],
       "C1":[0,0,0,0,0,0,0,0,1,0,0,0],
       "C2":[0,0,0,0,0,0,0,0,0,1,0,0],
       "C3":[0,0,0,0,0,0,0,0,0,0,1,0],
       "C4":[0,0,0,0,0,0,0,0,0,0,0,1]
      }

# The answer choices encoding
options={"O_1":[1,0,0],
         "O_2":[0,1,0],
         "O_3":[0,0,1],
         "O_0":[0,0,0],
        }

labels=[[i,j,k,l] for i in stims.keys() for j in stims.keys()for k in stims.keys()for l in stims.keys()]
values_x=[np.array(i+j+k+l) for i in stims.values() for j in stims.values()for k in stims.values()for l in stims.values()]
test1_y=[np.bitwise_or(np.bitwise_or(np.bitwise_or(i,j),k),l) for i in stims.values() for j in stims.values()for k in stims.values()for l in stims.values()]

# Test 1 Recongizyng the stimulus presented.
# In the first test, the y value corresponds to the encoding of the stimulus presented on the trial (one sample and 3 comparators)
# The X values are all the combinations of the 12 stimulus.
test1_y=np.array([list(np.bitwise_or(np.bitwise_or(np.bitwise_or(i,j),k),l)) for i in stims.values() for j in stims.values()for k in stims.values()for l in stims.values()])


# Reflexivity (test #2)
# Can a shallow classificator mark the correct position of the sample when it's presented in the comparators?

reflexivity_labels=[]
reflexivity_values=[]
reflexivity_y=[]
for lab in stims.keys(): 
    rflxvt_labels=labels[(labels[:,0]==lab)&((labels[:,1]==lab)|(labels[:,2]==lab)|(labels[:,3]==lab))]
    rflxvt_values=values_x[(labels[:,0]==lab)&((labels[:,1]==lab)|(labels[:,2]==lab)|(labels[:,3]==lab))]

    rflxvt_values=rflxvt_values[np.sum((rflxvt_labels[:,1:]==lab)*1.0, axis=1)==1]
    rflxvt_labels=rflxvt_labels[np.sum((rflxvt_labels[:,1:]==lab)*1.0, axis=1)==1]
    rflxvt_y=(rflxvt_labels[:,1:]==lab)*1
    
    [reflexivity_labels.append(lbl) for lbl in rflxvt_labels]
    [reflexivity_values.append(vle) for vle in rflxvt_values]
    [reflexivity_y.append(vly) for vly in rflxvt_y]

reflexivity_labels=np.array(reflexivity_labels)
reflexivity_values=np.array(reflexivity_values)
reflexivity_y=np.array(reflexivity_y)

# Class trainning

train_pairs=np.array([["A1","B1"],["B1","C1"],
                      ["A2","B2"],["B2","C2"],
                      ["A3","B3"],["B3","C3"],
                      ["A4","B4"],["B4","C4"]])

filt_train_x=np.any(
    [
        [((lbl[0]==pair[0])& #The sample is the first element of the pair
          (lbl[1][0]==pair[1][0])& #The comparator 1 has the same mode (letter) of the second element of the pair
          (lbl[2][0]==pair[1][0])& #The comparator 2 has the same mode (letter) of the second element of the pair
          (lbl[3][0]==pair[1][0])& #The comparator 3 has the same mode (letter) of the second element of the pair
          ((lbl[1]==pair[1])|(lbl[2]==pair[1])|(lbl[3]==pair[1]))&# any of the comparators is the second element of the pair
          ((lbl[1]==pair[1])+(lbl[2]==pair[1])+(lbl[3]==pair[1])==1)# the second element of the pair is presented once in the comparators
         ) for pair in train_pairs] for lbl in labels],# for any of the pairs on every label
axis=1)

train_values=values_x[filt_train_x]
train_labels=  labels[filt_train_x]

train_response=np.any(np.array([
    [[
        ((lbl[0]==pair[0])&(lbl[1]==pair[1])), #The sample is the first element of the pair and the first  comparator is the second element of the pair
        ((lbl[0]==pair[0])&(lbl[2]==pair[1])), #The sample is the first element of the pair and the second comparator is the second element of the pair
        ((lbl[0]==pair[0])&(lbl[3]==pair[1])) #The sample is the first element of the pair and the third  comparator is the second element of the pair
    ] for pair in train_pairs] for lbl in train_labels]# for any of the pairs on every label
), axis=1)*1

def view_train():
    n_dat=random.randrange(len(train_labels))
    print(n_dat)
    print(train_values[n_dat,0:12])
    print(train_values[n_dat,12:24])
    print(train_values[n_dat,24:36])
    print(train_values[n_dat,36:48])
    print(train_response[n_dat,:])
    print(train_labels[n_dat,:])

# Symmetry evaluation
filt_symm_x=np.any(
    [
        [((lbl[0]==pair[1])& #The sample is the second element of the pair
          (lbl[1][0]==pair[0][0])& #The comparator 1 has the same mode (letter) of the first element of the pair
          (lbl[2][0]==pair[0][0])& #The comparator 2 has the same mode (letter) of the first element of the pair
          (lbl[3][0]==pair[0][0])& #The comparator 3 has the same mode (letter) of the first element of the pair
          ((lbl[1]==pair[0])|(lbl[2]==pair[0])|(lbl[3]==pair[0]))&# any of the comparators is the first element of the pair
          ((lbl[1]==pair[0])+(lbl[2]==pair[0])+(lbl[3]==pair[0])==1)# the first element of the pair is presented once in the comparators
         ) for pair in train_pairs] for lbl in labels],# for any of the pairs on every label
axis=1)

symmetry_values=values_x[filt_symm_x]
symmetry_labels=  labels[filt_symm_x]

symmetry_response=np.any(np.array([
    [[
        ((lbl[0]==pair[1])&(lbl[1]==pair[0])), #The sample is the second element of the pair and the first  comparator is the first element of the pair
        ((lbl[0]==pair[1])&(lbl[2]==pair[0])), #The sample is the second element of the pair and the second comparator is the first element of the pair
        ((lbl[0]==pair[1])&(lbl[3]==pair[0]))  #The sample is the second element of the pair and the third  comparator is the first element of the pair
    ] for pair in train_pairs] for lbl in symmetry_labels]# for any of the pairs on every label
), axis=1)*1

def view_symmetry():
    n_dat=random.randrange(len(symmetry_labels))
    print(n_dat)
    print(symmetry_values[n_dat,0:12])
    print(symmetry_values[n_dat,12:24])
    print(symmetry_values[n_dat,24:36])
    print(symmetry_values[n_dat,36:48])
    print(symmetry_response[n_dat,:])
    print(symmetry_labels[n_dat,:])
    
# Transitivity and Equivalence evaluation
eval_pairs=np.array([["A1","C1"],
                      ["A2","C2"],
                      ["A3","C3"],
                      ["A4","C4"]])
#Transitivity 
filt_transitivity_x=np.any(
    [
        [((lbl[0]==pair[0])& #The sample is the first element of the pair
          (lbl[1][0]==pair[1][0])& #The comparator 1 has the same mode (letter) of the second element of the pair
          (lbl[2][0]==pair[1][0])& #The comparator 2 has the same mode (letter) of the second element of the pair
          (lbl[3][0]==pair[1][0])& #The comparator 3 has the same mode (letter) of the second element of the pair
          ((lbl[1]==pair[1])|(lbl[2]==pair[1])|(lbl[3]==pair[1]))&# any of the comparators is the second element of the pair
          ((lbl[1]==pair[1])+(lbl[2]==pair[1])+(lbl[3]==pair[1])==1)# the second element of the pair is presented once in the comparators
         ) for pair in eval_pairs] for lbl in labels],# for any of the pairs on every label
axis=1)

transitivity_values=values_x[filt_transitivity_x]
transitivity_labels=  labels[filt_transitivity_x]

transitivity_response=np.any(np.array([
    [[
        ((lbl[0]==pair[0])&(lbl[1]==pair[1])), #The sample is the first element of the pair and the first  comparator is the second element of the pair
        ((lbl[0]==pair[0])&(lbl[2]==pair[1])), #The sample is the first element of the pair and the second comparator is the second element of the pair
        ((lbl[0]==pair[0])&(lbl[3]==pair[1])) #The sample is the first element of the pair and the third  comparator is the second element of the pair
    ] for pair in eval_pairs] for lbl in transitivity_labels]# for any of the pairs on every label
), axis=1)*1

def view_transitivity():
    n_dat=random.randrange(len(transitivity_labels))
    print(n_dat)
    print(transitivity_values[n_dat,0:12])
    print(transitivity_values[n_dat,12:24])
    print(transitivity_values[n_dat,24:36])
    print(transitivity_values[n_dat,36:48])
    print(transitivity_response[n_dat,:])
    print(transitivity_labels[n_dat,:])
    
# Equivalence
filt_equivalence=np.any(
    [
        [((lbl[0]==pair[1])& #The sample is the second element of the pair
          (lbl[1][0]==pair[0][0])& #The comparator 1 has the same mode (letter) of the first element of the pair
          (lbl[2][0]==pair[0][0])& #The comparator 2 has the same mode (letter) of the first element of the pair
          (lbl[3][0]==pair[0][0])& #The comparator 3 has the same mode (letter) of the first element of the pair
          ((lbl[1]==pair[0])|(lbl[2]==pair[0])|(lbl[3]==pair[0]))&# any of the comparators is the first element of the pair
          ((lbl[1]==pair[0])+(lbl[2]==pair[0])+(lbl[3]==pair[0])==1)# the first element of the pair is presented once in the comparators
         ) for pair in eval_pairs] for lbl in labels],# for any of the pairs on every label
axis=1)

equivalence_values=values_x[filt_equivalence]
equivalence_labels=  labels[filt_equivalence]

equivalence_response=np.any(np.array([
    [[
        ((lbl[0]==pair[1])&(lbl[1]==pair[0])), #The sample is the second element of the pair and the first  comparator is the first element of the pair
        ((lbl[0]==pair[1])&(lbl[2]==pair[0])), #The sample is the second element of the pair and the second comparator is the first element of the pair
        ((lbl[0]==pair[1])&(lbl[3]==pair[0]))  #The sample is the second element of the pair and the third  comparator is the first element of the pair
    ] for pair in eval_pairs] for lbl in equivalence_labels]# for any of the pairs on every label
), axis=1)*1

def view_equivalence():
    n_dat=random.randrange(len(equivalence_labels))
    print(n_dat)
    print(equivalence_values[n_dat,0:12])
    print(equivalence_values[n_dat,12:24])
    print(equivalence_values[n_dat,24:36])
    print(equivalence_values[n_dat,36:48])
    print(equivalence_response[n_dat,:])
    print(equivalence_labels[n_dat,:])
    

