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

