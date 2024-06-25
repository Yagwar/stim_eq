import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from time import localtime, strftime, time, gmtime
import networkx as nx
import math
import itertools

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, mean_squared_error

from time import localtime, strftime

import pickle
import sys


################ Create pairs Trainning Structure
classes_n=4
members_n=7#5#
dummies_n=members_n*(classes_n-1) # Se agrega un grupo de N_members para crear la clase dummy
# Minimal Test 5 Members
# train_pairs_input=["AB","BC","CD","DE"]# LS minimal test
# train_pairs_input=["AB","AC","AD","AE"]# OTM minimal test
# train_pairs_input=["BA","CA","DA","EA"]# MTO minimal test

# Full Test 7 members
# train_pairs_input=["AB","BC","CD","DE","EF","FG"]# LS 
# train_pairs_input=["BA","CA","DA","EA","FA","GA"]# MTO 
# train_pairs_input=["AB","AC","AD","AE","AF","AG"]# OTM
train_pairs_input=sys.argv[1].split(',')

print(f'Trainig pairs for the experiment {train_pairs_input}')
print(f'Creating experiment with {members_n} members, {classes_n} classes and {dummies_n} dummy stimulus.')


# members, classes and stimuli
members_list=list(string.ascii_uppercase)[:members_n]
class_list=[class_n+1 for class_n in range(classes_n)]
# class_list.append("Z")
dummy_list=["Z_"+str(dmm+11) for dmm in range(dummies_n)]

# create pairs for members without classes - this will be repeated for stimulus encoding
train_pairs=np.array([[pair_in[0],pair_in[1]] for pair_in in train_pairs_input]) # validar que cada elemento de los pares se encuentre como miembro de clase en la lista de miembros.
reflexiv_pairs=np.array([[stm,stm]for stm in members_list])

symmetry_pairs=np.array([[tr_pr[1],tr_pr[0]]for tr_pr in train_pairs])

full_pairs=np.array([[stm_1,stm_2] for stm_1 in members_list for stm_2 in members_list])
pair_train_in_full=np.array([(np.isin(full_pairs[:,0], trn_pair[0]))&(np.isin(full_pairs[:,1], trn_pair[1])) for trn_pair in train_pairs]).sum(0)
pair_reflexiv_in_full=np.array([(np.isin(full_pairs[:,0], rflx_pair[0]))&(np.isin(full_pairs[:,1], rflx_pair[1])) for rflx_pair in reflexiv_pairs]).sum(0)
pair_symmetry_in_full=np.array([(np.isin(full_pairs[:,0], smtr_pair[0]))&(np.isin(full_pairs[:,1], smtr_pair[1])) for smtr_pair in symmetry_pairs]).sum(0)
transitivity_pairs=full_pairs[(pair_train_in_full+pair_reflexiv_in_full+pair_symmetry_in_full)==0]

# stimulus pairs dataframe
train_pairs_stims=[[pair_in[0]+str(class_n),pair_in[1]+str(class_n)] for pair_in in train_pairs for class_n in class_list]
reflexiv_pairs_stims=[[pair_in[0]+str(class_n),pair_in[1]+str(class_n)] for pair_in in reflexiv_pairs for class_n in class_list]
symmetry_pairs_stims=[[pair_in[0]+str(class_n),pair_in[1]+str(class_n)] for pair_in in symmetry_pairs for class_n in class_list]
transitivity_pairs_stims=[[pair_in[0]+str(class_n),pair_in[1]+str(class_n)] for pair_in in transitivity_pairs for class_n in class_list]

train_df=pd.DataFrame(train_pairs_stims, columns=["st_sample", "st_comparison"])
train_df["pair_subset"]="train"

reflexivity_df=pd.DataFrame(reflexiv_pairs_stims, columns=["st_sample", "st_comparison"])
reflexivity_df["pair_subset"]="reflexivity"

symmetry_df=pd.DataFrame(symmetry_pairs_stims, columns=["st_sample", "st_comparison"])
symmetry_df["pair_subset"]="symmetry"

transitivity_df=pd.DataFrame(transitivity_pairs_stims, columns=["st_sample", "st_comparison"])
transitivity_df["pair_subset"]="transitivity"

pairs_dataset_df=pd.concat([train_df,
                            reflexivity_df,
                            symmetry_df,
                            transitivity_df], 
                           ignore_index=True, sort=False) #output


# Stimulus encoding
stim_keys=[let+str(num) for let in members_list for num in class_list]
stim_keys.extend(dummy_list) #comment on class comparisson stimuli

# randomize stim order
np.random.shuffle(stim_keys)

stims_code=[]
for i in range(len(stim_keys)):
    stm_0=[0]*len(stim_keys)
    stm_0[i]=1
    stims_code.append(stm_0)

stims_random=dict(zip(stim_keys,stims_code))
random_keys = list(stims_random.keys())
random_keys.sort()
stims = {i: stims_random[i] for i in random_keys}

options={"O_1":[1,0,0],#[ 3,-3,-3],#
         "O_2":[0,1,0],#[-3, 3,-3],#
         "O_3":[0,0,1],#[-3,-3, 3],#
         "O_0":[0,0,0],#[-3,-3,-3],#
        }

def create_trials_dummy(subset_to_train):
    # create train trials with dummy comparison
    # subset_to_train="train" # variable
    # dummy_list # info required
    # pairs_dataset_df # info required

    trials_info_list=[]
    combs_dummies=itertools.permutations(dummy_list,2)
    combs_dummies_arr=np.array([np.array(comb_dmmy) for comb_dmmy in list(combs_dummies)])
    train_pairs_subset=pairs_dataset_df[pairs_dataset_df.pair_subset==subset_to_train]

    for pair_index in train_pairs_subset.index: 
        st_sample=train_pairs_subset.loc[pair_index,"st_sample"]
        st_comparison=train_pairs_subset.loc[pair_index,"st_comparison"]
        trial_group=train_pairs_subset.loc[pair_index,"pair_subset"]
        for cmb_dmm_arr in combs_dummies_arr:
            trial_info_complete_o1=[trial_group, st_sample, st_comparison, cmb_dmm_arr[0], cmb_dmm_arr[1], "O_1", st_comparison]
            trials_info_list.append(trial_info_complete_o1)
            trial_info_complete_o2=[trial_group, st_sample, cmb_dmm_arr[0], st_comparison, cmb_dmm_arr[1], "O_2", st_comparison]
            trials_info_list.append(trial_info_complete_o2)
            trial_info_complete_o3=[trial_group, st_sample, cmb_dmm_arr[0], cmb_dmm_arr[1], st_comparison, "O_3", st_comparison]
            trials_info_list.append(trial_info_complete_o3)

    train_trials_info_df=pd.DataFrame(np.array(trials_info_list), columns=["sample_subset",
                                             "st_sample","st_comp1","st_comp2","st_comp3",
                                             "option_answer","st_comparison"])
    return train_trials_info_df



def create_trials_comparison(subset_to_trials):
    # create train trials with class comparison
    trials_pairs_subset=pairs_dataset_df[pairs_dataset_df.pair_subset==subset_to_trials]
    stims_array=np.array([list(stm_ky) for stm_ky in stim_keys if len(stm_ky)==2])
    trials_info_list=[]

    for pair_index in trials_pairs_subset.index: 
        st_sample=trials_pairs_subset.loc[pair_index,"st_sample"]
        st_comparison=trials_pairs_subset.loc[pair_index,"st_comparison"]
        trial_group=trials_pairs_subset.loc[pair_index,"pair_subset"]
        sample_class=st_sample[1]
        dummy_signal="Z"
        stims_array_filt=stims_array[[not(stim[1]==sample_class)|(stim[0]==dummy_signal) for stim in stims_array]]#stims_array[~((stims_array[:,1]==sample_class)|(stims_array[:,0]==dummy_signal))]
        comps_list=[''.join(stim_filt) for stim_filt in stims_array_filt]
        combs_comps=itertools.permutations(comps_list,2)
        combs_comps_arr=np.array([np.array(comb_comprs) for comb_comprs in list(combs_comps)])
        for cmb_cmprs_arr in combs_comps_arr:
            trial_info_complete_o1=[trial_group, st_sample, st_comparison, cmb_cmprs_arr[0], cmb_cmprs_arr[1], "O_1", st_comparison]
            trials_info_list.append(trial_info_complete_o1)
            trial_info_complete_o2=[trial_group, st_sample, cmb_cmprs_arr[0], st_comparison, cmb_cmprs_arr[1], "O_2", st_comparison]
            trials_info_list.append(trial_info_complete_o2)
            trial_info_complete_o3=[trial_group, st_sample, cmb_cmprs_arr[0], cmb_cmprs_arr[1], st_comparison, "O_3", st_comparison]
            trials_info_list.append(trial_info_complete_o3)

    subset_trials_info_df=pd.DataFrame(np.array(trials_info_list), columns=["sample_subset",
                                             "st_sample","st_comp1","st_comp2","st_comp3",
                                             "option_answer","st_comparison"])
    return subset_trials_info_df



train_dummy_info=create_trials_dummy("train")
reflexivity_dummy_info=create_trials_dummy("reflexivity")

train_info=create_trials_comparison("train")
reflexivity_info=create_trials_comparison("reflexivity")
symmetry_info=create_trials_comparison("symmetry")
transitivity_info=create_trials_comparison("transitivity")



def process_trial_values(trial_info_df, stims_dict):
    trial_values_list=[]
    trial_answers_list=[]
    for i_trial in trial_info_df.index:
        st_sample=trial_info_df.loc[i_trial,"st_sample"]
        st_comp1=trial_info_df.loc[i_trial,"st_comp1"]
        st_comp2=trial_info_df.loc[i_trial,"st_comp2"]
        st_comp3=trial_info_df.loc[i_trial,"st_comp3"]
        option_answer=trial_info_df.loc[i_trial,"option_answer"]

        trial_embedding=[bit_emb for stml in [st_sample,st_comp1,st_comp2,st_comp3] for bit_emb in stims_dict[stml]]
        trial_values_list.append(trial_embedding)
        trial_answers_list.append(options[option_answer])
    return np.array(trial_values_list), np.array(trial_answers_list)


train_dummy_values, train_dummy_answers = process_trial_values(train_dummy_info, stims) # comment when train with class members comparissons
train_values, train_answers = process_trial_values(train_info, stims)
reflexivity_dummy_values, reflexivity_dummy_answers = process_trial_values(reflexivity_dummy_info, stims) # comment when train with class members comparissons
reflexivity_values, reflexivity_answers = process_trial_values(reflexivity_info, stims)
symmetry_values, symmetry_answers = process_trial_values(symmetry_info, stims)
transitivity_values, transitivity_answers = process_trial_values(transitivity_info, stims)

################ Experiment info dict
exper_info={'train_dummy':{"values":train_dummy_values,"answers":train_dummy_answers,"info":train_dummy_info}, # comment when train with class members comparissons
            'train':{"values":train_values,"answers":train_answers,"info":train_info},
            'reflexivity_dummy':{"values":reflexivity_dummy_values,"answers":reflexivity_dummy_answers,"info":reflexivity_dummy_info},  # comment when train with class members comparissons
            'reflexivity':{"values":reflexivity_values,"answers":reflexivity_answers,"info":reflexivity_info},  
            'symmetry':{"values":symmetry_values,"answers":symmetry_answers,"info":symmetry_info},  
            'transitivity':{"values":transitivity_values,"answers":transitivity_answers,"info":transitivity_info}
           }
print("Experiment trials created!")