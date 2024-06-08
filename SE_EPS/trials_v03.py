import numpy as np
import pandas as pd
import string
import itertools

def get_stimuli_list(members_n, classes_n):
    members_list = list(string.ascii_uppercase)[:members_n]
    class_list = [str(class_n+1) for class_n in range(classes_n)]
    stimuli_list = [let+str(num) for let in members_list for num in class_list]
    dummy_list=["Z_"+str(dmm+11) for dmm in range(members_n*(classes_n-1))]
    return members_list, class_list, stimuli_list, dummy_list

def create_pair_members(members_list, train_structure):
    
    pairs_dict = {
        "LS": [[members_list[ith], members_list[ith+1]] for ith in range(len(members_list)-1)],
        "OTM": [[members_list[0], members_list[ith+1]] for ith in range(len(members_list)-1)],
        "MTO": [[members_list[ith+1], members_list[0]] for ith in range(len(members_list)-1)]
    }

    train_pairs = pairs_dict.get(train_structure, [])

    reflexiv_pairs = [[stm, stm] for stm in members_list]
    symmetry_pairs = [[tr_pr[1], tr_pr[0]] for tr_pr in train_pairs]
    full_pairs = np.array([[stm_1, stm_2] for stm_1 in members_list for stm_2 in members_list])

    def evaluate_pair_membership(large_pairs, subset_pair):
        return np.array([(np.isin(large_pairs[:,0], sbst_pair[0]))&(np.isin(large_pairs[:,1], sbst_pair[1])) for sbst_pair in subset_pair]).sum(0)>0

    pair_train_in_full = evaluate_pair_membership(full_pairs, train_pairs)
    pair_reflexiv_in_full = evaluate_pair_membership(full_pairs, reflexiv_pairs)
    pair_symmetry_in_full = evaluate_pair_membership(full_pairs, symmetry_pairs)

    transitivity_pairs = np.array(full_pairs)[~(pair_train_in_full | pair_reflexiv_in_full | pair_symmetry_in_full)]

    pairs_members_df = pd.concat([
        pd.DataFrame(train_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="baseline"),
        pd.DataFrame(reflexiv_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="reflexivity"),
        pd.DataFrame(symmetry_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="symmetry"),
        pd.DataFrame(transitivity_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="transitivity")
    ], ignore_index=True, sort=False)

    return pairs_members_df

def create_pairs_classes(pairs_members_df, class_list):
    pairs_classes_arr = [[pair_in[0]+str(class_n),pair_in[1]+str(class_n), pair_in[2]] for pair_in in np.array(pairs_members_df) for class_n in class_list]
    pairs_classes_df=pd.DataFrame(pairs_classes_arr, columns=pairs_members_df.columns)
    return pairs_classes_df

def create_trials(subset_to_trials, pairs_dataset_df, stimuli_list, dummy_list, relation_type):
    ## Filter the dataframe based on the subset_to_trials
    trials_pairs_subset = pairs_dataset_df[pairs_dataset_df.pair_subset == subset_to_trials]
    trials_info_list = []
    
    # Iterate over the filtered dataframe
    for _, row in trials_pairs_subset.iterrows():
        st_sample = row["st_sample"]
        st_comparison = row["st_comparison"]
        trial_group = row["pair_subset"]
        sample_class = st_sample[1]
        relation_dict = {
            "select_only": dummy_list,
            "select_reject": [stim for stim in stimuli_list if not(stim[1] == sample_class)],
            "reject_only": [stim for stim in stimuli_set if not(stim[1] == sample_class)],
        }    
        # Filter the stimulus 
        comparison_list = relation_dict.get(relation_type, [])
    
        # Create a list of combinations
        combs_comps = itertools.permutations([''.join(stim_filt) for stim_filt in comparison_list], 2)
    
        # Iterate over the combinations
        for cmb_cmprs_arr in combs_comps:
            trials_info_list.extend([
                [trial_group, st_sample, st_comparison, cmb_cmprs_arr[0], cmb_cmprs_arr[1], "O_1", st_comparison],
                [trial_group, st_sample, cmb_cmprs_arr[0], st_comparison, cmb_cmprs_arr[1], "O_2", st_comparison],
                [trial_group, st_sample, cmb_cmprs_arr[0], cmb_cmprs_arr[1], st_comparison, "O_3", st_comparison]
            ])
    # Create a DataFrame from the list
    subset_trials_info_df = pd.DataFrame(trials_info_list, columns=["sample_subset", "st_sample", "st_comp1", "st_comp2", "st_comp3", "option_answer", "st_comparison"])
    if relation_type == "reject_only":
        # replace sample with dummy to allow reject relations in comparisons
        sample_stims = trials_pairs_subset['st_sample']
        sample_replace = dict(zip(sample_stims, np.random.choice(dummy_list, len(sample_stims), replace=False)))
        subset_trials_info_df['st_sample'] = subset_trials_info_df['st_sample'].replace(sample_replace) 

    return subset_trials_info_df

def encode_stims (stimuli_list, dummy_list):
    stim_keys = stimuli_list+dummy_list
    np.random.shuffle(stim_keys)
    stims_code = [[1 if i == j else 0 for j in range(len(stim_keys))] for i in range(len(stim_keys))]
    stims_random = dict(zip(stim_keys, stims_code))
    stims_dict = {i: stims_random[i] for i in stims_random.keys()}
    return stims_dict

def process_trial_values(trial_info_df, stims_dict):
    options = {"O_1": [1, 0, 0],
               "O_2": [0, 1, 0],
               "O_3": [0, 0, 1]}

    def process_row(row):
        trial_embedding = [bit for stml in [row["st_sample"], row["st_comp1"], row["st_comp2"], row["st_comp3"]] for bit in stims_dict[stml]]
        return trial_embedding, options[row["option_answer"]]

    trial_values_list, trial_answers_list = zip(*trial_info_df.apply(process_row, axis=1))
    return np.array(trial_values_list), np.array(trial_answers_list)

def decode_trial(encoded_trials_set, trial_index, stims_dict):
    reshaped_trial = encoded_trials_set[trial_index].reshape(4, -1)
    decoded_trial = []

    for i in range(4):
        random_trial = reshaped_trial[i]
        decoded_stim = next(key_stim for key_stim, encoded_stim in stims_dict.items() if encoded_stim == list(random_trial))
        decoded_trial.append(decoded_stim)

    return decoded_trial
