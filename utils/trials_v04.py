import numpy as np
import pandas as pd
import string
import itertools

rng = np.random.default_rng()

class TrialsGenerator:
    def __init__(self, 
                 members_n, 
                 classes_n, 
                 train_structure="LS",
                 relation_type="select_reject",
                 same_label_filter=True
                ):
        self.members_n = members_n
        self.classes_n = classes_n

        # Validate train_structure
        if train_structure not in ["LS", "MTO", "OTM"]:
            raise ValueError("Invalid train_structure. Must be one of 'LS', 'MTO', or 'OTM'.")

        self.train_structure = train_structure
        
        if relation_type not in ["select_only", "select_reject", "reject_only"]:
            raise ValueError(f"Invalid relation_type: {relation_type}. Must be one of 'select_only', 'select_reject', or 'reject_only'.")
        
        self.relation_type = relation_type

        self.same_label_filter = same_label_filter
        
        self.members_list, self.class_list, self.stimuli_list, self.dummy_list = self.get_stimuli_list()
        self.member_pairs_df = self.create_pair_members()
        self.experimental_pairs = self.create_pairs_classes(self.member_pairs_df)
        
        self.baseline_trials_info = self.create_trials(subset_to_trials="baseline")
        self.reflexivity_trials_info = self.create_trials(subset_to_trials="reflexivity")
        self.symmetry_trials_info = self.create_trials(subset_to_trials="symmetry")
        self.transitivity_trials_info = self.create_trials(subset_to_trials="transitivity")
        

    def get_stimuli_list(self):
        members_list = list(string.ascii_uppercase)[:self.members_n]
        class_list = [str(class_n+1) for class_n in range(self.classes_n)]
        stimuli_list = [let+str(num) for let in members_list for num in class_list]
        dummy_list=["Z_"+str(dmm+11) for dmm in range(self.members_n*self.classes_n)]
        return members_list, class_list, stimuli_list, dummy_list

    def evaluate_pair_membership(self, large_pairs, subset_pair):
        return np.array([(np.isin(large_pairs[:,0], sbst_pair[0]))&(np.isin(large_pairs[:,1], sbst_pair[1])) for sbst_pair in subset_pair]).sum(0)>0

    def create_pair_members(self):
        pairs_dict = {
            "LS": [[self.members_list[ith], self.members_list[ith+1]] for ith in range(len(self.members_list)-1)],
            "OTM": [[self.members_list[0], self.members_list[ith+1]] for ith in range(len(self.members_list)-1)],
            "MTO": [[self.members_list[ith+1], self.members_list[0]] for ith in range(len(self.members_list)-1)]
        }

        train_pairs = pairs_dict.get(self.train_structure, [])

        reflexiv_pairs = [[stm, stm] for stm in self.members_list]
        symmetry_pairs = [[tr_pr[1], tr_pr[0]] for tr_pr in train_pairs]
        full_pairs = np.array([[stm_1, stm_2] for stm_1 in self.members_list for stm_2 in self.members_list])

        pair_train_in_full = self.evaluate_pair_membership(full_pairs, train_pairs)
        pair_reflexiv_in_full = self.evaluate_pair_membership(full_pairs, reflexiv_pairs)
        pair_symmetry_in_full = self.evaluate_pair_membership(full_pairs, symmetry_pairs)

        transitivity_pairs = np.array(full_pairs)[~(pair_train_in_full | pair_reflexiv_in_full | pair_symmetry_in_full)]

        pairs_members_df = pd.concat([
            pd.DataFrame(train_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="baseline"),
            pd.DataFrame(reflexiv_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="reflexivity"),
            pd.DataFrame(symmetry_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="symmetry"),
            pd.DataFrame(transitivity_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="transitivity")
        ], ignore_index=True, sort=False)

        return pairs_members_df

    def create_pairs_classes(self, pairs_members_df):
        pairs_classes_arr = [[pair_in[0]+str(class_n),pair_in[1]+str(class_n), pair_in[2]] for pair_in in np.array(pairs_members_df) for class_n in self.class_list]
        pairs_classes_df=pd.DataFrame(pairs_classes_arr, columns=pairs_members_df.columns)
        return pairs_classes_df

    def get_dummy_twin_dict(self):
        dummy_twin_dict = dict(zip(self.stimuli_list, self.dummy_list))
        return dummy_twin_dict

    def create_trials(self, subset_to_trials):

        dummy_twin_dict = self.get_dummy_twin_dict()
        ## Filter the dataframe based on the subset_to_trials
        trials_pairs_subset = self.experimental_pairs[self.experimental_pairs.pair_subset == subset_to_trials]
        trials_info_list = []

        # Iterate over the filtered dataframe
        for _, row in trials_pairs_subset.iterrows():
            # create default select - reject pairs
            st_sample = row["st_sample"]
            st_comparison = row["st_comparison"]
            trial_group = row["pair_subset"]
            sample_class = st_sample[1]
            comparison_member = st_comparison[0]

            negative_comparison_list = [stim for stim in self.stimuli_list if not(stim[1] == sample_class)]  # all labels comparisons in select-reject

            # Filter the stimulus to same label comparison
            if self.same_label_filter:
                negative_comparison_list = [stim for stim in self.stimuli_list if ((stim[0] == comparison_member) and not(stim[1] == sample_class))]  # filtered  same label select-reject

            #### replace stimulus from relation type
            if self.relation_type == 'select_only':
                negative_comparison_list = [dummy_twin_dict[stim] for stim in negative_comparison_list]

            if self.relation_type == 'reject_only':
                st_comparison = dummy_twin_dict[st_comparison]

            # Create a list of combinations
            combs_comps = itertools.permutations([''.join(stim_filt) for stim_filt in negative_comparison_list], 2)

            # Iterate over the combinations
            for cmb_cmprs_arr in combs_comps:
                trials_info_list.extend([
                    [trial_group, st_sample, st_comparison, cmb_cmprs_arr[0], cmb_cmprs_arr[1], "O_1", st_comparison],
                    [trial_group, st_sample, cmb_cmprs_arr[0], st_comparison, cmb_cmprs_arr[1], "O_2", st_comparison],
                    [trial_group, st_sample, cmb_cmprs_arr[0], cmb_cmprs_arr[1], st_comparison, "O_3", st_comparison]
                ])

        # Create a DataFrame from the list
        subset_trials_info_df = pd.DataFrame(trials_info_list, columns=["sample_subset", "st_sample", "st_comp1", "st_comp2", "st_comp3", "option_answer", "st_comparison"])

        return subset_trials_info_df
