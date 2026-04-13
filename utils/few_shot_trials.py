import numpy as np
import pandas as pd
import string
import itertools
import random
import warnings

class TrialsGenerator:
    """Generates Stimulus Equivalence trials with sterile CVDC tokens."""
    
    def __init__(self, members_n=4, classes_n=3, train_structure="LS"):
        # Flexible Minimal Validation
        if classes_n < 3:
            warnings.warn(f"A 3-comparison task requires at least 3 classes. Got {classes_n}. Setting to 3.")
            classes_n = 3
            
        if members_n < 3:
            warnings.warn(f"Equivalence relations require at least 3 members. Got {members_n}. Setting to 3.")
            members_n = 3
            
        self.members_n = members_n
        self.classes_n = classes_n
        self.train_structure = train_structure

        self.members_list, self.class_list, self.stimuli_list = self.get_stimuli_list()
        
        # CVDC Translation Layer
        self.cvdc_pool = self._generate_cvdc_tokens(len(self.stimuli_list))
        self.translation_map = dict(zip(self.stimuli_list, self.cvdc_pool))

        self.member_pairs_df = self.create_pair_members()
        self.experimental_pairs = self.create_pairs_classes(self.member_pairs_df)

        # Generate Core DataFrames
        self.baseline_trials_df = self.create_trials("baseline")
        self.reflexivity_trials_df = self.create_trials("reflexivity")
        self.symmetry_trials_df = self.create_trials("symmetry")
        self.transitivity_trials_df = self.create_trials("transitivity")
    
    def _generate_cvdc_tokens(self, n: int) -> list:
        consonants = "BCDFGHJKLMNPQRSTVWXYZ" 
        vowels = "AEIOU"                   
        digits = "0123456789"              
        
        base_first_char_pool = list(consonants + vowels + digits)
        
        if n > len(base_first_char_pool):
            warnings.warn(f"Requested {n} tokens exceeds unique first-character pool. Repetition will occur.")
            multiplier = (n // len(base_first_char_pool)) + 1
            first_char_pool = base_first_char_pool * multiplier
        else:
            first_char_pool = base_first_char_pool.copy()
            
        random.shuffle(first_char_pool)
        tokens = set()
        
        while len(tokens) < n:
            if not first_char_pool:
                first_char_pool = base_first_char_pool.copy()
                random.shuffle(first_char_pool)
                
            char1 = first_char_pool.pop()
            char2 = random.choice(consonants)
            char3 = random.choice(vowels)
            char4 = random.choice(digits)
            
            tokens.add(char1 + char2 + char3 + char4)
            
        return list(tokens)
    
    def get_stimuli_list(self):
        members_list = list(string.ascii_uppercase)[:self.members_n]
        class_list = [str(class_n+1) for class_n in range(self.classes_n)]
        stimuli_list = [let+str(num) for let in members_list for num in class_list]
        return members_list, class_list, stimuli_list

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

        transitivity_pairs = full_pairs[~(pair_train_in_full | pair_reflexiv_in_full | pair_symmetry_in_full)]

        return pd.concat([
            pd.DataFrame(train_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="baseline"),
            pd.DataFrame(reflexiv_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="reflexivity"),
            pd.DataFrame(symmetry_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="symmetry"),
            pd.DataFrame(transitivity_pairs, columns=["st_sample", "st_comparison"]).assign(pair_subset="transitivity")
        ], ignore_index=True, sort=False)

    def create_pairs_classes(self, pairs_members_df):
        pairs_classes_arr = [[pair_in[0]+str(class_n), pair_in[1]+str(class_n), pair_in[2]] for pair_in in np.array(pairs_members_df) for class_n in self.class_list]
        return pd.DataFrame(pairs_classes_arr, columns=pairs_members_df.columns)

    def create_trials(self, subset_to_trials):
        trials_pairs_subset = self.experimental_pairs[self.experimental_pairs.pair_subset == subset_to_trials]
        trials_info_list = []

        for _, row in trials_pairs_subset.iterrows():
            st_sample, st_comparison, trial_group = row["st_sample"], row["st_comparison"], row["pair_subset"]
            sample_class, comparison_member = st_sample[1], st_comparison[0]

            negative_comparison_list = [
                stim for stim in self.stimuli_list 
                if (stim[0] == comparison_member) and (stim[1] != sample_class)
            ]

            combs_comps = itertools.permutations(negative_comparison_list, 2)

            for cmb in combs_comps:
                trials_info_list.extend([
                    [trial_group, st_sample, st_comparison, cmb[0], cmb[1], "O_1", st_comparison],
                    [trial_group, st_sample, cmb[0], st_comparison, cmb[1], "O_2", st_comparison],
                    [trial_group, st_sample, cmb[0], cmb[1], st_comparison, "O_3", st_comparison]
                ])

        df = pd.DataFrame(trials_info_list, columns=["sample_subset", "st_sample", "st_comp1", "st_comp2", "st_comp3", "option_answer", "st_comparison"])
        
        # TRANSLATION: Map internal logic directly to CVDC
        for col in ["st_sample", "st_comp1", "st_comp2", "st_comp3", "st_comparison"]:
            df[col] = df[col].map(self.translation_map)
        return df


class EvaluationDatasetBuilder:
    """Builds the few-shot contexts and final evaluation dataframes from a TrialsGenerator."""
    
    def __init__(self, generator: TrialsGenerator):
        self.generator = generator

    def get_few_shot_examples(self):
        """Selects exactly one trial per baseline pair with balanced answers (O_1, O_2, O_3)."""
        baseline_df = self.generator.baseline_trials_df
        unique_pairs = baseline_df[['st_sample', 'st_comparison']].drop_duplicates()
        n_pairs = len(unique_pairs)
        
        options_pool = ['O_1', 'O_2', 'O_3'] * ((n_pairs // 3) + 1)
        options_pool = options_pool[:n_pairs] 
        random.shuffle(options_pool) 
        
        sampled_rows = []
        for i, (_, row) in enumerate(unique_pairs.iterrows()):
            st_sample = row['st_sample']
            st_comp = row['st_comparison']
            target_option = options_pool[i]
            
            subset = baseline_df[
                (baseline_df['st_sample'] == st_sample) & 
                (baseline_df['st_comparison'] == st_comp) &
                (baseline_df['option_answer'] == target_option)
            ]
            
            sampled_row = subset.sample(n=1)
            sampled_rows.append(sampled_row)
            
        return pd.concat(sampled_rows).sample(frac=1).reset_index(drop=True)

    def format_trial_string(self, row, is_test=False):
        # Use neutral interface terms instead of logical or behavioral jargon
        base_text = (f"Given Symbol: '{row['st_sample']}'. "
                     f"Options: [O_1: '{row['st_comp1']}', O_2: '{row['st_comp2']}', O_3: '{row['st_comp3']}'].")
        
        if is_test:
            # The prompt for the zero-shot evaluation trial
            return base_text + "\nTask: Which option is the correct selection?"
        else:
            # The prompt for the few-shot training context
            return base_text + f"\nCorrect Selection: '{row['option_answer']}'."

    def build_evaluation_dataset(self) -> pd.DataFrame:
        """Combines Reflexivity, Symmetry, and Transitivity into a final master dataset."""
        test_df = pd.concat([
            self.generator.reflexivity_trials_df,
            self.generator.symmetry_trials_df,
            self.generator.transitivity_trials_df
        ], ignore_index=True)
        
        def generate_balanced_context():
            sampled_df = self.get_few_shot_examples()
            return "\n".join([
                f"Example {i+1}: {self.format_trial_string(row)}" 
                for i, (_, row) in enumerate(sampled_df.iterrows())
            ])

        test_df['few_shot_context'] = test_df.apply(lambda _: generate_balanced_context(), axis=1)
        test_df['formatted_test_trial'] = test_df.apply(lambda r: self.format_trial_string(r, is_test=True), axis=1)
        
        return test_df
