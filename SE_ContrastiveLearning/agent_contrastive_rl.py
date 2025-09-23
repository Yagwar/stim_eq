import numpy as np
import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt
from scipy.special import softmax

rng = np.random.default_rng()
options_dict = {'O_1': 0, 'O_2': 1, 'O_3': 2} 
######## Get trial
def get_trial_data(trials_subset_df, episode_trial_index, embedded_stimuli_dict, comps_columns = ["st_comp1","st_comp2","st_comp3"]):
    trial_info = trials_subset_df.iloc[episode_trial_index]
    anchor_key = trial_info.st_sample
    anchor = embedded_stimuli_dict[trial_info.st_sample]
    options_keys = [trial_info[cmp_colum]  for cmp_colum in comps_columns]
    options = [embedded_stimuli_dict[trial_info[cmp_colum]] for cmp_colum in comps_columns]
    positive_index = options_dict[trial_info.option_answer]
    positive_option = trial_info.st_comparison
    return anchor, options, anchor_key, options_keys, positive_index, positive_option

######## Compute Q values
# Compute Q-values for the options using dot product
def compute_q_values(anchor, options, temperature):
    q_values = [dot_product(anchor, option) / temperature for option in options]
    return q_values

######## Select option (action) epsilon
def choose_action (q_values, epsilon):
    # Choose action based on epsilon-greedy policy
    if rng.random() < epsilon:
        # Exploration: choose a random action
        chosen_action = np.random.randint(len(q_values))
    else:
        # Exploitation: choose the action with the highest Q-value
        chosen_action = np.argmax(q_values)
    return chosen_action

######## get response score (reward if correct positive option)
def get_response_score(chosen_action, options, positive_index):
    # Get the chosen option and whether it is the positive example
    chosen_option = options[chosen_action]
    is_positive = (chosen_action == positive_index)
    return chosen_option, is_positive

######## Calculate InfoNCE loss
def dot_product(vec1, vec2):
    """
    Compute the dot product of two vectors.

    Args:
    - vec1 (numpy array): First vector.
    - vec2 (numpy array): Second vector.

    Returns:
    - dot_product (float): Dot product of vec1 and vec2.
    """
    return np.dot(vec1, vec2)

## InfoNCE loss function using log-sum-exp trick for numerical stability
def info_nce_loss(anchor, chosen_option, is_positive, temperature=0.1):
    """
    Calculate the InfoNCE loss between the anchor and the chosen option.

    Args:
    - anchor (numpy array): The anchor embedding.
    - chosen_option (numpy array): The chosen option embedding.
    - is_positive (bool): Whether the chosen option is the positive example.
    - temperature (float): Temperature parameter for scaling.

    Returns:
    - loss (float): Calculated InfoNCE loss.
    """
    # Compute the dot product of anchor and chosen_option, scaled by temperature
    chosen_score = np.dot(anchor, chosen_option) / temperature
    epsln_value = 1e-10  # Small constant to avoid taking log of zero

    if is_positive:
        # For positive example: maximize similarity
        loss = -np.log(1 / (1 + np.exp(-chosen_score)) + epsln_value)
    else:
        # For negative example: minimize similarity
        loss = -np.log(1 - (1 / (1 + np.exp(-chosen_score))) + epsln_value)

    return loss  # Return the calculated loss

######## Compute Gradients
### Gradient computation for InfoNCE loss

def _calculate_gradients(anchor, target_option, is_positive_for_grad, temperature=0.1):
    score = np.dot(anchor, target_option) / temperature
    sigmoid_score = 1 / (1 + np.exp(-score))

    if is_positive_for_grad:
        d_loss_d_score = sigmoid_score - 1 # Pushes score lower, so that 1/(1+exp(-score)) becomes 1 (positive loss)
    else:
        d_loss_d_score = sigmoid_score # Pushes score higher, so that 1/(1+exp(-score)) becomes 0 (negative loss)

    anchor_grad = d_loss_d_score * (1 / temperature) * target_option
    target_option_grad = d_loss_d_score * (1 / temperature) * anchor

    return anchor_grad, target_option_grad


######## Update Embeddings
def update_embeddings(embedded_stimuli_dict, anchor_key, anchor_updated, options_keys, chosen_action, chosen_option_updated, is_positive, only_correct_choice):
    if only_correct_choice:
        if is_positive:
            embedded_stimuli_dict[anchor_key] = anchor_updated
            options_keys[chosen_action] = chosen_option_updated
    else:
        embedded_stimuli_dict[anchor_key] = anchor_updated
        options_keys[chosen_action] = chosen_option_updated  


def train_q_network(
    trials_info_df,
    embedded_stimuli_dict,
    n_epochs = 1,
    learning_rate = .025, #.05
    temperature = .1,
    epsilon = .3,
    known_negative_gradient_ratio = 1, # default 1 for full negative update # unselected options on correct choice (known negatives)
    unknown_positive_gradient_ratio=.1, # unselected options on incorrect choice are pulled this scale in update gradients.
    positive_sensitivity = 1, #default (1) #
    negative_sensitivity = 1, #default (1)
):
    epoch_losses = []
    epsilon_val = epsilon

    for epoch in range(n_epochs):
        index_list = np.arange(trials_info_df.shape[0])
        rng.shuffle(index_list)

        for epsd_trial_index in index_list:
            # 1. Get trial data (initial states for the current trial)
            initial_anchor, initial_options, anchor_key, options_keys, positive_index, positive_option = get_trial_data(
                trials_subset_df=trials_info_df,
                episode_trial_index=epsd_trial_index,
                embedded_stimuli_dict=embedded_stimuli_dict
            )

            # --- Agent's Decision Process (Q-values and Action Choice) ---
            q_values = compute_q_values(initial_anchor, initial_options, temperature)
            chosen_action = choose_action(q_values, epsilon_val)
            
            chosen_option_embedding_actual = initial_options[chosen_action]
            is_agent_choice_correct = (chosen_action == positive_index)

            # Calculate InfoNCE loss for tracking
            loss = info_nce_loss(initial_anchor, chosen_option_embedding_actual, is_agent_choice_correct, temperature)
            epoch_losses.append(loss)

            # --- Gradient Aggregation and Update Process based on Agent's Choice ---
            anchor_accumulated_grad = np.zeros_like(initial_anchor)
            temp_updated_option_embeddings = {key: val.copy() for key, val in zip(options_keys, initial_options)}

            # SCENARIO 1: AGENT SELECTED THE CORRECT OPTION (is_agent_choice_correct == True)
            if is_agent_choice_correct:
                # Update anchor (sample) and positive example (correct comparison)
                true_positive_option_embedding = initial_options[positive_index]

                current_anchor_grad, current_option_grad = _calculate_gradients(
                    initial_anchor,
                    true_positive_option_embedding,
                    is_positive_for_grad=True,
                    temperature=temperature
                )
                # The chosen correct option contributes its full gradient to the anchor
                positive_scaled_lr = learning_rate * positive_sensitivity
                
                anchor_accumulated_grad += current_anchor_grad * positive_sensitivity
                temp_updated_option_embeddings[options_keys[positive_index]] -= positive_scaled_lr * current_option_grad

                # Process unselected options as negative examples (ground truth negatives)
                for i, current_option_embedding_from_initial in enumerate(initial_options):
                    if i != positive_index: # This option is an unselected negative
                        option_key = options_keys[i]
                        neg_anchor_grad, neg_option_grad = _calculate_gradients(
                            initial_anchor,
                            current_option_embedding_from_initial,
                            is_positive_for_grad=False, # This is a true negative pair
                            temperature=temperature
                        )
                        negative_scaler =  min(negative_sensitivity, known_negative_gradient_ratio)
                        scaled_negative_option_lr = learning_rate * negative_scaler
                        anchor_accumulated_grad += (neg_anchor_grad * negative_scaler)
                        temp_updated_option_embeddings[option_key] -= scaled_negative_option_lr * neg_option_grad

            # SCENARIO 2: AGENT SELECTED THE WRONG OPTION (is_agent_choice_correct == False)
            else:
                # Update anchor (sample) and chosen negative example (incorrect selected comparison)
                chosen_negative_option_embedding = initial_options[chosen_action]
                chosen_negative_option_key = options_keys[chosen_action]

                # Push ANCHOR and CHOSEN NEGATIVE apart with full learning rate
                neg_anchor_grad, neg_option_grad = _calculate_gradients(
                    initial_anchor,
                    chosen_negative_option_embedding,
                    is_positive_for_grad=False,
                    temperature=temperature
                )
                # The chosen incorrect option contributes its full gradient to the anchor
                ## incorrect selected option scaler
                negative_scaled_lr = learning_rate * negative_sensitivity
                
                anchor_accumulated_grad += neg_anchor_grad * negative_sensitivity
                temp_updated_option_embeddings[chosen_negative_option_key] -= negative_scaled_lr * neg_option_grad

                # Process unselected options: Both treated as "possible positive examples"
                for i, current_option_embedding_from_initial in enumerate(initial_options):
                    if i != chosen_action: # This option was unselected by the agent
                        option_key = options_keys[i]

                        # Apply a "slightly positive gradient" (a pull) to both unselected options.
                        unsel_anchor_grad, unsel_option_grad = _calculate_gradients(
                            initial_anchor,
                            current_option_embedding_from_initial,
                            is_positive_for_grad=True, # Treat as positive for gradient calculation direction
                            temperature=temperature
                        )
                        positive_scaler =  min(positive_sensitivity, unknown_positive_gradient_ratio)
                        # Scale *both* the option's gradient and the anchor's corresponding contribution
                        scaled_option_lr = learning_rate * positive_scaler
                        
                        anchor_accumulated_grad += (unsel_anchor_grad * positive_scaler) # Scale anchor's contribution
                        temp_updated_option_embeddings[option_key] -= scaled_option_lr * unsel_option_grad
            
            # --- Apply Aggregated Anchor Update ---
            # The anchor_accumulated_grad already contains scaled contributions from unselected options.
            # So, we just apply the total.
            updated_anchor_embedding = initial_anchor - learning_rate * anchor_accumulated_grad

            # --- Finally, Update the main embedded_stimuli_dict ---
            embedded_stimuli_dict[anchor_key] = updated_anchor_embedding
            for key in options_keys:
                embedded_stimuli_dict[key] = temp_updated_option_embeddings[key]

        # --- End of Epoch Visualization and Output ---
        clear_output(wait=True)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
        plt.plot(range(len(epoch_losses)),
                 epoch_losses,
                 alpha = .8,
                 linewidth=.5
                )
        plt.title ("Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.show()

    return embedded_stimuli_dict

def get_evaluation_responses(test_trials_info_df, trained_embedding_stimuli_dict):
    actions_to_options_dict = {0:'O_1', 1:'O_2', 2:'O_3'}
    index_list = np.arange(test_trials_info_df.shape[0])
    
    option_responses = []
    test_rewards = []
    test_losses = []
    q_values_trials = []
    for epsd_trial_index in index_list:
        ######## Get trial
        anchor, options, anchor_key, options_keys, positive_index, positive_option = get_trial_data(
            trials_subset_df = test_trials_info_df, 
            episode_trial_index = epsd_trial_index, 
            embedded_stimuli_dict = trained_embedding_stimuli_dict
        )
        ## Compute Q values
        q_values = compute_q_values(anchor, options, temperature = 1)
        q_values_trials.append(q_values)
        
        ## Select option (action) epsilon
        chosen_action = choose_action (q_values, epsilon=0)
        response_option = actions_to_options_dict[chosen_action]
        option_responses.append(response_option)
        
        ## get response score (reward if correct positive option)
        chosen_option, is_positive = get_response_score(chosen_action, options, positive_index)
        
        ## Calculate InfoNCE loss
        loss = info_nce_loss(anchor, chosen_option, is_positive, temperature = 1)
       
        test_rewards.append(is_positive*1)
        test_losses.append(loss)
    
    softmax_df = pd.DataFrame(
        softmax(np.array(q_values_trials), axis=1), 
        index = test_trials_info_df.index,
        columns = ["opt1_prob","opt2_prob","opt3_prob"]
    )
    
    response_df = pd.DataFrame(
        option_responses, 
        index = test_trials_info_df.index,
        columns = ["agent_response"]
    )
    
    output_info_df = pd.concat([test_trials_info_df, softmax_df, response_df], axis =1)
    output_info_df["response_score"] = (output_info_df["agent_response"]==output_info_df["option_answer"])*1
    output_info_df['sample_member']=[stim[0] for stim in output_info_df.st_sample]
    output_info_df['comparison_member']=[stim[0] for stim in output_info_df.st_comparison]
    
    return output_info_df
    