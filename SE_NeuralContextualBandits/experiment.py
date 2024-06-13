import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.data import TensorDataset, DataLoader

# %matplotlib inline
import time
import pylab as pl
from IPython import display

import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
pd.options.display.float_format = "{:,.3f}".format

from IPython.display import clear_output

# Check if CUDA is available and set PyTorch to use the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import numpy as np
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim

##### Agent Environment
class Bandit_multi:
    def __init__(self, X, y, is_shuffle=True, seed=None):
        # No need to fetch data, use provided arrays
        self.X = X
        self.y = y
    
        # Shuffle data if needed
        if is_shuffle:
            self.X, self.y = shuffle(X, y, random_state=seed)
    
        # Cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = self.y.shape[1]  # Number of arms based on one-hot encoding shape
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]
    
    def step(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim + self.act_dim] = self.X[self.cursor]
        arm = np.argmax(self.y[self.cursor])  # Get index of the active arm
        rwd = self.y[self.cursor].copy()  # Reward based on one-hot encoded response
        self.cursor += 1
        return X, rwd
    
    def finish(self):
        return self.cursor == self.size
    
    def reset(self):
        self.cursor = 0

##### Neural UCB agent
class Network(nn.Module):
    def __init__(self, dim, hidden_size=50000):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class NeuralUCBDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=50000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,), device=device)
        self.nu = nu
        self.device = device
    
    def select(self, context):
        # tensor = torch.from_numpy(context).float().to(self.device)
        # mu = self.func(tensor)
        mu = self.func(context)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            
            sample_r = fx.item() + sigma.item()
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
    
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew

    def train(self, context, reward):
        self.context_list.append(context.to(self.device))  
        #self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float().to(self.device))
        self.reward.append(reward)
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0

        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.to(self.device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
                if batch_loss / length <= 1e-3:
                    return batch_loss / length

def run_evaluation(agent, test_trials, test_answers):
    test_environment = Bandit_multi(test_trials, test_answers, is_shuffle=False)
    selected_arm_list = []
    reward_selection_list = []
    agent_response_list = []
    
    # with torch.no_grad():
    for t in range(test_environment.size): # All trials must pass
        # Get context and reward from Bandit environment
        context, rwd = test_environment.step()
        context = torch.from_numpy(context).float().to(device)  # Move context to device
        
        # Select arm, norm, sigma, and average reward
        arm_select, _, _, _ = agent.select(context)
        # arm_select
        action_reward = rwd[arm_select]
        
        # Create one-hot encoded vector for the selected arm
        arm_one_hot = np.zeros(test_environment.n_arm)
        arm_one_hot[arm_select] = 1
        agent_response = arm_one_hot
    
        selected_arm_list.append(arm_select)
        reward_selection_list.append(action_reward)
        agent_response_list.append(agent_response)
            
    return selected_arm_list, reward_selection_list#, np.array(agent_response_list)

