import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    """ Actor (Policy) Model"""
    
    def __init__(self, state_size, action_size, hidden_size = 256):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

class Critic(nn.Module):
    """ Critic (Value) Model """
    
    def __init__(self, state_size, value_size, hidden_size = 256):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, value_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        device = torch.device("cpu")

        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, 1, hidden_size)
        
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)
        
    def forward(self, states):
        obs = torch.FloatTensor(states)
        
        mu    = self.actor(obs)
        values = self.critic(obs)
        
        std   = self.log_std.exp().expand_as(mu)
        dist  = torch.distributions.Normal(mu, std)
        
        return dist, values