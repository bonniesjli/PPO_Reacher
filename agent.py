import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import ActorCritic

device = torch.device("cpu")

BATCH_SIZE = 32    # minibatch size
GAMMA = 0.99       # discount rate
TAU = 0.95         # tau

GRADIENT_CLIP = 5   # gradient clip
NUM_EPOCHS = 10     # optimization epochs
CLIP = 0.2          # PPO clip 

BETA = 0.01         # entropy coefficient 
LR = 3e-4           # Adam learning rate
EPSILON = 1e-5      # Adam epsilon

class Agent(object):
    """Interacts and learns from the environment"""
    
    def __init__(self, num_agents, state_size, action_size):
        """ Initialize an Agent object 
        
        Params
        ======
            num_agent (int): number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.num_agents = num_agents
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.model = ActorCritic(state_size, action_size, 256)
        self.optimizer = optim.Adam(self.model.parameters(), LR, eps = EPSILON)
        

    def step(self, rollout, num_agents):
        """ Compute advantage estimates at each time steps given a trajectory"""
        
        storage = [None]* (len(rollout) - 1)
        
        shape = (num_agents, 1)
        advantage = torch.Tensor(np.zeros(shape))
        
        for i in reversed(range(len(rollout)-1)):
            # rollout --> tuple ( s, a, p(a|s), r, dones, V(s) ) FOR ALL AGENT
            # rollout --> last row (s, none, none, none, pending_value) FOR ALL AGENT
            state, action, log_prob, reward, done, value = rollout[i]
            
            # last step - next_return = pending_value
            if i == len(rollout) - 2:
                next_return = rollout[i+1][-1]

            state = torch.Tensor(state)
            action = torch.Tensor(action)
            reward = torch.Tensor(reward).unsqueeze(1)
            done = torch.Tensor(done).unsqueeze(1)
            next_value = rollout[i+1][-1]
            
            # G(t) = r + G(t+1)
            g_return = reward + GAMMA * next_return * done
            next_return = g_return
            # g_return = reward + GAMMA * g_return*done
            
            # Compute TD error
            td_error = reward + GAMMA * next_value - value
            # Compute advantages
            advantage = advantage * TAU * GAMMA * done + td_error
            
            # Add (s, a, p(a|s), g, advantage) 
            storage[i] = [state, action, log_prob, g_return, advantage]
        
        state, action, log_prob, g_return, advantage = map(lambda x: torch.cat(x, dim=0), zip(*storage))   
        advantage = (advantage - advantage.mean()) / advantage.std()
        
        # Check dimensions
        # print ("States :", states.size(0), " * ", states.size(1) )
        # print ("Actions :", actions.size(0), " * ", actions.size(1) )
        # print ("Log Prob :", log_prob.size(0), " * ", log_prob.size(1) )
        # print ("Return :", g_return.size(0), " * ", g_return.size(1) )
        # print ("Advantage :", advantage.size(0), " * ", advantage.size(1) )
        
        self.learn(state, action, log_prob, g_return, advantage, num_agents)
               
    def act(self, states):
        """Given state as per current policy model, returns action, log probabilities and estimated state values"""
        dist, values = self.model(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=1, keepdim=True)

        return actions, log_probs, values
    
    def sample(self, states, actions, log_probs, returns, advantages):
        """Randomly sample learning batches from trajectory"""  
        rand_idx = np.random.randint(0, states.size(0), BATCH_SIZE)
        return states[rand_idx, :], actions[rand_idx, :], log_probs[rand_idx, :], returns[rand_idx, :], advantages[rand_idx, :]
        
    def learn(self, states, actions, log_probs_old, returns, advantages, num_agents):
        """ Optimize surrogate loss with policy and value parameters using given learning batches."""

        for _ in range(NUM_EPOCHS):
            for _ in range(states.size(0) // BATCH_SIZE):
                state_samples, action_samples, log_prob_samples, return_samples, advantage_samples = self.sample(states, actions, log_probs_old, returns, advantages)

                dist, values = self.model(state_samples)

                log_probs = dist.log_prob(action_samples)
                log_probs = torch.sum(log_probs, dim=1, keepdim=True)
                entropy = dist.entropy().mean()

                ratio = (log_probs - log_prob_samples).exp()

                # Surrogate Objctive
                obj = ratio * advantage_samples

                # Clipped Surrogate Objective
                obj_clipped = ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * advantage_samples

                # Compute policy loss: L = min[ r(θ), clip ( r(θ), 1-Ɛ, 1+Ɛ )*A ] - β * entropy
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - BETA * entropy

                # Compute value loss: L = ( V(s) - V_t )^2
                value_loss = (return_samples - values).pow(2).mean()

                # Optimize
                self.optimizer.zero_grad()
                (policy_loss + 0.5 * value_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                self.optimizer.step()