from matplotlib.pyplot import title
from minipong import Minipong
import numpy as np
from itertools import count, repeat
from collections import namedtuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation

# Initialize pong object
pong = Minipong(level=3, size = 5)

# Create actor critic class
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy2(nn.Module):
    """
    implements a policy module for Actor-Critic
    """
    def __init__(self, ninputs, noutputs, gamma = 0.99):
        super(Policy2, self).__init__()
        self.fc1 = nn.Linear(ninputs, 128)
        
        self.actor = nn.Linear(128, noutputs)
        self.critic = nn.Linear(128, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)

        # discount factor
        self.gamma = gamma
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        
        # smallest useful value
        self.eps = np.finfo(np.float32).eps.item()


    def forward(self, x):
        x = F.relu(self.fc1(x))
        # actor: choose action by returning probability of each action
        action_prob = F.softmax(self.actor(x), dim=1)
        # critic: evaluates being in the x
        state_values = self.critic(x)
        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state x 
        return action_prob, state_values
    
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()
        
    
    def update(self):
        R = 0
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # standardise returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        # compute policy losses, using stored log probs and returns
        loss = 0
        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss 
            policy_loss = -log_prob * advantage
            # calculate critic (value) loss using L1 smooth loss
            value_loss = F.smooth_l1_loss(value, torch.tensor([R]).unsqueeze(0))
            loss += (policy_loss + value_loss)
            
        # run backprop through all that
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # delete stored values 
        del self.rewards[:]
        del self.saved_actions[:]




ninputs = pong.observationspace()
noutputs = 3
policy = Policy2(ninputs, noutputs)    

policy.load_state_dict(torch.load("/home/zealouspriest/Python_Projects/AML/Project/A2C.pth "))
policy.eval()

# Testing the created model
episodes = range(50)
gamma = 0.99
render = True
finalrender = True
log_interval = 100
render_interval = 1000
running_reward = 0

starttime = time.time()
test_reward = []
for i in episodes:
    state, ep_reward, done = pong.reset(), 0, False
    
    rendernow = i % render_interval == 0
    for j in range(10000):
        # select action (randomly)
        action = policy.select_action(state)

        # take the action
        state, reward, done = pong.step(action)
        reward = float(reward)     # strange things happen if reward is an int

                  
        ep_reward += reward

        if done:
            test_reward.append(ep_reward)
            break

    # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # log results
        if i % log_interval == 0:
            print('Episode {}\t  Run: {} \t Last reward: {:.2f}\t Average reward: {:.2f}'.format(
                i, j, ep_reward, running_reward))

        # check if we have solved minipong
        if running_reward > 300:
            secs = time.time() - starttime
            mins = int(secs/60)
            secs = round(secs - mins * 60.0, 1)
            test_reward.append(ep_reward)
            print("Solved in {}min {}s!".format(mins, secs))
            
            print("Running reward is now {:.2f} and the last episode {} "
                    "runs to {} time steps!".format(running_reward, i, j))

            if finalrender:
                state, ep_reward, done = pong.reset(), 0, False
                fig = plt.figure()
                data = []
                tot_reward = 0
                # plt.ion()
                for t in range(1, 10000):
                    action = policy.select_action(state)
                    state, reward, done = pong.step(action)
                    if finalrender:
                        tot_reward += reward
                        im = plt.imshow(pong.to_pix(pong.s1), origin="lower", animated = True, cmap = plt.cm.coolwarm)
                        data.append([im, plt.text(5.95, 14, f"Reward = {tot_reward}")])
                        
                    # ax.imshow(im, origin = "lower", cmap=plt.cm.coolwarm)
                    # plt.title(f"Reward = {tot_reward}", loc="right")
                    # fig.canvas.draw()
                    # plt.pause(0.00001)

                    if done and finalrender:
                        ani = ArtistAnimation(fig, data, interval = 50, blit=True, repeat=False)
                        plt.show()
                        finalrender = False
                        break
            break   


