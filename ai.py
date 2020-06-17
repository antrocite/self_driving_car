import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


#creating architecture of neural network
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()   #attach nn.Module things
        self.input_size = input_size #amount of input neurons
        self.nb_action = nb_action #num of possible actions
        self.fc1 = nn.Linear(self.input_size, 50) #make connections between input layer and hidden layer
        self.fc2 = nn.Linear(50, self.nb_action) #make connections between hidden layer and output layer

    def forward(self, state):   #forward propogation
        x = F.relu(self.fc1(state)) #activate hidden neurons. relu - rectifier function
        q_values = self.fc2(x) #output neurons
        return q_values

#implemetning Experience replay
class ReplayMemory(object):  #object is just for python2

    def __init__(self, capacity):
        self.capacity = capacity  #amount of states in one batch
        self.memory = []          #list in which we push events
    
    def push(self, event):        #to push events to memory
        self.memory.append(event)
        if(len(self.memory) > self.capacity):
            del self.memory[0]           #if memory overflow delete oldest event

    def sample(self, batch_size):        #take few samples from memory
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

class Dqn():             #the q-learning algorithm

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma  # delay discount
        self.reward_window = [] #?list of mean of last 100 rewards
        self.model = Network(input_size, nb_action) #1 object of our network
        self.memory = ReplayMemory(100000) #1 object of memory batch
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)  #lr - learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward  = 0

    def select_action(self, state):  #select the action
        probs = F.softmax(self.model(Variable(state, volatile = True))*10)  #Returns probabilities for actions. temperature = 7. the higher it is, the more explorative ai. volatile prevents using 'gradient' from state tensor and thus saves memory
        action = probs.multinomial(1)  #returns random draw as a fake batch 1 IS WARNING
        return action.data[0, 0]   

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):   #calc the error and perform back propagation
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        #td_loss.backward()
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 200:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(200)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000: #??  100?
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)

    def save(self):
        torch.save({ 'state_dict' : self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print("No brains found...")


