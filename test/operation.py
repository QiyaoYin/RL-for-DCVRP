from config import config
from device import device

import time
import torch
import torch.nn as nn
import torch.optim as optim
from DQN import DQN

# Helper Class to perform operations on Network
class Operation():
    def __init__(self):
        global config
        self.network = DQN().to(device) #instance of network
        self.optimizer = optim.Adam(self.network.parameters(), lr = config.optimizerLR) #optimizer
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = config.schedulerGamma) # Scheduler, reduce the learning rate as the number of epoches increases
        self.lossFunc = nn.MSELoss() #Loss Function
    
    # Load an existing model and update parameters
    def loadModel(self, fileName):
        global config
        if fileName is not None:
            checkpoint = torch.load(fileName)
            self.network.load_state_dict(checkpoint['network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
    

    # Predict the rewards of all possible actions based on current state
    def predict(self, state):
        with torch.no_grad(): # temporarily sets all of the requires_grad flags to false. 
            rewards = self.network(state.getTensor().unsqueeze(0))
        return rewards[0]

    #Get the next best action    
    def getNextAction(self, state, timer, disabledNode):  # old state, left capacity, timer
        rewards = self.predict(state) # Calculate Rewards for all actions
        rewardsindices = rewards.argsort(descending = True) # Sort the rewards
        # Pick the node which results in maximum reward and making sure that it is not visited previously and the demand can be satisfied by current truck capacity
        for node in rewardsindices.tolist():
            if(node in disabledNode):
                continue
            
            arrivalTime = timer + config.nodeDistances[state.newNode, node] // config.speed
            if ((node >= config.numDepot) and 
                (node not in config.visitedNodes) and 
                (config.demands[node] <= state.leftCapacity) and 
                (config.serviceTime[node, 1] >= arrivalTime)):
                return node, rewards[node].item()
        # If no such node is left, return a random depot node as next node
        # randomDepot = np.random.randint(0, config.numDepot)
        randomDepot = state.newNode
        return randomDepot, rewards[randomDepot].item()