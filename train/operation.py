from config import config
from device import device

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from DQN import DQN

class Operation():
    '''
        Operation Class to perform operations on Network
    '''
    def __init__(self):
        global config
        self.network = DQN().to(device) #instance of network
        self.optimizer = optim.Adam(self.network.parameters(), lr = config.lr) #optimizer
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = config.gamma) # Scheduler, reduce the learning rate as the number of epoches increases
        self.lossFunc = nn.MSELoss() #Loss Function
    
    def loadModel(self, fileName):
        '''
            Load an existing model and update parameters  
        '''
        global config
        if fileName is not None:
            model = torch.load(fileName)
            self.network.load_state_dict(model['network'])
            self.optimizer.load_state_dict(model['optimizer'])
            self.scheduler.load_state_dict(model['scheduler'])
    
    def saveModel(self, episode, loss, averageLength):
        '''
            Save the current model parameters
        '''
        global config

        # Check whether the specified path exists or not
        if not os.path.exists(config.modelFolder):
            # Create a new directory because it does not exist 
            os.makedirs(config.modelFolder)

        fname = config.modelFolder + F"{str(loss) }_length_{averageLength}.tar"
        torch.save({'episode': episode, 'network': self.network.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'loss': loss, 'averageLength': averageLength
        }, fname)
    
    def predict(self, state):
        '''
            Predict the rewards of all possible actions based on current state
        '''
        # temporarily sets all of the requires_grad flags to false. 
        with torch.no_grad(): 
            rewards = self.network(state.getTensor().unsqueeze(0))
        return rewards[0]

    def getNextAction(self, state, leftCapacity, timer):
        '''
            Get the next best action  
            old state, left capacity, timer
        '''
        rewards = self.predict(state) # Calculate Rewards for all actions
        
        rewardsindices = rewards.argsort(descending = True) # Sort the rewards
        
        serviceTime = state.serviceTime
        nodeDistances = state.nodeDistances
        # Pick the node which results in maximum reward and making sure that it is not visited previously and the demand can be satisfied by current truck capacity
        for node in rewardsindices.tolist():
            arrivalTime = timer + nodeDistances[state.newNode][node] // config.speed
            if ((node >= config.numDepot) and 
                (node not in state.visitedNodes) and 
                (state.demands[node] <= leftCapacity) and 
                (serviceTime[node][1] >= arrivalTime)):
                # print("----: %f" %(serviceTime[node][1] - arrivalTime))
                return node, rewards[node].item()
        # If no such node is left, return current node
        randomDepot = state.newNode
        return randomDepot, rewards[randomDepot].item()
    
    def update(self, events, timer):
        '''
            Update the paramters of the network for a given sample of past events
        '''
        global config
        stateTensors = []
        # distanceTensors = []
        actions = []
        targetRewards = []
        # Generate a list of tensors for performing update
        for event in events:
            stateTensors.append(event.stateTensor)
            # distanceTensors.append(event.state.nodeDistances)
            actions.append(event.action)
            
        # distanceTensors = torch.stack(distanceTensors).to(device)
        stateTensors = torch.stack(stateTensors).to(device)

        # Calculate the target reward as the sum of rewards at each step for every event and the reward for the next state for a non terminal State
        for i, event in enumerate(events):
            targetReward = event.reward
            leftCapacity = np.max(config.capacity - (sum([event.state.demands[node] for node in event.state.visitedNodes])), 0)
            if not event.nextState.isFinalState():
                
                node, reward = self.getNextAction(event.nextState, leftCapacity, timer)
                
                if node >= config.numDepot:
                    leftCapacity -= event.state.demands[node]
                # else:
                #     leftCapacity = config.capacity
                targetReward += config.qValueGamma * reward
            targetRewards.append(targetReward)

        #Clear the existing gradients in the optimiser       
        self.optimizer.zero_grad()
        #Calculate the rewards
        rewards = self.network(stateTensors)[range(len(actions)), actions]
        #Calculate loss with respect to target rewards
        loss = self.lossFunc(rewards, torch.tensor(targetRewards, device = device, dtype=torch.float32))
        lossValue = loss.item()
        #Propagate the loss and update parameters of network
        loss.backward()
        self.optimizer.step()        
        self.scheduler.step()
        
        return lossValue