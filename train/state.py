from config import config
from device import device

import torch

class State:
    '''
    The state for each vehicle at each node
    '''
    def __init__(self, 
                 numCust = None, numDepots = None, capacity = None, visitedNodes = None, 
                 nodeLocations = None, demands = None, nodeDistances = None, serviceTime = None,
                 oldNode = None, newNode = None, currentTimestamp = None, totalDistance = None):
        global config
      # Creates a state based on the existing information and updates the values of various parameters accordingly
        # global state
        self.visitedNodes = visitedNodes
        self.satisfiedDemand = sum([demands[node] for node in visitedNodes])
        self.nodeLocations, self.nodeDistances, self.demands = nodeLocations, nodeDistances, demands
        self.truckCapacity = capacity
        self.serviceTime = serviceTime
        self.currentTimestamp = currentTimestamp

        # the state for a single vehicle
        self.newNode = newNode
        self.oldNode = oldNode
        self.totalTravelledDistance = totalDistance
        self.toNewNodeDistance = self.nodeDistances[oldNode][newNode]
        self.leftDemand = self.truckCapacity - self.satisfiedDemand
        
        self.currentServiceEndTime = max(currentTimestamp + self.toNewNodeDistance // config.speed, self.serviceTime[newNode][0])
        self.waitingTime = serviceTime[newNode, 0] - self.currentServiceEndTime
        self.remainingTime = serviceTime[newNode, 1] - self.currentServiceEndTime
        if newNode != 0:
            self.currentServiceEndTime += config.serviceDuration
    
    def isFinalState(self):
        return self.leftDemand == 0
        
        
    def getTensor(self):
        '''
            Generate a vector (Tensor) of the current state with the features to generate embedding  
        '''
        global config
        numNodes = self.nodeLocations.shape[0]
        tensor = []
        
        for i in range(0, numNodes):
            tmp = []

            waitingTime = self.serviceTime[i, 0] - self.nodeDistances[i, self.newNode] // config.speed - self.currentServiceEndTime
            if(waitingTime < 0):
                waitingTime = 0
            
            remainingTime = self.serviceTime[i, 1] - self.nodeDistances[i, self.newNode] // config.speed - self.currentServiceEndTime
            if(i in self.visitedNodes):
                waitingTime = -1
                remainingTime = -1
                

            tmp.append(self.demands[i]) # the demand for each node
            tmp.append(self.leftDemand - self.demands[i] if i not in self.visitedNodes else self.leftDemand) # the left demand after visited node i
            tmp.append(self.nodeDistances[self.newNode, i] if i not in self.visitedNodes else 0) # the distance to node
            tmp.append(self.nodeDistances[i, 0] if i not in self.visitedNodes else 0) # the distance to the depot
            tmp.append(self.totalTravelledDistance + self.nodeDistances[self.newNode, i] if i not in self.visitedNodes else self.totalTravelledDistance) # total travelled distance after visiting node i
            tmp.append(self.nodeDistances[self.newNode, 0]) # the distance of latest visited node to the depot
            tmp.append(self.currentServiceEndTime) # the current time stamp
            tmp.append(waitingTime) # waiting time after visiting newNode to arrive node i
            tmp.append(remainingTime) # remaining time after visiting newNode to arrve node i
            tmp.append(self.serviceTime[i, 0] - self.currentServiceEndTime if i not in self.visitedNodes else 0) # the gap between node i and current time

            tensor.append(tmp)
        return torch.tensor(tensor, dtype = torch.float32, requires_grad = False, device = device)