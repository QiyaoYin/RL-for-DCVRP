from config import config
from device import device
import torch
# State Class which is used through out the code to maintain the state
class State:
    def __init__(self, visitedNodes = None, oldNode = None, newNode = None, currentTimestamp = None, totalDistance = None, 
                        actions = None, rewards = None, vehicle = None, leftCapacity = None):
        global config
        # Creates a state based on the existing information and updates the values of various parameters accordingly
        self.actions = actions
        self.rewards = rewards
        self.visitedNodes = visitedNodes
        self.currentTimestamp = currentTimestamp
        self.newNode = newNode
        self.oldNode = oldNode
        self.totalTravelledDistance = totalDistance
        self.vehicle = vehicle
        self.leftCapacity = leftCapacity
        
        # self.satisfiedDemand = sum([config.demands[node] for node in self.visitedNodes])
        self.toNewNodeDistance = config.nodeDistances[oldNode, newNode]
        
        
        self.currentServiceEndTime = max(currentTimestamp + self.toNewNodeDistance // config.speed, config.serviceTime[newNode, 0])
        self.waitingTime = config.serviceTime[newNode, 0] - self.currentServiceEndTime
        self.remainingTime = config.serviceTime[newNode, 1] - self.currentServiceEndTime
        if newNode != 0:
            self.currentServiceEndTime += config.serviceDurationPerCust
        
        
    # Generate a vector (Tensor) of the current state with the features to generate embedding
    def getTensor(self):
        global config
        tensor = []
        
        for i in range(0, config.numNodes):
            tmp = []
            waitingTime = config.serviceTime[i, 0] - config.nodeDistances[i, self.newNode] // config.speed - self.currentServiceEndTime
            if(waitingTime < 0):
                waitingTime = 0
            
            remainingTime = config.serviceTime[i, 1] - config.nodeDistances[i, self.newNode] // config.speed - self.currentServiceEndTime
            if(i in config.visitedNodes):
                waitingTime = -1
                remainingTime = -1
                

            tmp.append(config.demands[i]) # the demand for each node
            tmp.append(self.leftCapacity - config.demands[i] if i not in config.visitedNodes else self.leftCapacity) # the left demand after visited node i
            tmp.append(config.nodeDistances[self.newNode, i] if i not in config.visitedNodes else 0) # the distance to node
            tmp.append(config.nodeDistances[i, 0] if i not in config.visitedNodes else 0) # the distance to the depot
            tmp.append(self.totalTravelledDistance + config.nodeDistances[self.newNode, i] if i not in config.visitedNodes else self.totalTravelledDistance) # total travelled distance after visiting node i
            tmp.append(config.nodeDistances[self.newNode, 0]) # the distance of latest visited node to the depot
            tmp.append(self.currentServiceEndTime) # the current time stamp
            tmp.append(waitingTime) # waiting time after visiting newNode to arrive node i
            tmp.append(remainingTime) # remaining time after visiting newNode to arrve node i
            tmp.append(config.serviceTime[i, 0] - self.currentServiceEndTime if i not in config.visitedNodes else 0) # the gap between node i and current time

            tensor.append(tmp)
        return torch.tensor(tensor, dtype = torch.float32, requires_grad = False, device = device)