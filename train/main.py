import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
from datetime import datetime

from scipy.spatial import distance_matrix

# local files
from config import config
from state import State
from utils import Utils
from operation import Operation
from action import History

# create utils
utils = Utils()

# Training the model
opration = Operation()
history = History()
route = []
visitedNum = []
avgRouteLen = float('inf')

DATA_DIR = '../Data/data.npy'
config.data = np.load(DATA_DIR,allow_pickle='TRUE').item()
 # Iterate over number of episodes
for episode in range(config.numEpisode):
    
    config.timer = -1
    currentEpsilon = utils.getEpsilonValueForEpisode(episode) # Get current epsilon value

    _, config.capacity, demands, nodeLocations, serviceTime, config.startTime, config.endTime, config.serviceDuration = utils.readData(random.randint(0, len(config.data) - 1))
    
    config.totalDuration = config.endTime - config.startTime
    config.numCust = len(demands) - 1
    config.numDepot = 1
    config.numNodes = len(demands)
    nodeDistances = distance_matrix(nodeLocations, nodeLocations).astype(int)

    visitedNodes = [random.randint(0, config.numDepot - 1)] #Select a random depot node to start with

    oldState = State(numCust = config.numCust, numDepots = config.numDepot, capacity = config.capacity, 
                     visitedNodes = visitedNodes, nodeLocations = nodeLocations, demands = demands, 
                     nodeDistances = nodeDistances, serviceTime = serviceTime, oldNode = 0, newNode = 0, 
                     currentTimestamp = config.timer, totalDistance = 0)

    oldStateTensor = oldState.getTensor() #Generate State Tensor
    states = [oldState] # List of states for current episode
    stateTensors = [oldStateTensor] # List of state tensors
    rewards = [] # List of rewards
    actions = [] # list of actions performed which is essentially the order of nodes visited
    
    oldNode = 0
    leftCapacity = config.capacity
    # While we have demands to be satisfied
    while (not oldState.isFinalState() and config.timer < config.endTime):
        config.timer += 1
        # if the vehicle does not finish its current service
        if(oldState.currentServiceEndTime >= config.timer):
            continue
        # epsilon marks the trade-off between exploration and exploitation.
        if currentEpsilon < random.random(): # here is exploitation
          # Get node based on Q-Function
            newNode, reward = opration.getNextAction(oldState, leftCapacity, config.timer)
        else: # here is exploration
            newNode = utils.nextRandomNode(oldState, leftCapacity, demands, oldNode, config.timer) # Get Random node
        # If the node is depot, update the leftover truck capacity
        if(newNode == oldNode):
            continue
        leftCapacity -= demands[newNode]
        # Update the new list of visited nodes
        newVisitedNodes = list(visitedNodes)
        newVisitedNodes.append(newNode)
        
        newTotalDistance = oldState.totalTravelledDistance + oldState.nodeDistances[oldNode, newNode]
        # Calculate Reward  
        reward = utils.calculateReward(oldState.totalTravelledDistance, newTotalDistance, newNode, serviceTime, config.timer, oldNode)
        
        # Create new State and corresponding tensor
        newState = State(numCust = config.numCust, numDepots = config.numDepot, capacity = config.capacity, 
                     visitedNodes = newVisitedNodes, nodeLocations = nodeLocations, demands = demands, 
                     nodeDistances = nodeDistances, serviceTime = serviceTime, oldNode = oldNode, newNode = newNode, 
                     currentTimestamp = config.timer, totalDistance = newTotalDistance)
        

        newStateTensor = newState.getTensor()

        states.append(newState)
        stateTensors.append(newStateTensor)
        rewards.append(reward)
        actions.append(newNode)
        
        # Add event to history
        history.addToHistory(visitedNodes, rewards, states, newState, actions)
        
        # update the state to new state
        oldState = newState
        oldStateTensor = newStateTensor
        visitedNodes = newVisitedNodes
        oldNode = newNode
        
        
        loss = 0

        # If there are enough events, sample and update the model parameters
        if history.getNumPastEvents() > config.minEventsForGD:
            events = history.getSamplePast()
            loss = opration.update(events, config.timer)
        

    # Save the model
    if (episode % config.modelSavingTHreshold == 0):
        avgRouteLen = np.mean(route[-config.modelSavingTHreshold:])
        print('loss: %f' % (loss))
        opration.saveModel(episode, loss, avgRouteLen)
    
    # Calculate total path length            
    length = utils.distanceTravelled(nodeDistances, visitedNodes)
    
    route.append(length)
    visitedNum.append(len(np.unique(visitedNodes)))

    if episode % 10 == 0:
        with open('./result/loss.text', 'a') as f:
            f.write(str(loss) + " ")

        with open('./result/visitedNum.text', 'a') as f:
            f.write(str(np.mean(visitedNum[-50:])) + " ")
        
        with open('./result/rout.text', 'a') as f:
            f.write(str(np.mean(route[-50:])) + " ")

    if episode % 100 == 0:
        print('Episode %d. length = %.3f visited nodes = %.3f time = %s' % (episode, np.mean(route[-50:]), np.mean(visitedNum[-50:]), datetime.now()))
    