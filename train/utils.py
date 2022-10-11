from config import config
from device import device

import numpy as np
import torch
import random

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

import torch

# Utilities Class for methods required through out
class Utils:
    def __init__(self):
        pass
    
    def readData(self, fileIdx):
        '''
            read data from fileIdx th data from ./Data/data.py
        '''
        global config
        fileData = config.data[str(fileIdx)]
        vehicleNum = fileData['vehicleNum']
        capacity = fileData['capacity']
        startTime = fileData['startTime']
        endTime = fileData['endTime']
        serviceDuration = fileData['serviceDuration']
        demands = [0]
        locations = [fileData['depot']]
        serviceTime = [[0, 0]]

        group = fileData['group']
        groupIdx = random.randint(0, len(group) - 1)
        demands.extend(group[groupIdx]['demands'])
        locations.extend(group[groupIdx]['locations'])
        serviceTime.extend(group[groupIdx]['serviceTime'])

        if(len(demands) < config.numNodes):
            _idx = config.numNodes - len(demands)
            while(_idx > 0):
                demands.append(random.randint(config.minimumDemand, config.maximumDemand))
                locations.append([random.randint(0, 80), random.randint(0, 80)])
                _startTime = random.randint(0, endTime // 2)
                serviceTime.append([_startTime, _startTime + random.randint(0, endTime // 2)])
                _idx -= 1

        return vehicleNum, capacity, np.asarray(demands, dtype = np.int), np.asarray(locations, dtype = np.int), np.asarray(serviceTime, dtype = np.int), startTime, endTime, serviceDuration
    
    def generateNodes(self):
        '''
            Generate the graph with nodes, paisise distances and demands
        The limitation:
            The minimum unit for both distance and time is 1.

            The maximum distance between the node and the depot is (config.totalDuration - config.serviceDuration) / 2 * config.speed.
            The minimum distance between each node is 1.

            The earlist service time for a cumstor should be greater than config.startTime + the distance between this node and the depot / config.speed.
            The latest service time for a cumstor should be smaller than config.totalDuration - the distance between this node and the depot / config.speed  - config.serviceDuration;
        '''
        global config
        # generate the nodes assume location of the depot is (0, 0)
        nodeLocations = np.zeros((1, 2), dtype = int)
        nodeLocations = np.concatenate([nodeLocations, np.random.randint(config.minimumLocationValue, config.maximumLocationValue + 1, size = (config.numCust, 2))]).astype(int)
        nodeDistances = distance_matrix(nodeLocations, nodeLocations).astype(int)
        
        row = 1
        
        while(row < config.numNodes): 
            col = 0
            while(col < row):
                if(col == 0 and (nodeDistances[row][col] > config.maximumLocationValue or nodeDistances[row][col] < 1)): 
                    nodeLocations[row] = np.random.randint(config.minimumLocationValue, config.maximumLocationValue + 1, size = 2)
                    nodeDistances = distance_matrix(nodeLocations, nodeLocations).astype(int)
                    col = -1
                elif(nodeDistances[row][col] < 1): 
                    nodeLocations[row] = np.random.randint(config.minimumLocationValue, config.maximumLocationValue + 1, size = 2)
                    nodeDistances = distance_matrix(nodeLocations, nodeLocations).astype(int)
                    col = -1

                col += 1
            row += 1
        
        # generate the service times
        serviceTime = np.zeros((config.numNodes, 2), dtype = int)
        for node in range(1, config.numCust + 1):
            left = config.startTime + nodeDistances[node][0] // config.speed
            right = config.totalDuration - nodeDistances[node][0] // config.speed  - config.serviceDuration + 1
            rand1 = np.random.randint(left, right)
            rand2 = np.random.randint(left, right)
            while(left + 1 < right and rand2 == rand1):
                rand2 = np.random.randint(left, right)
            if(rand1 < rand2):
                serviceTime[node][0] = rand1
                serviceTime[node][1] = rand2
            else:
                serviceTime[node][0] = rand2
                serviceTime[node][1] = rand1

        # generate the demands
        demands = np.random.randint(config.minimumDemand, config.maximumDemand + 1, config.numNodes)
        for i in range(config.numDepot):
            demands[i] = 0
        
        return nodeLocations, nodeDistances, serviceTime, demands
    
    def distanceTravelled(self, nodeDistances, visitedNodes):
        '''
            calculate the total distance travelled till the moment
        '''
        distTravelled = 0
        if len(visitedNodes) > 1:
            for i in range(len(visitedNodes) - 1):
                distTravelled += nodeDistances[visitedNodes[i], visitedNodes[i + 1]].item()
        return distTravelled
    

    def nextRandomNode(self, state, leftCapacity, demands, currentNode, timer):
        '''
            get a random next node based on the current state, capacity left in the truck
        The limitation:
            leftCapacity should be greater or equal than the demand of the customer
            
            the node should not be visited before
            
            after the vehicle arrived at the servering cumstomer, the time should within the serviceTime
            
            after the vehicle finished serving, it should have time to return back to the depot.
        '''
        global config
        
        serviceTime = state.serviceTime
        nodeDistances = state.nodeDistances
        #If no nodes are visited, select a random Node which is a depot
        if len(state.visitedNodes) == 0:
            return np.random.randint(0, (config.numDepot))
        # If there are few nodes visited, we select the next node as a random customer node which is not visited earlier
        visited = set(state.visitedNodes)
        nodes = list(range(config.numDepot, config.numNodes))
        np.random.shuffle(nodes)
        for node in nodes:
          #We select the node whose demand can be satisfied with left over truck capacity
            arrivalTime = timer + nodeDistances[currentNode][node] // config.speed
            if ((node not in visited) and (leftCapacity >= demands[node]) and (serviceTime[node][1] >= arrivalTime)):
                return node
        #If no such node is found, we visit current node
        return currentNode
    
    def calculateReward(self, oldTotalDistance, newTotalDistance, newNode, serviceTime, timer, oldNode):
        '''
            calculate the reward for list of visited nodes
        '''
        global config

        toNextNodeDistance = newTotalDistance - oldTotalDistance
        
        arrivedTime = timer + toNextNodeDistance // config.speed
        waitingTime = serviceTime[newNode, 0] - arrivedTime

        reward = config.rewardParam1 * newTotalDistance + \
                 config.rewardParam2 * toNextNodeDistance + \
                 config.rewardParam3 * (serviceTime[newNode, 1] - max(serviceTime[newNode, 0], timer)) + \
                 config.rewardParam4 * (0 if waitingTime < 0 else waitingTime) + \
                 config.rewardParam5 * (serviceTime[newNode, 0] - timer)

        return reward
    
    
    def plotGraph(self, solution):
        '''
            plot the result graph based on the solution
        '''
        nodeLocations = solution[0]
        visitedNodes = solution[1]
        serviceTime = solution[2]
        demands = solution[3]
        
        
        for idx in range(len(nodeLocations)):
            plt.text(nodeLocations[idx,0], nodeLocations[idx,1], str(serviceTime[idx][0]) + '-' + str(serviceTime[idx][1]) + ', ' + str(demands[idx]))
        
        plt.scatter(nodeLocations[:,0], nodeLocations[:,1], s = 30, color = 'g')
        n = len(nodeLocations)
        for idx in range(len(visitedNodes) - 1):
            originalPos = [nodeLocations[visitedNodes[idx], 0], nodeLocations[visitedNodes[idx], 1]]
            nxPos = [nodeLocations[visitedNodes[idx + 1], 0], nodeLocations[visitedNodes[idx+1], 1]]
            plt.plot([originalPos[0], nxPos[0]],[originalPos[1], nxPos[1]] , 'k', lw = 1)
            
            dx = (nxPos[0] - originalPos[0]) * 0.1
            dy = ((nxPos[1] - originalPos[1]) / (nxPos[0] - originalPos[0])) * dx
            
            plt.arrow((nxPos[0] + originalPos[0]) / 2, (nxPos[1] + originalPos[1]) / 2, dx, dy, lw = 0, length_includes_head=True, head_width = 1)
        for node in visitedNodes:
            #Mark the depots to make sure we identify them on the graph
            if node < config.numDepot:
                plt.plot(nodeLocations[node, 0], nodeLocations[node, 1], 'X', markersize = 15, color = 'r')
    

    def getEpsilonValueForEpisode(self, episode):
        '''
            Get the updated Epsilon value for the episode, for exploitation and exploration
        '''
        global config
        episodeValue = (1 - config.epsilonDR) ** episode
        return max(config.epsilonMin, episodeValue)
    
    def getSamples(self):
        '''
            get all the required sample data to begin an episode. 
        '''
        nodeLocations, nodeDistances, serviceTime, demands = self.generateNodes()
        global config
        nodeDistances = torch.tensor(nodeDistances, dtype=torch.float32, requires_grad=False, device=device) #Generate a graph
        visitedNodes = [random.randint(0, config.numDepot - 1)] #Select a random depot node to start with
        return nodeLocations, nodeDistances, serviceTime, demands, visitedNodes