from config import config
from device import device
from state import State

import random
import numpy as np
import torch
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
class Utils:
    def __init__(self):
        pass
    # read data from text
    def readData(self, fileDir):
        lines = []
        with open(fileDir) as f:
            lines = f.read().splitlines()
            f.close()
        vehicleData = lines[4].split()
        vehicleNum = int(vehicleData[0].strip())
        capacity = int(vehicleData[1].strip())

        demands = []
        locations = []
        serviceTime = []
        serviceTime.append([0, 0])
        depot = lines[9].split()
        locations.append([int(depot[1].strip()), int(depot[2].strip())])
        demands.append(int(depot[3].strip()))
        startTime = int(depot[4].strip())
        endTime = int(depot[5].strip())
        serviceDuration = 90

        for i in range(10, len(lines)):
            node = lines[i].split()
            locations.append([int(node[1].strip()), int(node[2].strip())])
            demands.append(int(node[3].strip()))
            serviceTime.append([int(node[4].strip()), int(node[5].strip())])
        return vehicleNum, capacity, np.asarray(demands, dtype = np.int), np.asarray(locations, dtype = np.int), np.asarray(serviceTime, dtype = np.int), startTime, endTime, serviceDuration
    

    # read data from text
    def readDynamicData(self, fileDir):
        global config
        # fileDir = './Data/c102.txt'
        lines = []
        with open(fileDir) as f:
            lines = f.read().splitlines()
            f.close()
        vehicleData = lines[4].split("         ")
        vehicleNum = int(vehicleData[0].strip())
        capacity = int(vehicleData[1].strip())

        demands = []
        locations = []
        serviceTime = []
        serviceTime.append([0, 0])
        depot = lines[9].split("      ")
        locations.append([int(depot[1].strip()), int(depot[2].strip())])
        demands.append(int(depot[3].strip()))
        startTime = int(depot[4].strip())
        endTime = int(depot[5].strip())
        serviceDuration = 90

        nodeLen = len(lines)
        maskSize = int(config.maskRate * (nodeLen - 10))
        maskIdx = np.random.randint(nodeLen - 10 - 1, size = maskSize) + 10

        for i in range(10, len(lines)):
            node = lines[i].split("      ")
            if i in maskIdx:
                tempMap = {}
                tempMap['location'] = [int(node[1].strip()), int(node[2].strip())]
                tempMap['demand'] = int(node[3].strip())
                tempMap['serviceTime'] = [int(node[4].strip()), int(node[5].strip())]
                config.maskedNodes.append(tempMap)
                continue
            locations.append([int(node[1].strip()), int(node[2].strip())])
            demands.append(int(node[3].strip()))
            serviceTime.append([int(node[4].strip()), int(node[5].strip())])
        return vehicleNum, capacity, np.asarray(demands, dtype = np.int), np.asarray(locations, dtype = np.int), np.asarray(serviceTime, dtype = np.int), startTime, endTime, serviceDuration

    # Generate the graph with nodes, paiwise distances and demands
    def generateNodes(self):
        '''
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
                if(col == 0 and (nodeDistances[row, col] > config.maximumLocationValue or nodeDistances[row, col] < 1)): 
                    nodeLocations[row] = np.random.randint(config.minimumLocationValue, config.maximumLocationValue + 1, size = 2)
                    nodeDistances = distance_matrix(nodeLocations, nodeLocations).astype(int)
                    col = -1
                elif(nodeDistances[row, col] < 1): 
                    nodeLocations[row] = np.random.randint(config.minimumLocationValue, config.maximumLocationValue + 1, size = 2)
                    nodeDistances = distance_matrix(nodeLocations, nodeLocations).astype(int)
                    col = -1

                col += 1
            row += 1
            
        # generate the service times
        serviceTime = np.zeros((config.numNodes, 2), dtype = int)
        for node in range(1, config.numCust + 1):
            left = config.startTime + nodeDistances[node, 0] // config.speed
            right = config.totalDuration - nodeDistances[node, 0] // config.speed  - config.serviceDurationPerCust + 1
            rand1 = np.random.randint(left, right)
            rand2 = np.random.randint(left, right)
            while(left + 1 < right and rand2 == rand1):
                rand2 = np.random.randint(left, right)
            if(rand1 < rand2):
                serviceTime[node, 0] = rand1
                serviceTime[node, 1] = rand2
            else:
                serviceTime[node, 0] = rand2
                serviceTime[node, 1] = rand1

        # generate the demands
        demands = np.random.randint(config.minimumDemand, config.maximumDemand + 1, config.numNodes)
        for i in range(config.numDepot):
            demands[i] = 0
        
        return nodeLocations, nodeDistances, serviceTime, demands
    
    # Calculates the total distance travelled till the moment. It uses pairwise distances calculated earlier
    def distanceTravelled(self, nodeDistances, visitedNodes):
        distTravelled = 0
        if len(visitedNodes) > 1:
            for i in range(len(visitedNodes) - 1):
                distTravelled += nodeDistances[visitedNodes[i], visitedNodes[i + 1]].item()
        return distTravelled
    
    # Plot the graph of nodes visited with path
    # soluton: (config.nodeLocations, config.states, config.serviceTime, config.demands)
    def plotGraph(self, solution):
        global config

        nodeLocations = solution[0]
        states = solution[1]
        serviceTime = solution[2]
        demands = solution[3]
        
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, config.numVehicle)]
        
        for idx in range(len(nodeLocations)):
            plt.text(nodeLocations[idx, 0], nodeLocations[idx, 1], str(serviceTime[idx, 0]) + ' ' + str(serviceTime[idx, 1]))

        #
        
        plt.scatter(nodeLocations[:,0], nodeLocations[:,1], s = 30, color = 'g')
        
        visitedNodesNum = 0
        vehicleUsedNum = 0
        travelledLen = 0

        for vechieStateIdx in range(len(states)):
            visitedNodes = states[vechieStateIdx][-1].visitedNodes
            if(len(visitedNodes) > 1):
                vehicleUsedNum += 1
                visitedNodesNum += len(visitedNodes) - 1

            for idx in range(len(visitedNodes) - 1):
                originalPos = [nodeLocations[visitedNodes[idx], 0], nodeLocations[visitedNodes[idx], 1]]
                nxPos = [nodeLocations[visitedNodes[idx + 1], 0], nodeLocations[visitedNodes[idx + 1], 1]]
                plt.plot([originalPos[0], nxPos[0]],[originalPos[1], nxPos[1]] , color = colors[vechieStateIdx], lw = 1)
                travelledLen += config.nodeDistances[visitedNodes[idx], visitedNodes[idx + 1]]

                dx = (nxPos[0] - originalPos[0]) * 0.1
                dy = ((nxPos[1] - originalPos[1]) / (nxPos[0] - originalPos[0])) * dx
                plt.arrow((nxPos[0] + originalPos[0]) / 2, (nxPos[1] + originalPos[1]) / 2, dx, dy, lw = 0, length_includes_head=True, head_width = 1)
            travelledLen += config.nodeDistances[0, visitedNodes[-1]]
        # plot depot
        plt.plot(nodeLocations[0, 0], nodeLocations[0, 1], 'X', markersize = 15, color = 'r')
        plt.text(0, 0, 'vehicles: %d/%d    nodes: %d/%d    length: %d' % (vehicleUsedNum,config.numVehicle, visitedNodesNum, config.numCust, travelledLen))
    
    def plotDynamicGraph(self, solution, dataname):
        global config

        nodeLocations = solution[0]
        states = solution[1]
        serviceTime = solution[2]
        demands = solution[3]
        
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, config.numVehicle)]
        
        for idx in range(len(nodeLocations)):
            plt.text(nodeLocations[idx, 0], nodeLocations[idx, 1], str(serviceTime[idx, 0]) + ' ' + str(serviceTime[idx, 1]))
        
        for node in config.dynamicNodes:
            plt.text(node['location'][0], node['location'][1] - 1, str(node['time']), color="red")
        #
        
        plt.scatter(nodeLocations[:,0], nodeLocations[:,1], s = 30, color = 'g')
        
        visitedNodesNum = 0
        vehicleUsedNum = 0
        travelledLen = 0
        travelledLenWithoutDepot = 0

        for vechieStateIdx in range(len(states)):
            visitedNodes = states[vechieStateIdx][-1].visitedNodes
            if(len(visitedNodes) > 1):
                vehicleUsedNum += 1
                # print(visitedNodes)
                visitedNodesNum += len(visitedNodes) - 1

            for idx in range(len(visitedNodes) - 1):
                originalPos = [nodeLocations[visitedNodes[idx], 0], nodeLocations[visitedNodes[idx], 1]]
                nxPos = [nodeLocations[visitedNodes[idx + 1], 0], nodeLocations[visitedNodes[idx + 1], 1]]
                plt.plot([originalPos[0], nxPos[0]],[originalPos[1], nxPos[1]] , color = colors[vechieStateIdx], lw = 1)
                travelledLen += config.nodeDistances[visitedNodes[idx], visitedNodes[idx + 1]]
                travelledLenWithoutDepot += config.nodeDistances[visitedNodes[idx], visitedNodes[idx + 1]] if idx > 0 else 0
                
                dx = (nxPos[0] - originalPos[0]) * 0.1
                dy = ((nxPos[1] - originalPos[1]) / (nxPos[0] - originalPos[0])) * dx
                plt.arrow((nxPos[0] + originalPos[0]) / 2, (nxPos[1] + originalPos[1]) / 2, dx, dy, lw = 0, length_includes_head=True, head_width = 1)
            
            travelledLen += config.nodeDistances[0, visitedNodes[-1]]
        # plot depot
        plt.plot(nodeLocations[0, 0], nodeLocations[0, 1], 'X', markersize = 15, color = 'r')
        # plt.text(0, 0, 'vehicles: %d/%d    nodes: %d/%d    length: %d' % (vehicleUsedNum,config.numVehicle, visitedNodesNum, config.numCust, travelledLen))
        plt.text(0, 0, 'vehicles: %d/%d   length: %d' % (vehicleUsedNum,config.numVehicle, travelledLen))

        with open(config.resultFile + str(config.maskRate) + '_' + dataname, 'a') as f:
            f.write("%d %d %d %d %d %d\n" %(vehicleUsedNum, config.numVehicle, visitedNodesNum, config.numCust, travelledLen, travelledLenWithoutDepot))

    # Get all the required sample data to begin an episode. 
    # do not need to change here
    def getSamplesForEpisode(self):
        # nodeLocations, nodeDistances, demands = self.generateNodes()
        nodeLocations, nodeDistances, serviceTime, demands = self.generateNodes()
        global config
        nodeDistances = torch.tensor(nodeDistances, dtype = torch.float32, requires_grad = False, device = device) #Generate a graph
        visitedNodes = [random.randint(0, config.numDepot - 1)] # Select a random depot node to start with
        return nodeLocations, nodeDistances, serviceTime, demands, visitedNodes

    # initialize the state for all vehicles
    def initializeVehicleStates(self):
        global config
        config.states = [[] for y in range(config.numVehicle)]
        config.waitingList = {}
        config.vehicleInWaitingList = []
        config.disabledVehicles = []

        for i in range(config.numVehicle):
            initialState = State(visitedNodes = [0], oldNode = 0, newNode = 0, currentTimestamp = config.timer, totalDistance = 0, actions = [], rewards = [], vehicle = i, leftCapacity = config.capacity)
            config.states[i].append(initialState)

    # if a vehicle v in the waiting list
    def inWaitingList(self, v):
        global config
        for _, vehicle in config.waitingList.items():
            if vehicle == v:
                return True
        return False
    
    def removeFromWaitingList(self, node):
        global config
        vehicle = config.waitingList[node][0]
        config.vehicleInWaitingList.remove(vehicle)
        del config.waitingList[node]
        return vehicle

    def addToWaitingList(self, node, vehicle, reward, startServiceTime, latestServiceTime):
        global config
        if vehicle in config.vehicleInWaitingList:
            config.vehicleInWaitingList.remove(vehicle)
            for node, vehiInfo in config.waitingList.items():
                if vehiInfo[0] == vehicle:
                    config.waitingList.pop(node)
                    break
        config.waitingList[node] = [vehicle, reward, startServiceTime, latestServiceTime]
        config.vehicleInWaitingList.append(vehicle)
    
    def assignToWaiting(self, helper, vehicles, timer): # free vehicles
        global config
        isAddVehicle = True
        disabledNode = [[] for i in range(config.numVehicle)] # store all nodes that can not be served by vehicle i
        vehicles = [[v, 0] for v in vehicles]
        while(len(vehicles) > 0 and isAddVehicle):
            isAddVehicle = False
            # tempVehicles = []
            for vehicle in vehicles:
                if vehicle[1] > 3:
                    continue
                isAddVehicle = True
                vehicle[1] += 1
                
                lastState = config.states[vehicle[0]][-1]
                newNode, reward = helper.getNextAction(lastState, timer, disabledNode[vehicle[0]])
                if(newNode == lastState.newNode): # if this vechile disabled
                    config.disabledVehicles.append(vehicle[0])
                    # remove this vehicle from vehicles
                    vehicles.remove(vehicle)
                    continue
                
                # earliestServiceTime = config.serviceTime[newNode, 0] - config.nodeDistances[lastState.newNode, newNode] // config.speed
                # latestServiceTime = config.serviceTime[newNode, 1] - config.nodeDistances[lastState.newNode, newNode] // config.speed
                earliestServiceTime = config.serviceTime[newNode, 0] - config.nodeDistances[lastState.newNode, newNode] // config.speed
                # earliestServiceTime = config.serviceTime[newNode, 0]
                latestServiceTime = config.serviceTime[newNode, 1] - config.nodeDistances[lastState.newNode, newNode] // config.speed
                # latestServiceTime = config.serviceTime[newNode, 1]
                
                # the distance to the newNode
                disToNewNode = config.nodeDistances[newNode, lastState.newNode]
                
                # if(earliestServiceTime < lastState.currentServiceEndTime):
                #     print('time: %f %f' % (earliestServiceTime, lastState.currentServiceEndTime))
                
                isLastTime = (latestServiceTime == timer and lastState.newNode == 0)
                '''
                    here add to the waiting list, should add one more constraint, 
                    if the chosen node can be visited by the nearest vehicle after its serving, then should 
                    not serve this node to reduce the total distance
                '''
                
                if(newNode in config.waitingList): # if meet a conflict
                    LastVehicleIdx = config.waitingList[newNode][0]
                    LastVehicleState = config.states[LastVehicleIdx][-1]
                    if(disToNewNode > config.nodeDistances[newNode, LastVehicleState.newNode]):
                        disabledNode[vehicle[0]].append(newNode)
                        continue
                    
                    if(config.waitingList[newNode][1] < reward): # compare reward if the reward of this new vehicle is greater than the old vehicle
                        lastVehicle = self.removeFromWaitingList(newNode)
                        
                        # waiting list :[vehicle, reward, serviceStartTime, serviceEndTime]
                        self.addToWaitingList(newNode, vehicle[0], reward, earliestServiceTime, latestServiceTime)
                        # remove this vehicle from vehicles
                        vehicles.remove(vehicle)
                        # add the vehicle removed from waitinglist into vehicles
                        vehicles.append([lastVehicle, 0])
                        # tempVehicles.append(lastVehicle)
                    else:
                        disabledNode[vehicle[0]].append(newNode)
                    # isAddVehicle = True
                else: # append newNode into waiting list
                    dummyDis = disToNewNode
                    for state in config.states: # comparing all nodes
                        dummyLastState = state[-1]
                        if lastState.vehicle == vehicle[0]: # if itself
                            continue
                        
                        if dummyLastState.newNode == lastState.newNode: # if this vehicle at the same pos
                            continue
                        
                        if dummyLastState.vehicle in config.vehicleInWaitingList:# if its in the waitinglist
                            continue
                        
                        # if dummyLastState.toNewNodeDistance * 5 < config.nodeDistances[newNode, dummyLastState.newNode]:
                        #     continue
                        # if dummyLastState.oldNode != 0 and config.nodeDistances[dummyLastState.oldNode, dummyLastState.newNode] * 1.1 < config.nodeDistances[dummyLastState.newNode, newNode]:
                        #     continue

                        dummyServiceTime = dummyLastState.currentServiceEndTime + config.nodeDistances[newNode, dummyLastState.newNode] // config.speed
                        # dummyServiceTime = dummyLastState.currentServiceEndTime
                        if dummyServiceTime <= config.serviceTime[newNode, 1]:
                            dummyDis = min(dummyDis, config.nodeDistances[newNode, dummyLastState.newNode])
                    
                    for _node, vehiclePair in config.waitingList.items(): # comparing the node in the waiting list
                        _state = config.states[vehiclePair[0]][-1]
                        _arrivingTime = _state.currentServiceEndTime + config.nodeDistances[_node, _state.newNode] // config.speed + config.serviceDurationPerCust + config.nodeDistances[_node, newNode] // config.speed
                        
                        # if _state.oldNode != 0 and config.nodeDistances[_state.oldNode, _state.newNode] * 1.1 < config.nodeDistances[_node, newNode]:
                        #     continue
                        
                        if _arrivingTime <= config.serviceTime[newNode, 1]:
                            dummyDis = min(dummyDis, config.nodeDistances[newNode, _node])
                    
                    if disToNewNode <= dummyDis or isLastTime:
                        self.addToWaitingList(newNode, vehicle[0], reward, earliestServiceTime, latestServiceTime)
                        # remove this vehicle from vehicles
                        vehicles.remove(vehicle)
                    else:
                        disabledNode[vehicle[0]].append(newNode)
        
        # we have to assign a vehicle in the depot to due customers
        for node in range(0, config.numNodes):
            if node in config.visitedNodes or node in config.waitingList:
                continue
            dueTime = config.serviceTime[node, 1] - timer - config.nodeDistances[0, node]
            # config.serviceTime[newNode, 1] - config.nodeDistances[lastState.newNode, newNode] // config.speed
            if dueTime == 0 and config.serviceTime[node, 0] == 47:
                for vehicle in vehicles:
                    if config.states[vehicle[0]][-1].newNode != 0:
                        continue
                    self.addToWaitingList(node, vehicle[0], 100, config.serviceTime[node, 0] - config.nodeDistances[0, node] // config.speed, config.serviceTime[node, 1] - config.nodeDistances[0, node] // config.speed)
                    vehicles.remove(vehicle)
                    break
        
        return [v[0] for v in vehicles]
            # vehicles = tempVehicles

    def assignDeadNodes(self, helper, vehicles, timer):
        for node in range(config.numNodes):
            if node in config.visitedNodes or node in config.waitingList:
                continue

            disToDepot = config.nodeDistances[node, 0]
            if(disToDepot // config.speed + timer >= config.serviceTime[node, 1]):
                for vehicle in vehicles:
                    if config.states[vehicle][-1].newNode == 0 and vehicle not in config.vehicleInWaitingList:
                        rewards = helper.predict(config.states[vehicle][-1])
                        self.addToWaitingList(node, vehicle, rewards[node].item(),config.serviceTime[node, 0], config.serviceTime[node, 1])
                        break
    
    def assignNodes(self, node, timer):
        global config
        config.maskedNodes.remove(node)
        config.demands = np.append(config.demands, np.array([node['demand']]), axis = 0)
        
        # print(config.nodeLocations)
        config.nodeLocations = np.append(config.nodeLocations, np.array([node['location']]), axis = 0)
        config.serviceTime = np.append(config.serviceTime, np.array([node['serviceTime']]), axis = 0)
        
        # print(config.nodeLocations)
        config.dynamicNodes.append({'location': node['location'], 'time': timer})
        
        config.numCust += 1
        config.numNodes += 1
        config.nodeDistances = distance_matrix(config.nodeLocations, config.nodeLocations).astype(int)

    def addNodes(self, timer):
        global config
        for node in config.maskedNodes:
            if(timer >= 0.7 * node['serviceTime'][0] or random.random() * random.random() > 0.9):
                # assign this to global 
                self.assignNodes(node, timer)