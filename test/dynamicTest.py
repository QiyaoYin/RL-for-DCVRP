import warnings
warnings.filterwarnings('ignore')
import argparse

import random
import numpy as np
import torch
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# local 
from config import config
from operation import Operation
from utils import Utils
from state import State

parser = argparse.ArgumentParser()
parser.add_argument("--save", dest='save', help="save graph. 1: save", type = int, default = 0)
parser.add_argument("--data", dest="data", help = "use which data set", type = str, default = "c101.txt")
parser.add_argument("--percent", dest="percent", help = "dyanmic percent", type = float, default = 0.5)
args = parser.parse_args()

config.maskRate = args.percent


# set random set for np random and random
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)  # set random seed for torch

utils = Utils()

DATA_DIR = '../Data/test/'

filename = "33337.84765625_length_184.0.tar"
modelname = args.data
bestModelFileName = config.modelFolder + filename; #Select model with best performance from the checkpoints

operator = Operation()
operator.loadModel(bestModelFileName)

sampleResults = list()
config.timer = -1
# Generate Initial data to begin with
# config.nodeLocations, config.nodeDistances, config.serviceTime, config.demands, config.visitedNodes = utils.getSamplesForEpisode()
# return vehicleNum, capacity, demands, locations, serviceTime, startTime, endTime, serviceDuration
config.numVehicle, config.capacity, config.demands, config.nodeLocations, config.serviceTime, config.startTime, config.endTime, config.serviceDurationPerCust = utils.readDynamicData(DATA_DIR + modelname)
config.totalDuration = config.endTime - config.startTime
config.numCust = len(config.demands) - 1
config.numDepot = 1
config.numNodes = len(config.demands)
config.nodeDistances = distance_matrix(config.nodeLocations, config.nodeLocations).astype(int)

config.visitedNodes = [random.randint(0, config.numDepot - 1)] #Select a random depot node to start with

utils.initializeVehicleStates()

# if there still has unvisited nodes and time does not reach the due time
while (len(config.visitedNodes) < config.numNodes and config.timer < config.endTime):
    config.timer += 1
    # add dynamic nodes
    utils.addNodes(config.timer)    
    # if there exist a free vehicle (in the depot or finished serving)
    freeVehicles = []
    for i in range(config.numVehicle):
        if(i in config.disabledVehicles):
            continue
        if config.states[i][-1].currentServiceEndTime <= config.timer and i not in config.vehicleInWaitingList: # exist
            freeVehicles.append(i)
    
    # if does not exist one free vehicle and no vehicle in the waiting list at this time, just continue
    if(len(freeVehicles) == 0 and len(config.vehicleInWaitingList) == 0):
        continue
    # assign one (node, vehicle) pair into waiting list
    freeVehicles = utils.assignToWaiting(operator, freeVehicles, config.timer)

    removeNodeList = []
    # if exists some (node, vehicle) pairs can going into serving, remove from waiting list
    for node, vehicle in config.waitingList.items():
        if(config.timer >= vehicle[2]): # if this vehicle have to serve node
            removeNodeList.append(node)
            config.visitedNodes.append(node)
            oldState = config.states[vehicle[0]][-1]
            partVisitedNodes = [*oldState.visitedNodes, node]

            reward = vehicle[1]
            partActions = [*oldState.actions, node]
            partRewards = [*oldState.rewards, reward]

            newState = State(visitedNodes = partVisitedNodes, oldNode = oldState.newNode, newNode = node, 
                    currentTimestamp = config.timer, totalDistance = utils.distanceTravelled(config.nodeDistances, partVisitedNodes), 
                    actions = partActions, rewards = partRewards, vehicle = vehicle[0], leftCapacity = oldState.leftCapacity - config.demands[node])
            config.states[vehicle[0]].append(newState)
    for node in removeNodeList:
        utils.removeFromWaitingList(node)
    
    

if args.save:
    plt.figure(figsize=(30, 30), dpi = 100)
else:
    plt.figure()
utils.plotDynamicGraph([config.nodeLocations, config.states, config.serviceTime, config.demands], modelname)
plt.savefig('../images/dynamic/' + modelname + '_' + filename + '_' + str(args.percent)  + '_test.png', bbox_inches = 'tight')
    