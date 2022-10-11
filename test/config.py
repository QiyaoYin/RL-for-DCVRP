import numpy as np
class Config:
    def __init__(self):
        self.capacity = 200 # Capacity of truck
        self.speed = 1 # the speed of vehicle
        self.numVehicle = 25 # the initial vehicle number

        self.timer = -1 # the timer
        self.startTime = 0 # the time vhicles start service
        self.endTime = 1236 # the time vhicles must return back to the depot
        self.totalDuration = self.endTime - self.startTime # vehicles service duration
        self.serviceDurationPerCust = 90 # the service duration for each customer

        # self.maxDistanceToDepot = int((self.totalDuration - self.serviceDurationPerCust) * self.speed // 2) # the maximum distance between the depot and each customers.
        # nodes will spread within the block minimumLocationValue - maximumLocationValue
        # assume the location of depot is (0, 0), so the maximumLocationValue will be maxDistanceToDepot
        # self.minimumLocationValue = 0 
        # self.maximumLocationValue = self.maxDistanceToDepot
        
        # reward parameters
        # self.rewardParam1 = -0.1 # total distance parameter
        # self.rewardParam2 = -3.0 # the distance to next new node parameter
        # self.rewardParam3 = -0.5 # the difference between end time and start time
        # self.rewardParam4 = -0.4 # how long this vehicle need to wait
        # self.rewardParam5 = -0.8 # the different between current timestap and arrived timestamp
        
        # self.randomSeed = 1  # Random Seed
        
        self.embeddingDimension1 = 32 # embedding dimension of DQN
        self.embeddingDimension2 = 32 # embedding dimension of DQN
        self.embeddingDimension3 = 32 # embedding dimension of DQN
        self.outputDimension = 1
        self.nnNodeDimension = 10 # Dimension of features of node, using in DQN, the input dimension for the DQN
        
        # self.numPrevStates = 50000 # the number of maximum previous states stored in the history array
        # self.numQLStep = 2 # the N-Step Q-Learning step paramter.
        # self.gdBatchSize = 32 # Number of samples to perform gradient descent in network
        self.optimizerLR = 0.0001 # the optimizer learning rate
        self.schedulerGamma = 0.99998 # reduce the learning rate as the number of epoches increases

        # self.epsilonMin = 0.1 # Minimum value of Epsilon, for exploitation and exploration
        # self.epsilonDR = 0.0006 # Decay rate of Epsilon, for training the DQN
        
        self.modelFolder = "./model/" # model saving folder
        # self.minEventsForGD = 500 # Minimum number of events required to perform gradient descent in the network
        # self.saveModelThread = 200 # saving model Threshold, when episode reaches n * saveModelThread, store the model paramters

        # global states
        self.numCust = 25 # Number of customer nodes
        self.numDepot = 1 # Number of depots
        self.numNodes = self.numCust + self.numDepot # total number of nodes
        self.minimumDemand = 1 # the minimum demand for a customer
        self.maximumDemand = 10 # the maximum demand for a customer
        self.visitedNodes = [] # the visited nodes storage
        self.nodeLocations = np.zeros((self.numNodes, 2)) # all nodes locations
        self.nodeDistances = np.zeros((self.numNodes, self.numNodes)) # the distance between each other
        self.demands = np.zeros(self.numNodes) # all nodes demands, depot is 0
        self.serviceTime = np.zeros((self.numNodes, 2)) # the service time for all customers the depot is (0, 0)
        self.states =  [[] for y in range(self.numVehicle)] # store all vhicle states
        self.waitingList = {} # key is node index, value is vehicle index
        self.vehicleInWaitingList = [] # store all vehicles that in the waiting list
        self.disabledVehicles = [] # store all vehicles that can not serve any nodes
        # self.disabledNode = [[] for y in range(self.numVehicle)] # store all nodes that can not be served by vehicle i

        self.maskRate = 0.5
        self.resultFile = '../dynamicResult/'
        self.maskedNodes = []
        self.dynamicNodes = []

global config
config = Config()