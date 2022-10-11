class Config:
    def __init__(self):
        self.timer = 0 # the timer
        
        self.capacity = 200 # Capacity of truck
        self.speed = 1 # the speed of vehicle
        self.numCust = 13 # Number of customer nodes
        self.numDepot = 1 # Number of depots
        self.numNodes = self.numCust + self.numDepot # total number of nodes
        self.minimumDemand = 10 # the minimum demand for a customer
        self.maximumDemand = 50 # the maximum demand for a customer

        self.startTime = 0 # the time vhicles start service
        self.endTime = 1236 # the time vhicles must return back to the depot
        self.totalDuration = self.endTime - self.startTime # vehicles service duration
        self.serviceDuration = 90 # the service duration for each customer

        self.maxDistanceToDepot = int((self.totalDuration - self.serviceDuration) * self.speed // 2) # the maximum distance between the depot and each customers
        
        # nodes will spread within the block minimumLocationValue - maximumLocationValue
        # assume the location of depot is (0, 0), so the maximumLocationValue will be maxDistanceToDepot
        self.minimumLocationValue = 0
        self.maximumLocationValue = self.maxDistanceToDepot
        
        # reward parameters
        self.rewardParam1 = -0.1 # total distance parameter
        self.rewardParam2 = -3.0 # the distance to next new node parameter
        self.rewardParam3 = -0.5 # the difference between end time and start time
        self.rewardParam4 = -0.4 # how long this vehicle need to wait
        self.rewardParam5 = -0.8 # the different between current timestap and arrived timestamp

        self.qValueGamma = 0.99 # exptected q value parameter

        self.numEpisode = 50000 # Number of Episodes, train the model in 10000 episodes
        
        self.embeddingDimension1 = 32 # embedding dimension of DQN
        self.embeddingDimension2 = 32 # embedding dimension of DQN
        self.embeddingDimension3 = 32 # embedding dimension of DQN
        self.outputDimension = 1

        
        self.nnNodeDimension = 10 # Dimension of features of node, using in DQN, the input dimension for the DQN
        
        self.numPrevStates = 50000 # the number of maximum previous states stored in the history array
        self.numQLStep = 2 # the N-Step Q-Learning step paramter.
        self.gdBatchSize = 32 # Number of samples to perform gradient descent in network
        self.epsilonMin = 0.1 # Minimum value of Epsilon, for exploitation and exploration
        self.epsilonDR = 0.0001 # Decay rate of Epsilon, for training the DQN 0.0001
        
        self.modelFolder = "./model/1/" # model folder
        self.minEventsForGD = 1000 # Minimum number of events required to perform gradient descent in the network
        self.modelSavingTHreshold = 200 # save model Threshold, when episode reaches n * modelSavingTHreshold, store the model paramters

        self.lr = 0.001
        self.gamma = 0.998

        self.data = {}

global config
config = Config()