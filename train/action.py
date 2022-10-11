from config import config

import random

class Event():
    '''
        Event Class which is stored in the history. History Class is a collection of Events (Transitions)  
    '''
    def __init__(self, state, nextState, reward, action):
        self.state = state # Current State
        self.stateTensor = state.getTensor() # Current State Tensor
        self.nextState = nextState # Next State
        self.nextStateTensor = nextState.getTensor() # Next State Tensor
        self.reward = reward # Reward obtained
        self.action = action # Action performed
        
class History(): # replay memory
    def __init__(self):
        global config
        self.pastEvents = []
        self.numEvents = 0
    
    def addPastEvent(self, event):
        '''
            Add an event to History
        '''
        global config
        # Override a past event if capacity if full
        if self.numEvents > config.numPrevStates:
            self.pastEvents[self.numEvents % config.numPrevStates] = event
        else:
            self.pastEvents.append(event)
        self.numEvents += 1 # Update the number of events in history
    
    def getNumPastEvents(self):
        '''
            Get the number of events in History  
        '''
      # If more than capacity, return capacity
        if self.numEvents > len(self.pastEvents):
            return len(self.pastEvents)
        else:
            return self.numEvents
    
    def getSamplePast(self):
        '''
            Sample a batch of events from the history  
        '''
        global config
        return random.sample(self.pastEvents, config.gdBatchSize) # gdBatchSize = 25
    
    def addToHistory(self, visitedNodes, rewards, states, newState, actions):
        '''
            Add event and past states based on the N-Step Q Learning
        '''
        global config
        # Add previous N-States information
        if len(visitedNodes) >= config.numQLStep:
            lastState = states[-config.numQLStep]
            rewardsSum = rewards[-config.numQLStep + 1]
            actionPerformed = actions[-config.numQLStep + 1]
            newEvent = Event(lastState, newState, rewardsSum, actionPerformed)
            self.addPastEvent(newEvent)
        
        # If we reached a final state, add all the states results in the final state
        if newState.isFinalState():
            for step in range(1, config.numQLStep):
                lastState = states[-step]
                rewardsSum = sum(rewards[-step:])
                actionPerformed = actions[-step]
                event = Event(lastState, newState, rewardsSum, actionPerformed)
                self.addPastEvent(event)