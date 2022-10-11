from config import config

import torch.nn as nn

class DQN(nn.Module):
    '''
        Neural Network for Q-Learning, 1 input layer, 2 hidden layers and 1 output layer.
    '''
    def __init__(self):
        super(DQN, self).__init__()
        
        global config
        self.linear1 = nn.Linear(config.nnNodeDimension, config.embeddingDimension1, True)
        self.linear2 = nn.Linear(config.embeddingDimension1, config.embeddingDimension2, True)
        self.linear3 = nn.Linear(config.embeddingDimension2, config.embeddingDimension3, True)
        self.output = nn.Linear(config.embeddingDimension3, config.outputDimension, True)
    
    def forward(self, features):
        global config
        out = nn.functional.tanh(self.linear1(features))
        out = nn.functional.tanh(self.linear2(out))
        out = nn.functional.tanh(self.linear3(out))
        out = self.output(out)
        return out.squeeze(dim = 2).float()