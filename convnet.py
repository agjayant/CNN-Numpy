import numpy as np
import config

inp_width = config.width
inp_height = config.height
inp_channels = config.channels

layers = config.layers

minW = config.minW
maxW = config.maxW

initBias = config.initBias

class cnn:
    def __init__(self):

        self.Weights = [np.random.uniform(minW,maxW,size=(layers[0][1],layers[0][2],layers[0][2]))]
        self.Weights.append(np.random.uniform(minW,maxW,size=(layers[2][1],layers[2][2],layers[2][2],layers[0][1])))
        finalPoolOut = 4
        self.Weights.append(np.random.uniform(minW,maxW,size=(layers[4][1],finalPoolOut,finalPoolOut,layers[2][1])))
        self.Weights.append(np.random.uniform(minW,maxW,size=(layers[5][1],layers[4][1])))
        self.Biases = [initBias*np.ones(layers[5][1])]
        self.Weights.append(np.random.uniform(minW,maxW,size=(layers[6][1],layers[5][1])))
        self.Biases.append(initBias*np.ones(layers[6][1]))

    def forward(self, inputData):

        weights = self.Weights
        biases = self.Biases

        # Forward Pass Code
