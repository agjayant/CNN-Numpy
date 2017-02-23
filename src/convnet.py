import numpy as np
from scipy import signal
import activations as act
import sys
sys.path.append('../')
import config
inp_width = config.width
inp_height = config.height
inp_channels = config.channels

layers = config.layers

minW = config.minW
maxW = config.maxW

initBias = config.initBias
activation = config.activation
pool = config.pool

def convFwd(X, convFilters, bias):

    featureMaps = []
    for i in range(len(convFilters)):
        featureMap = []
        convFilter = convFilters[i]
        depth = len(convFilter)
        assert(depth == len(X)), 'Dimension Mismatch'
        for j in range(depth):
            featureMap.append(signal.convolve2d(X[j], convFilter[j],'valid'))
        featureMap = sum(np.asarray(featureMap)) + bias[i]

        featureMaps.append(act.activation(featureMap,'relu'))
    return np.asarray(featureMaps)

def poolFwd(X, kernel, stride):
    assert( kernel == 2 ), 'Only Size 2 Kernel Supported Currently'
    assert( stride == 2 ), 'Only Size 2 Stride Supported Currently'
    postPool = []
    for i in range(len(X)):
        pre = X[i].reshape(X[i].shape[0]/2,2,X[i].shape[1]/2,2)
        if pool == 'max':
            postPool.append(pre.max(axis=(1,3)))
        elif pool == 'mean':
            postPool.append(pre.mean(axis=(1,3)))
        else :
            assert(1==2),'Invalid Pool option'
    return np.asarray(postPool)

class cnn:
    def __init__(self):

        self.Weights = [np.random.uniform(minW,maxW,size=(layers[0][1],inp_channels,layers[0][2],layers[0][2]))]
        out_Size =  inp_width - layers[0][2] + 1 ########### Only for Height = Width
        self.Biases = [initBias*np.ones((layers[0][1], out_Size,out_Size))]

        self.poolParams = [(layers[1][1], layers[1][2])]
        out_Size = out_Size/2  ########## Only for Kernel = 2 and Stride = 2

        self.Weights.append(np.random.uniform(minW,maxW,size=(layers[2][1],layers[0][1],layers[2][2],layers[2][2])))
        out_Size = out_Size - layers[2][2]+1
        self.Biases.append(initBias*np.ones((layers[2][1], out_Size,out_Size)))

        self.poolParams.append((layers[3][1],layers[3][2]))
        out_Size = out_Size/2  ########## Only for Kernel = 2 and Stride = 2

        self.Weights.append(np.random.uniform(minW,maxW,size=(layers[4][1],layers[2][1],out_Size,out_Size)))
        out_Size = 1
        self.Biases.append(initBias*np.ones((layers[4][1],out_Size,out_Size)))

        self.Weights.append(np.random.uniform(minW,maxW,size=(layers[5][1],layers[4][1])))
        self.Biases.append(initBias*np.ones(layers[5][1]))

        self.Weights.append(np.random.uniform(minW,maxW,size=(layers[6][1],layers[5][1])))
        self.Biases.append(initBias*np.ones(layers[6][1]))

    def forward(self, inputData):

        weights = self.Weights
        biases = self.Biases
        poolParams = self.poolParams

        # layer0 = input Layer
        X = np.asarray(inputData)
        # layer0 = X.reshape(X.size**0.5,X.size**0.5)
        layer0 = X

        # layer1 = conv1 layer
        layer1 = convFwd(np.asarray([layer0]),weights[0] , biases[0])

        #layer2 = pool1 layer
        layer2 = poolFwd(layer1, poolParams[0][0], poolParams[0][1])

        # layer3 = conv2 layer
        layer3 = convFwd(layer2,weights[1], biases[1] )

        #layer4 = pool2 layer
        layer4 = poolFwd(layer3, poolParams[1][0], poolParams[1][1])

         # layer5 = fc1 layer
        layer5 = convFwd( layer4,weights[2] ,biases[2] )

        # layer6 = fc2 layer
        layer6 = act.activation(np.dot(weights[3],layer5[:,0])+biases[3] , 'relu')

        #layer7 = softmax layer
        layer7_in = act.activation(np.dot( weights[4], layer6[:,0] )+biases[4] , 'relu')
        layer7 = np.exp(layer7_in)/sum(np.exp(layer7_in))
        return layer7

    def predict(self, inputVal):

        outProb = self.forward(inputVal)
        return outProb.argmax()





