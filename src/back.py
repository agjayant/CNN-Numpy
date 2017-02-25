from scipy import signal
import numpy as np
import activations as act
import sys
sys.path.append('../')
import config

def conv(X, convFilters):

    featureMaps = []
    for i in range(len(convFilters)):
        featureMap = []
        convFilter = convFilters[i]
        depth = len(convFilter)
        assert(depth == len(X)), 'Dimension Mismatch'
        for j in range(depth):
            featureMap.append(signal.convolve2d(X[j], convFilter[j],'valid'))

        featureMaps.append( sum(featureMap) )

    return np.asarray(featureMaps)


def convBack(X, dy, W):

    ### Compute dx, dW , dB

    dy = np.pad(dy, W.shape[2]-1 ,'constant' )[1:-1]

    Wb = np.zeros((W.shape[1],W.shape[0],W.shape[2],W.shape[3]))
    dW = []
    dB = []

    for i in range(W.shape[0]):
        kernel = []
        for j in range(W.shape[1]):
            kernel.append( signal.convolve2d(X[j], dy[i],'valid') )
            Wb[j,i] = np.rot90(W[i,j],2)
        dW.append(np.asarray(kernel))
        dB.append(sum(dy[i]))

    dX = conv(dy, Wb)

    dW = np.asarray(dW)
    dB = np.asarray(dB)

    return [dX, dW, dB]

# def convBack(X, dy, W):

    ## Compute dx

    # dy = np.pad(dy, W.shape[2]-1 ,'constant' )[1:-1]

    # Wb = []
    # for j in range(W.shape[1]):
        # kernel = []
        # for i in range(W.shape[0]):
            # kernel.append( np.rot90(W[i,j],2) )
        # Wb.append(np.asarray(kernel))

    # dX = conv(dy, Wb)

    ## Compute dW

    # dW = []
    # for i in range(dy.shape[0]):
        # kernel = []
        # for j in range(X.shape[0]):
            # kernel.append( signal.convolve2d(X[j], dy[i],'valid') )
        # dW.append(np.asarray(kernel))

    # dW = np.asarray(dW)

    ## Compute dB

    # dB = []
    # for i in range(dy.shape[0]):
        # dB.append(sum(dy[i]))
    # dB = np.asarray(dB)

    ## Return dX, dW, dB

    # return [dX, dW, dB]
