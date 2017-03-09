import numpy as np
import scipy.io as scio
import sys
sys.path.append('../')
from convnet import cnn
import config

numEpoch = config.numEpoch
trainExamples = config.trainExamples
batchSize = config.batchSize

net = cnn()
mnist = scio.loadmat('../data/mnist_2D.mat')

for epoch in range(numEpoch):

    trainList = [np.random.randint(0,60000) for i in range(trainExamples)]
    # valList = [np.random.randint(0,10000) for i in range(valExamples)]

    trainLabel = np.asarray([[0 for i in range(10)] for j in range(trainExamples)])
    trainData = np.zeros(( trainExamples, mnist['X_train'][0].shape[0], mnist['X_train'][0].shape[1] ))

    # valLabel = np.asarray([0 for i in range(valExamples)])
    # valData = np.asarray([[0 for i in range(config.n_inputs)] for j in range(valExamples)])

    j=0
    for i in trainList:
        trainLabel[j,mnist['Y_train'][i]] = 1
        trainData[j] = mnist['X_train'][i]
        j += 1

    # j=0
    # for i in valList:
        # valLabel[j] = mnist['Y_test'][i]
        # valData[j] = mnist['X_test'][i]
        # j += 1

    j = 0
    numIter = 1
    while( j < trainExamples ):

        batchData = trainData[j:j+batchSize]
        batchLabel = trainLabel[j:j+batchSize]

        batchLoss = net.backward(batchData, batchLabel)

        print 'Iteration ', numIter,': Train Loss =  ', batchLoss

        numIter += 1
        j += batchSize

    ### Debugging: For Overfitting exercise
    acc = 0
    for i in range(trainExamples):
        if trainLabel[i][net.predict(trainData[i])] == 1:
            acc += 1

    print 'Epoch ', epoch+1, " : Accuracy:  ", acc*100.0/trainExamples



