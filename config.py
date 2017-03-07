############################
## Network Parameters     ##
############################

#Input Size
height = 28
width = 28
channels = 1

#Layers
layers = [ ('conv',6, 5, 1), ('pool',2,2), ('conv',16,5, 1 ),('pool',2,2),('fc',120),('fc',84), ('softmax',10)  ]

activation = 'relu'
pool = 'max' # 'mean' or 'max'

#Network Initialisation
minW = -0.1
maxW =  0.1       # Network weights are initialised in range [minW,maxW]
initBias = 0.01   # Initial Bias Value for all layers

###########################
## Training Parameters   ##
###########################

lr = 0.001
numEpoch = 5
batchSize = 5
trainExamples = 20
