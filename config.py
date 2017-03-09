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
initBias = 0.01   # Initial Bias Value for all layers

###########################
## Training Parameters   ##
###########################

alpha = 0.9 # Momentum
lr = 0.01
numEpoch = 100
batchSize = 2
trainExamples = 10
