import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return x * (1 - x)

trainingInputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
trainingOutputs = np.array([[0,1,1,0]]).T

synapticWeights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synapticWeights)

for iteration in range(20000):
    inputLayer = trainingInputs
    outputs = sigmoid(np.dot(inputLayer, synapticWeights))
    error = trainingOutputs - outputs
    adjustments = error * sigmoidDerivative(outputs)
    synapticWeights += np.dot(inputLayer.T, adjustments)

print('Synaptic weights after training: ')
print(synapticWeights)
print('Outputs after training: ')
print(outputs)
