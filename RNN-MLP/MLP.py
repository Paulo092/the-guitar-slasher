from math import exp as mathEulerExp

from Matrix import Matrix

class LayerNeuralNetwork():
    def __init__(self, nbInputs, nbNodes, initializeRandom):
        self.nbInputs = nbInputs
        self.nbNodes = nbNodes
        self.nbOutputs = nbNodes+1
        self.input = None
        self.weights = None
        self.outputI = None
        self.outputY = None
        self.initializeLayer(initializeRandom)

    def initializeLayer(self, initializeRandom):
        self.input = Matrix(self.nbInputs, 1)
        self.weights = Matrix(self.nbNodes, self.nbInputs, initializeRandom=True)
        self.outputI = Matrix(self.nbNodes, 1)
        self.outputY = Matrix(self.nbOutputs, 1)

#MultiLayerPerceptron implementation
#BIAS will be always the last one
class NeuralNetwork():
    def __init__(self, layersSize, beta=1.0):
        self.beta = beta

        self.nbInputs = layersSize[0]
        self.nbOutputs = layersSize[-1]
        self.layersSize = layersSize[1:]
        self.nbLayers = len(self.layersSize)
        self.layers = None
        self.initializeNeuralNetwork()
    
    def initializeNeuralNetwork(self):
        self.layers = []
        print(self.nbLayers)
        for i in range(self.nbLayers):
            print("--> ", i)
            nbInputs = (self.layersSize[i-1] if (i) else self.nbInputs) + 1 #to BIAS
            nbNodes = self.layersSize[i]
            layer = LayerNeuralNetwork(nbInputs, nbNodes, initializeRandom=True)
            if (i):
                layer.input = self.layers[-1].outputY
            self.layers.append(layer)

    def insertInputs(self, inputs):
        for i in range(self.nbInputs):
            self.layers[0].input[i] = inputs[i]
        self.layers[0].input[self.nbInputs] = -1

    def logisticFunction(self, value):
        return 1.0 / (1.0 + mathEulerExp(-1.0 * self.beta * value))

    def hyperbolicTangent(self, value):
        expValue = mathEulerExp(-1.0 * self.beta * value)
        return (1.0 - expValue) / (1.0 + expValue)

    def activationFunction(self, value):
        return self.hyperbolicTangent(value)

    def computeOutputI(self, layerIdx):
        self.layers[layerIdx].outputI = self.layers[layerIdx].weights * self.layers[layerIdx].input
    
    def computeOutputY(self, layerIdx):
        for i in range(self.layers[layerIdx].nbNodes):
            self.layers[layerIdx].outputY[i] = self.activationFunction(self.layers[layerIdx].outputI[i])
        self.layers[layerIdx].outputY[self.layers[layerIdx].nbNodes] = -1

    def forward(self):
        for layerIdx in range(self.nbLayers):
            self.computeOutputI(layerIdx)
            self.computeOutputY(layerIdx)

    def getAns(self, idx):
        return self.layers[-1].outputY[idx]

if (__name__ == '__main__'):
    nn = NeuralNetwork([2, 3, 2, 1])
    print(nn.nbLayers)
    nn.insertInputs([1, 1])

    nn.forward()
    for i in range(0, nn.nbLayers):
        print(nn.layers[i].outputY)