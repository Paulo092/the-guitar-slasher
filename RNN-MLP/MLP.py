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
    def __init__(self, layersSize, learningRate=0.1, beta=1.0):
        self.learningRate = learningRate
        self.beta = beta
        self.nbInputs = None
        self.nbOutputs = None
        self.layersSize = None
        self.nbLayers = None
        self.layers = None
        if (len(layersSize) == 0):
            return

        self.nbInputs = layersSize[0]
        self.nbOutputs = layersSize[-1]
        self.layersSize = layersSize[1:]
        self.nbLayers = len(self.layersSize)
        self.initializeNeuralNetwork(initializeRandom=True)
    
    def initializeNeuralNetwork(self, initializeRandom=False):
        self.layers = []
        for i in range(self.nbLayers):
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

    def saveStateOnAFile(self, filePathName):
        with open(filePathName, "w") as file:
            #hiperparameters
            file.write("{} {}\n".format(self.learningRate, self.beta))
            file.write("{} {}\n".format(self.nbInputs, self.nbOutputs))
            file.write("{}\n".format(self.nbLayers))
            for layerIdx in range(self.nbLayers):
                if (layerIdx):
                    file.write(" ")
                file.write("{}".format(self.layersSize[layerIdx]))
            file.write("\n")
            #Layers
            for layerIdx in range(self.nbLayers):
                #parameters
                file.write("{} {} {}\n".format(self.layers[layerIdx].nbInputs, self.layers[layerIdx].nbNodes, self.layers[layerIdx].nbOutputs))
                #weights
                for i in range(self.layers[layerIdx].nbNodes):
                    for j in range(self.layers[layerIdx].nbInputs):
                        if (j):
                            file.write(" ")
                        file.write("{}".format(self.layers[layerIdx].weights.matrix[i][j]))
                    file.write("\n")

    def loadStateOnAFile(self, filePathName):
        with open(filePathName, "r") as file:
            def getFileLine(useFloat=False):
                return [float(value) if (useFloat) else int(value) for value in file.readline().split()]

            self.learningRate, self.beta = getFileLine(True)
            self.nbInputs, self.nbOutputs = getFileLine()
            self.nbLayers, = getFileLine()
            self.layersSize = getFileLine()
            self.initializeNeuralNetwork()
            for layerIdx in range(self.nbLayers):
                nbInputs, nbNodes, _ = getFileLine()
                for i in range(nbNodes):
                    values = getFileLine(useFloat=True)
                    for j in range(nbInputs):
                        self.layers[layerIdx].weights.matrix[i][j] = values[j]


if (__name__ == '__main__'):
    nn = NeuralNetwork([])
    nn.loadStateOnAFile("./RNN-MLP/test.txt")
    print(nn.nbLayers)
    print(nn.layersSize)
    for i in range(nn.nbLayers):
        print(nn.layers[i].weights)