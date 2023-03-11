from math import exp as mathEulerExp, tanh

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
        self.delta = None
        self.initializeLayer(initializeRandom)

    def initializeLayer(self, initializeRandom):
        self.input = Matrix(self.nbInputs, 1)
        self.weights = Matrix(self.nbNodes, self.nbInputs, initializeRandom=True)
        self.outputI = Matrix(self.nbNodes, 1)
        self.outputY = Matrix(self.nbOutputs, 1)
        self.delta = Matrix(self.nbNodes, 1)

#MultiLayerPerceptron implementation
#BIAS will be always the last one
class NeuralNetwork():
    def __init__(self, layersSize, learningRate=0.01, precision=0.0000001, beta=1.0, maxEpochs=15000):
        self.learningRate = learningRate
        self.precision = precision
        self.beta = beta
        self.maxEpochs = maxEpochs
        self.nbInputs = None
        self.nbOutputs = None
        self.layersSize = None
        self.nbLayers = None
        self.layers = None
        self.epochs = 0
        
        self.nbSamples = None
        self.dataset = None
        self.loadedDataset = False

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
        #return (mathEulerExp(value)-mathEulerExp(-value))/(mathEulerExp(value)+mathEulerExp(-value))
        expValue = mathEulerExp(-1.0 * self.beta * value)
        return (1.0 - expValue) / (1.0 + expValue)

    def derivatedHyperbolicTangent(self, value):
        #return 1-self.hyperbolicTangent(value)**2
        expValue = mathEulerExp(self.beta * value)
        return (2.0 * self.beta * expValue) / ((expValue + 1.0) * (expValue + 1.0))

    def activationFunction(self, value):
        return self.hyperbolicTangent(value)
    
    def derivatedActivationFunction(self, value):
        return self.derivatedHyperbolicTangent(value)

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

    def loadDatasetFromAFile(self, filePathName):
        with open(filePathName, "r") as file:
            def getFileLine(useFloat=False):
                return [float(value) if (useFloat) else int(value) for value in file.readline().split()]
            
            nbInputs, nbOutputs = getFileLine()
            if (nbInputs != self.nbInputs or nbOutputs != self.nbOutputs):
                raise Exception("Neural Network: loadedDatasetFromAFile -> nbInputs != self.nbInputs or nbOutputs != self.nbOutputs.")
            
            self.nbSamples, = getFileLine()
            self.dataset = [[] for _ in range(self.nbSamples)]
            for i in range(self.nbSamples):
                self.dataset[i].append(getFileLine(useFloat=True))
                self.dataset[i].append(getFileLine(useFloat=True))
            
        self.loadedDataset = True

    def quadraticErrorOfASample(self, sampleIdx):
        if (not self.loadedDataset):
            raise Exception("NeuralNetwork: quadraticErrorOfASample -> Dataset não carregado.")
        self.insertInputs(self.dataset[sampleIdx][0])
        self.forward()
        quadraticError = 0
        for i in range(self.nbOutputs):
            quadraticError += (self.dataset[sampleIdx][1][i] - self.getAns(i)) * (self.dataset[sampleIdx][1][i] - self.getAns(i))
        return quadraticError / 2.0

    def rootMeanSquareError(self):
        if (not self.loadedDataset):
            raise Exception("NeuralNetwork: rootMeanSquareError -> Dataset não carregado.")
        meanSquareError = 0
        for i in range(self.nbSamples):
            meanSquareError += self.quadraticErrorOfASample(i)
        return meanSquareError / self.nbSamples

    def train(self):
        if (not self.loadedDataset):
            raise Exception("NeuralNetwork: rootMeanSquareError -> Dataset não carregado.")
        
        while self.epochs < self.maxEpochs:
            meanSquareErrorBefore = self.rootMeanSquareError()
            print("Epoch {}".format(self.epochs))
            self.trainEpoch()
            meanSquareErrorAfter = self.rootMeanSquareError()
            if (abs(meanSquareErrorAfter - meanSquareErrorBefore) <= self.precision):
                break

    def trainEpoch(self):
        for sampleIdx in range(self.nbSamples):
            self.insertInputs(self.dataset[sampleIdx][0])
            self.forward()
            for layerIdx in range(self.nbLayers-1, -1, -1):
                for j in range(self.layers[layerIdx].nbNodes):
                    if (layerIdx+1 != self.nbLayers): #is not the last layer
                        firstTerm = 0
                        for k in range(self.layers[layerIdx+1].nbNodes):
                            firstTerm += self.layers[layerIdx+1].delta[k] * self.layers[layerIdx+1].weights.matrix[k][j]
                        firstTerm *= -1.0
                    else:
                        firstTerm = self.dataset[sampleIdx][1][j] - self.getAns(j)
                    self.layers[layerIdx].delta[j] = firstTerm * self.derivatedActivationFunction(self.layers[layerIdx].outputI[j])

                for j in range(self.layers[layerIdx].nbNodes):
                    for i in range(self.layers[layerIdx].nbInputs):
                        #print(self.layers[layerIdx].delta.matrix)
                        self.layers[layerIdx].weights.matrix[j][i] += self.learningRate * self.layers[layerIdx].delta[j] * self.layers[layerIdx].input[i]
        self.epochs += 1

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

    def loadStateFromAFile(self, filePathName):
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
    nn = NeuralNetwork([2, 2, 1])
    nn.loadDatasetFromAFile("./RNN-MLP/Tests-Datasets/xor-problem-dataset.txt")
    print(nn.rootMeanSquareError())
    for layerIdx in range(nn.nbLayers):
        print(nn.layers[layerIdx].nbOutputs)
    nn.train()
    print(nn.rootMeanSquareError())

    for i in range(nn.nbSamples):
        nn.insertInputs(nn.dataset[i][0])
        nn.forward()
        print(nn.dataset[i][0])
        print(nn.getAns(0))