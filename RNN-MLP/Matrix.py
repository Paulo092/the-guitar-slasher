from random import random

class Matrix:
    def __init__(nbRows, nbCols, initializeRandom=False):
        self.matrix = None
        self.nbRows = nbRows
        self.nbCols = nbCols
        self.initializeMatrix()
    
    def __getitem__(self, key):
        return self.matrix[key]
    
    def __mul__(self, otherMatrix):
        return self.multiply(otherMatrix)

    def initializeMatrix():
        self.matrix = [
                [ random() if (initializeRandom) else 0 for j in range(nbCols) ]
                    for i in range(nbRows)
            ]

    def getNbRows():
        return self.nbRows
    
    def getNbCols():
        return self.nbCols

    def multiply(self, otherMatrix):
        if (self.nbCols != otherMatrix.nbRows):
            raise Exception("Multiply Matrix: self.nbCols != otherMatrix.nbRows.")
        ans = Matrix(self.nbRows, otherMatrix.nbCols)
        for i in range(self.nbRows):
            for j in range(otherMatrix.getNbCols()):
                for k in range(self.nbCols):
                    ans[i][j] += self.matrix[i][k] * otherMatrix[k][j]
        return ans