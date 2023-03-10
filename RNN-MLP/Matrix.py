from random import random

class Matrix:
    def __init__(self, nbRows, nbCols, initializeRandom=False):
        self.matrix = None
        self.nbRows = nbRows
        self.nbCols = nbCols
        self.initializeMatrix(initializeRandom)
    
    def __getitem__(self, key):
        if (self.nbRows == 1):
            return self.matrix[0][key]
        if (self.nbCols == 1):
            return self.matrix[key][0]
        return self.matrix[key]
    
    def __setitem__(self, key, item):
        if (self.nbRows == 1):
            self.matrix[0][key] = item
        if (self.nbCols == 1):
            self.matrix[key][0] = item
    
    def __mul__(self, otherMatrix):
        return self.multiply(otherMatrix)

    def initializeMatrix(self, initializeRandom):
        self.matrix = [
                [ random() if (initializeRandom) else 0 for j in range(self.nbCols) ]
                    for i in range(self.nbRows)
            ]

    def getNbRows(self):
        return self.nbRows
    
    def getNbCols(self):
        return self.nbCols

    def multiply(self, otherMatrix):
        if (self.nbCols != otherMatrix.nbRows):
            raise Exception("Multiply Matrix: self.nbCols != otherMatrix.nbRows.")
        ans = Matrix(self.nbRows, otherMatrix.nbCols)
        for i in range(self.nbRows):
            for j in range(otherMatrix.getNbCols()):
                for k in range(self.nbCols):
                    ans.matrix[i][j] += self.matrix[i][k] * otherMatrix.matrix[k][j]
        return ans

    def __str__(self):
        maxSizeValue = 0
        for i in range(self.nbRows):
            for j in range(self.nbCols):
                maxSizeValue = max(maxSizeValue, len(str(self.matrix[i][j])))
        printedMatrix = "Matrix {}x{}\n".format(self.nbRows, self.nbCols)
        for i in range(self.nbRows):
            printedMatrix += "[ "
            for j in range(self.nbCols):
                if (j):
                    printedMatrix += ", "
                printedMatrix += str(self.matrix[i][j])
                printedMatrix += " " * (maxSizeValue-len(str(self.matrix[i][j])))
            printedMatrix += " ]\n"
        return printedMatrix
