"""
The Artificial Neural Network
"""
import numpy as np
import scipy
import math

class ArtificialNeuralNetwork(object):


    def __init__(self, gamma, layer_sizes, num_hidden, epsilon=None, max_iters=None):
        """
        Construct an artificial neural network classifier

        @param gamma : weight decay coefficient
        @param layer_sizes:  Number of hidden layers
        @param num_hidden:  Number of hidden units in each hidden layer
        @param epsilon : cutoff for gradient descent
                         (need at least one of [epsilon, max_iters])
        @param max_iters : maximum number of iterations to run
                            gradient descent for
                            (need at least one of [epsilon, max_iters])
        """
        self.gamma = gamma
        self.layer_sizes = layer_sizes
        self.num_hidden = num_hidden
        self.epsilon = epsilon
        self.max_iters = max_iters
        if self.layer_sizes > 1:
            print "Did not design for more than 1 hidden layer."

        
    def fit(self, X, y, sample_weight=None):
        """ Fit a neural network of layer_sizes * num_hidden hidden units using X, y """
        self.numAttributes = len(X[0])
        if self.layer_sizes == 0:
            self.hiddenWeights = None
            self.outputWeights = np.random.uniform(-0.1, 0.1, (self.numAttributes))
        else:
            #make random weight matrices
            self.hiddenWeights = np.random.uniform(-0.1, 0.1, (self.num_hidden,self.numAttributes))
            self.outputWeights = np.random.uniform(-0.1, 0.1, (self.num_hidden))

        for _ in range(self.max_iters):
            #print self.outputWeights
            self.updateWeights(X,y,sample_weight)
        return

    def updateWeights(self, X, y, sample_weight=None):
        examples = np.array(X, np.float)
        if self.hiddenWeights is not None:
            hiddenOutputs = self.getAllHiddenOutputs(X) 

            #import pdb;pdb.set_trace()
            #print hiddenOutputs
            outputs = list()
            for i in range(len(X)):
                h = hiddenOutputs[i]
                outputSum = h.dot(self.outputWeights)
                outputs.append(ArtificialNeuralNetwork.sigmoid(outputSum))
            
            outputUpdates = self.getOutputUpdates(hiddenOutputs, outputs, y)
            #print outputUpdates
            hiddenUpdates = self.getHiddenUpdates(hiddenOutputs, outputs, outputUpdates, X)
            #print hiddenUpdates

            self.applyUpdates(outputUpdates, hiddenUpdates)

    def getAllHiddenOutputs(self,X):
        toReturn = list()
        for i in range(len(X)):
            curExample = X[i]
            toReturn.append(self.getHiddenOutputs(curExample))
        return toReturn

    def getHiddenOutputs(self,curExample):
        toReturn = np.zeros((self.num_hidden),np.float)
        for j in range(self.num_hidden):
            toReturn[j] = ArtificialNeuralNetwork.sigmoid(curExample.dot(self.hiddenWeights[j]))
        return toReturn
            

    @staticmethod
    def applySigmoid(matrix):
        toReturn = np.zeros((len(matrix),len(matrix[0])), np.float)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                toReturn[i,j] = ArtificialNeuralNetwork.sigmoid(matrix[i,j])
        return toReturn

    def getOutputUpdates(self, hiddenOutputs, outputs, y):
        toReturn = list(0.0 for _ in range(self.num_hidden))
        for i in range(len(y)):
            yi = y[i]
            hn = outputs[i]
            for j in range(self.num_hidden):
                xji = hiddenOutputs[i][j]
                toReturn[j] += (hn - yi)*hn*(1 - hn)*xji
        return toReturn

    def getHiddenUpdates(self,hiddenOutputs, outputs, outputUpdates, X):
        toReturn = np.zeros((self.num_hidden,self.numAttributes),np.float)
        for i in range(len(X)):
            for j in range(self.numAttributes):
                xji = X[i][j]
                for k in range(self.num_hidden):
                    hn = hiddenOutputs[i][k]
                    outputUpdate = outputUpdates[k]
                    outputWeight = self.outputWeights[k]
                    toReturn[k,j] += (1-hn)*xji*outputUpdate*outputWeight
        return toReturn

    def applyUpdates(self, outputUpdates, hiddenUpdates):
        for i in range(self.num_hidden):
            self.outputWeights[i] -= 0.01*(outputUpdates[i] + self.gamma*self.outputWeights[i])
        for i in range(self.numAttributes):
            for j in range(self.num_hidden):
                self.hiddenWeights[j,i] -= 0.01*(hiddenUpdates[j,i] + self.gamma*self.hiddenWeights[j,i])
        return

    def predict(self, X):
        """ Return the -1/1 predictions of the decision tree """
        #The list that will hold the predictions
        predictions = self.predict_proba(X)
        toReturn = list()
        for prob in predictions:
            #If the confidence is above or equal to 0.5, we will predict positive
            if prob >= 0:
                toReturn.append(1)
            #Otherwise, predict negative
            else:
                toReturn.append(-1)
        return toReturn

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        toReturn = list()
        for i in range(len(X)):
            curExample = X[i]
            hiddenLayerOutputs = self.getHiddenOutputs(curExample)
            toReturn.append(ArtificialNeuralNetwork.sigmoid(hiddenLayerOutputs.dot(self.outputWeights)))

        return toReturn

    def predict_proba_example(self, example):
        return self.calc_output(self.outputPerceptron, self.layer_sizes, example)

    def calc_output(self, curPerceptron, curLayer, example):
        sumOfWXs = 0.0
        x = self.calc_xVector(curPerceptron, curLayer, example)
        for i in range(curPerceptron.numInputs):
            sumOfWXs += x[i]*curPerceptron.w[i]
        return ArtificialNeuralNetwork.sigmoid(sumOfWXs)

    def calc_xVector(self, curPerceptron, curLayer, example):
        xVector = list()
        for i in range(curPerceptron.numInputs):
            x = None
            if curLayer > 0:
                prevLayer = curLayer - 1
                xVector.append(self.calc_output(self.layers[prevLayer][i], prevLayer, example))
            else:
                xVector.append(example[i])
        return xVector

    @staticmethod
    def sigmoid(wx):
        return 2*(1 / (1 + np.exp(-wx))) - 1

    def buildLayer(self):
        newLayer = list()
        for i in range(self.num_hidden):
            newLayer.append(Perceptron(self.num_hidden))
        return newLayer

    def buildLowestLayer(self):
        self.lowestLayer = list()
        for i in range(self.num_hidden):
            self.lowestLayer.append(Perceptron(self.numAttributes, True))
        return self.lowestLayer
        
class Perceptron(object):

    def __init__(self, numInputs, isLowest = False):
        self.isLowest = isLowest
        self.numInputs = numInputs
        self.w = Perceptron.generateRandomW(numInputs)
        
    @staticmethod
    def generateRandomW(numInputs):
        w = list()
        for i in range(numInputs):
            randf = np.random.random()
            randf = randf / 5
            randf = randf - 0.1
            w.append(randf)
        return w
