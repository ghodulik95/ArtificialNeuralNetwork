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
            print self.hiddenWeights
            self.updateWeights(X,y,sample_weight)
        return

    def updateWeights(self, X, y, sample_weight=None):
        if self.hiddenWeights is not None:
            #Get the outputs of the hidden layer for each example
            #   hiddenOutputs[i] corresponds to all the outputs of the hidden layer for example i
            hiddenOutputs = self.getAllHiddenOutputs(X) 

            #mport pdb;pdb.set_trace()
            #print hiddenOutputs
            #Get all the outputs for the examples
            outputs = list()
            for i in range(len(X)):
                h = hiddenOutputs[i]
                outputSum = np.dot(h,self.outputWeights)
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
            toReturn[j] = ArtificialNeuralNetwork.sigmoid(np.dot(curExample,self.hiddenWeights[j]))
        return toReturn
            

    def getOutputUpdates(self, hiddenOutputs, outputs, y):
        toReturn = list(self.gamma*self.outputWeights[i] for i in range(self.num_hidden))
        #initialize the list to the weight decay terms
        for i in range(len(y)):
            yi = y[i]
            hn = outputs[i]
            for j in range(self.num_hidden):
                xji = hiddenOutputs[i][j]
                #print xji
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
            self.outputWeights[i] -= 0.01*(outputUpdates[i])
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
            if prob > 0.5:
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
            toReturn.append(ArtificialNeuralNetwork.sigmoid(np.dot(hiddenLayerOutputs,self.outputWeights)))

        return toReturn

    @staticmethod
    def sigmoid(wx):
        return 1 / (1 + np.exp(-wx))