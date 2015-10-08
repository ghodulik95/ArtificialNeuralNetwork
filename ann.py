"""
The Artificial Neural Network
"""
import numpy as np
import scipy

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
        self.learningRate = 0.01
        if self.layer_sizes > 1:
            print "Did not design for more than 1 hidden layer."

        
    def fit(self, X, y, sample_weight=None):
        """ Fit a neural network of layer_sizes * num_hidden hidden units using X, y """
        self.numAttributes = len(X[0])
        if self.layer_sizes == 0 or self.num_hidden == 0:
            self.hiddenWeights = None
            self.outputWeights = np.random.uniform(-0.1, 0.1, (self.numAttributes))
            self.num_hidden = self.numAttributes
        else:
            #make random weight matrices
            self.hiddenWeights = np.random.uniform(-0.1, 0.1, (self.num_hidden,self.numAttributes))
            self.outputWeights = np.random.uniform(-0.1, 0.1, (self.num_hidden))

        if self.max_iters >= 0:
            for _ in range(self.max_iters):
                #print self.outputWeights
                self.updateWeights(X,y,sample_weight)
        else:
            convergenceSlack = np.power(10.0, -11.0)
            while True:
                prevOutputWeights = np.copy(self.outputWeights)
                self.updateWeights(X,y,sample_weight)
                proportion = np.divide(prevOutputWeights, self.outputWeights)
                #print prevOutputWeights
                #print self.outputWeights
                #print proportion
                compare = map(lambda x: 1 - np.absolute(x) < convergenceSlack, proportion)
                if np.all(compare):
                    break

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
            for h in hiddenOutputs:
                outputSum = np.dot(h,self.outputWeights)
                #print outputSum
                outputs.append(ArtificialNeuralNetwork.sigmoid(outputSum))
            #print outputs            
            outputUpdates = self.getOutputUpdates(hiddenOutputs, outputs, y)
            #print outputUpdates
            hiddenUpdates = self.getHiddenUpdates(hiddenOutputs, outputs, outputUpdates, X)
            #print hiddenUpdates

            self.applyUpdates(outputUpdates, hiddenUpdates)
        else:
            outputs = list()
            for example in X:
                outputSum = np.dot(example, self.outputWeights)
                outputs.append(ArtificialNeuralNetwork.sigmoid(outputSum))

            outputUpdates = self.getOutputUpdates(X, outputs, y)
            self.applyUpdates(outputUpdates)

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

    def applyUpdates(self, outputUpdates, hiddenUpdates=None):
        if hiddenUpdates is not None:
            for i in range(self.num_hidden):
                self.outputWeights[i] -= self.learningRate*(outputUpdates[i])
            
            for i in range(self.numAttributes):
                for j in range(self.num_hidden):
                    self.hiddenWeights[j,i] -= self.learningRate*(hiddenUpdates[j,i] + self.gamma*self.hiddenWeights[j,i])
        else:
            for i in range(len(self.outputWeights)):
                #print outputUpdates[i]
                self.outputWeights[i] -= self.learningRate*(outputUpdates[i])
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
            if self.hiddenWeights is not None:
                hiddenLayerOutputs = self.getHiddenOutputs(curExample)
                toReturn.append(ArtificialNeuralNetwork.sigmoid(np.dot(hiddenLayerOutputs,self.outputWeights)))
            else:
                toReturn.append(ArtificialNeuralNetwork.sigmoid(np.dot(curExample,self.outputWeights)))

        return toReturn

    @staticmethod
    def sigmoid(wx):
        return 1 / (1 + np.exp(-wx))