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

        
    def fit(self, X, y, sample_weight=None):
        """ Fit a neural network of layer_sizes * num_hidden hidden units using X, y """
        self.numAttributes = len(X[0])
        self.layers = list()
        self.layers.append(self.buildLowestLayer())
        for _ in range(self.layer_sizes - 1):
            self.layers.append(self.buildLayer())
        self.outputPerceptron = Perceptron(self.num_hidden)

        for _ in range(self.max_iters):
            self.updateWeights(X,y,sample_weight)
        return

    def updateWeights(self,X,y,sample_weight=None):
        dLdWOutput = self.getdLdWOutput(X,y,sample_weight)
        self.updateHiddenLayer(X,y,dLdWOutput,sample_weight)
        self.updateOutputWeights(dLdWOutput)
        return

    def updateHiddenLayer(self,X,y,dLdWOutput,sample_weight=None):
        xVectorsByExample = self.getXVectorsByExample(X)
        for i in range(self.num_hidden):
            curPerceptron = self.layers[0][i]
            dLdW = list(0.0 for _ in range(curPerceptron.numInputs))
            for j in range(len(X)):
                xVector = xVectorsByExample[j]
                hn = xVector[i]
                downstreamDLDW = dLdWOutput[i]
                for attr in range(len(X[j])):
                    x = X[j][attr]
                    #hns cancel
                    dLdW[attr] += (1-hn)*x*downstreamDLDW*self.outputPerceptron.w[i]
            for wIndex in range(curPerceptron.numInputs):
                curPerceptron.w[wIndex] -= 0.01*(dLdW[wIndex] + curPerceptron.w[wIndex]*self.gamma) 
        return

    def getXVectorsByExample(self, X):
        toReturn = list()
        for i in range(len(X)):
            toReturn.append(self.calc_xVector(self.outputPerceptron,self.layer_sizes,X[i]))
        return toReturn

    def updateOutputWeights(self,dLdW):
        for i in range(self.outputPerceptron.numInputs):
            #minus or plus?
            self.outputPerceptron.w[i] -= 0.01*(dLdW[i] + self.outputPerceptron.w[i]*self.gamma) 

    def getdLdWOutput(self,X,y,sample_weight=None):
        dLdW = list(0.0 for _ in range(self.outputPerceptron.numInputs))
        for i in range(len(X)):
            curExample = X[i]
            curLabel = y[i]
            hn = self.predict_proba_example(curExample)
            x = self.calc_xVector(self.outputPerceptron,self.layer_sizes,curExample)
            for j in range(self.outputPerceptron.numInputs):
                dLdW[j] += (hn-curLabel)*hn*(1-hn)*x[j] 
        return dLdW


        
    def predict(self, X):
        """ Return the -1/1 predictions of the decision tree """
        #The list that will hold the predictions
        predictions = []

        for example in X:
            #Get the confidence prediction for the example
            # Note that confidence ranges from 0 to 1, and is the number of positive labels / number of examples at the leaf
            prob = self.predict_proba_example(example)
            #If the confidence is above or equal to 0.5, we will predict positive
            if prob >= 0.5:
                predictions.append(1)
            #Otherwise, predict negative
            else:
                predictions.append(-1)

        return predictions

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        predictions = []
        for example in X:
            prob = self.predict_proba_example(example)
            predictions.append(prob)
        return predictions


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
            if curLayer >= 0:
                prevLayer = curLayer - 1
                xVector.append(self.calc_output(self.layers[prevLayer][i], prevLayer, example))
            else:
                xVector.append(example[i])
        return xVector

    @staticmethod
    def sigmoid(wx):
        return 1 / (1 + np.exp(-wx))

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
