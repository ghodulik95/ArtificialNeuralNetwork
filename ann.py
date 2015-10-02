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
        pass
		self.gamma = gamma
		self.layer_sizes = layer_sizes
		self.num_hidden = num_hidden
		self.epsilon = epsilons
		self.max_iters = max_iters

		
		
		

    def fit(self, X, y, sample_weight=None):
        """ Fit a neural network of layer_sizes * num_hidden hidden units using X, y """
        self.numAttributes = len(X[0])
        self.layers = list()
        self.layers.append(self.buildLowestLayer())
        for i in range(self.layer_sizes - 1):
        	self.layers.append(self.buildLayer())
        self.outputPerceptron = Perceptron(self.num_hidden)
        pass

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
    	self.calc_output(self.outputPerceptron, self.layer_sizes, example)

    def calc_output(self, curPerceptron, curLayer, example):
    	prevLayer = curLayer - 1
    	sumOfWXs = 0.0
    	for i in range(curPeceptron.numInputs):
    		x = None
    		if curLayer >= 0:
    			x = self.calc_output(self.layers[prevLayer][i], prevLayer, example)
    		else:
    			x = example[i]
    		sumOfWXs += x*curPerceptrion.w[i]
    	return ArtificialNeuralNetwork.sigmoid(sumOfWXs)

    @staticmethod
    def sigmoid(wx):
    	return 1 / (1 + np.exp(-wx))

    def buildLayer(self):
    	newLayer = list()
    	for i in range(self.num_hidden):
    		newLayer.append(Perceptron(self.num_hidden)
    	return newLayer

	def buildLowestLayer(self):
		self.lowestLayer = list()
		for i in range(self.num_hidden):
			self.lowestLayer.append(Perceptron(self.numAttributes, True))
		return self.lowestLayer
		
class Perceptron(object)

	def __init__(self, numInputs, isLowest = False):
		self.isLowest = isLowest
		self.numInputs = numInputs
		self.w = Perceptron.generateRandomW(numInputs)
		
	@staticmethod
	def generateRandomW(numInputs):
		w = list()
		for i in range(numInputs):
			randf = np.rand
			randf = randf / 5
			randf = randf - 0.1
			w.append(randf)
