"""
Statistics Computations
"""
import numpy as np
import scipy


class StatisticsManager(object):

    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []
        self.prediction_scores = []
        self.training_times = []
        self.statistics = {
            'accuracy' : (accuracy,  self.predicted_labels),
            'precision': (precision, self.predicted_labels),
            'recall'   : (recall,    self.predicted_labels),
            'auc'      : (auc,       self.prediction_scores),
        }

    def add_fold(self, true_labels, predicted_labels,
                 prediction_scores, training_time):
        """
        Add a fold of labels and predictions for later statistics computations

        @param true_labels : the actual labels
        @param predicted_labels : the predicted binary labels
        @param prediction_scores : the real-valued confidence values
        @param training_time : how long it took to train on the fold
        """
        self.true_labels.append(true_labels)
        self.predicted_labels.append(predicted_labels)
        self.prediction_scores.append(prediction_scores)
        self.training_times.append(training_time)

    def get_statistic(self, statistic_name, pooled=True):
        """
        Get a statistic by name, either pooled across folds or not

        @param statistic_name : one of {accuracy, precision, recall, auc}
        @param pooled=True : whether or not to "pool" predictions across folds
        @return statistic if pooled, or (avg, std) of statistic across folds
        """
        if statistic_name not in self.statistics:
            raise ValueError('"%s" not implemented' % statistic_name)

        statistic, predictions = self.statistics[statistic_name]

        if pooled:
            predictions = np.hstack(map(np.asarray, predictions))
            labels = np.hstack(map(np.asarray, self.true_labels))
            return statistic(labels, predictions)
        else:
            stats = []
            for l, p in zip(self.true_labels, predictions):
                stats.append(statistic(l, p))
            return np.average(stats), np.std(stats)

def accuracy(labels, predictions):
    numTotal = len(labels)
    numCorrect = 0
    for i in range(numTotal):
        if(labels[i] == predictions[i]):
            numCorrect += 1
    return float(numCorrect)/numTotal

def precision(labels, predictions):
    tp = 0
    fp = 0
    for i in range(len(labels)):
        if predictions[i] == 1:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
    if tp == 0 and fp == 0:
        return 0.0
    else:
        return float(tp)/(tp+fp)

def recall(labels, predictions):
    tp = 0
    fn = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            if predictions[i] == 1:
                tp += 1
            else:
                fn += 1
    if tp == 0 and fn == 0:
        return 1.0
    else:
        return float(tp)/(tp+fn)

def auc(labels, predictions):
    
    pairs = zip(labels, predictions)
    #print pairs
    pairs.sort(key=lambda x: x[1])
    prevVal = pairs[0][1]
    rocVals = set()
    rocVals.add((0.0, 0.0))
    rocVals.add((1.0, 1.0))

    numTotal = len(labels)
    #Intially, all example are >= the split (first split candidate is -inf), so the number of positve example GOE
    # is just the number of all positive examples
    numTP = len(filter(lambda x: x == 1, labels))
    #Similarly, no exaples are < -inf, so there are no LT positive examples
    numFN = 0
    numFP = numTotal - numTP
    prevPrediction = None
    i = 0
    for i in range(numTotal):
        curLabel = pairs[i][0]
        curPrediction = pairs[i][1]
        if curLabel == 1:
            numTP -= 1
            numFN += 1
        if curLabel == 0:
            numFP -= 1
        
        if prevPrediction != curPrediction:
            rocVals.add( ((float(numFP)/(numTP + numFP)), (float(numTP)/(numTP+numFN))) )
        prevPrediction = curPrediction

    rocVals = list(rocVals)
    rocVals.sort(key=lambda x: x[0])
    #print rocVals
    area = 0.0
    for i in range(len(rocVals)-1):
        (prevfp, prevtp) = rocVals[i]
        (nextfp, nexttp) = rocVals[i+1]
        h = nextfp - prevfp
        avg = (prevtp + nexttp)/2
        area += avg*h

    return area
