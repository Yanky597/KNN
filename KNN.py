import numpy as np


class KNN:
    amountOfNeighbors = 0
    train_data = []
    train_target = []
    correlationValueByColumn = []

    def __init__(self, neighbors):
        self.amountOfNeighbors = neighbors

    def fit(self, data, target):
        self.train_data = data
        self.train_target = target

        # calculate the correlation of the features to the target and save the values
        for i in range(len(self.train_data[0])):
            self.correlationValueByColumn.append(np.corrcoef(self.train_data[:, i], self.train_target).mean())

    def predict(self, dataPoint):
        storedIndexes = {}
        classifications = {}
        valueOfDP = self.sumWeightedValues(dataPoint)

        # store the value returned by the getDifference method in a dictionary
        for i in range(len(self.train_data)):
            storedIndexes[self.getDifference(valueOfDP, self.train_data[i, :])] = i

        # sort the keys
        sortedKeys = sorted(storedIndexes)

        # return the classification with the most similar neighbors in the range of amountOfNeighbors
        for i in range(self.amountOfNeighbors):
            currentClassification = self.train_target[storedIndexes[sortedKeys[i]]]

            # add or increment values in dictionary
            if currentClassification not in classifications:
                classifications[currentClassification] = 1
            else:
                classifications[currentClassification] += 1

        # returns the classification that occurred most often
        return max(classifications, key=classifications.get)


    def sumWeightedValues(self, listOfValues):
        total = 0
        for val, cf in zip(listOfValues, self.correlationValueByColumn):
            total += (val * cf)
        return total

    def getDifference(self, dataPointValue,  currentRow):
        return abs(dataPointValue - self.sumWeightedValues(currentRow))

    def score(self, y_true, y_predict):
        counter = 0
        for X, y in zip(y_true, y_predict):
            if self.predict(X) == y:
                counter += 1

        return counter / len(y_true)
