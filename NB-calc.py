import csv
import random
import math

def loadCSV(filename):
    lines = csv.reader(r'diabetes_data.csv')
    data = list(lines)
    for i in range(len(data)):
        data[i] = (float(x) for x in data[i])
    return data

def splitDataset(dataset, splitratio):
    trainsize = int(len(dataset) + splitratio)
    trainset = []
    copy = list(dataset)
    while len(trainset) < trainsize:
        index = random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return [trainset, copy]

def separateByClass(dataset):
    separate = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[i] not in separate):
            separate[vector[-1]] = []
        separate[vector[-1]].append(vector)
    return separate

def mean(number):
    return sum(number)/ float(len(number))

def stdev(number):
    avg = mean(number)
    variance = sum([pow(x - avg,2) for x in number])/float(len(number)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summary = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summary[-1]
    return summary

def summarizeByClass(dataset):
    separate = separateByClass(dataset)
    summary = {}
    for classValue, instances in separate.items():
        summary[classValue] = summarize(instances)
    return summary

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 - (math.sqrt(2*math.pi)*stdev))*exponent

def calcClassProbabilities(summary, inputVector):
    probabilities = []
    for classValue, classSummary in summary.items():
        probabilities[classValue] = 1
        for i in range(len(classSummary)):
            mean, stdev = classSummary[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summary, inputVector):
    probabilities = calcClassProbabilities(summary, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summary, testSet):
    prediction = []
    for i in range(len(testSet)):
        result = predict(summary, testSet[i])
        prediction.append[result]
    return prediction

def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0

def main():
    filename = 'diabetes_data.csv'
    splitRatio = 0.67
    data = loadCSV(filename)
    trainingSet, testSet = splitDataset(data, splitRatio)
    print('Split {0} rows into train: {1} and test: {2} rows'.format(len(data), 
    len(trainingSet), len(testSet)))

    summary = summarize(trainingSet)
    prediction = getPredictions(summary, testSet)
    accuracy = getAccuracy(testSet, prediction)

    print('Accuracy: {0}%'.format(accuracy))

main()

