import numpy as np
import random
from matplotlib import pyplot as plt


def loadData(filename):
    X = []
    Y = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            row = [1.0] + [float(x) for x in line[:-1]]
            X.append(row)
            Y.append(int(line[-1]))

    return np.array(X), np.array(Y)


def sign(x):
    if x <= 0:
        return -1
    else:
        return 1


def train(weight, trainX, trainY, pocketSize):
    countUpdate = 0
    while countUpdate < pocketSize:
        randomIndex = random.randint(0, len(trainY)-1)
        predictedY = sign(trainX[randomIndex].dot(weight))

        if predictedY != trainY[randomIndex]:
            weight += trainY[randomIndex] * trainX[randomIndex]
            countUpdate += 1

    return weight


def predict(weight, testX, testY):
    length = len(testY)
    weight = weight.reshape((-1, 1))
    testY = testY.reshape((-1,1))

    # vectorized the sign function to apply on numpy array
    sign_ufunc = np.frompyfunc(sign, 1, 1)
    predictedY = sign_ufunc(testX.dot(weight))

    # return the error rate
    return float(sum(abs(testY - predictedY))) / 2 / length


def run(trainX, trainY, testX, testY, pocket=True, pocketSize=50):
    histogram = []

    sum = 0.0
    bestErrorRate = 2.0
    bestWeight = np.zeros(trainX.shape[1])
    for i in range(2000):
        weight = train(bestWeight, trainX, trainY, pocketSize)
        errorRate = predict(weight, testX, testY)

        if pocket:
            if errorRate < bestErrorRate:
                sum += errorRate
                bestErrorRate = errorRate
                bestWeight = weight
            else:
                sum += bestErrorRate
        else:
            sum += errorRate

        histogram.append(sum / (i+1))

    return histogram


if __name__ == '__main__':
    trainX, trainY = loadData('../data/hw1_18_train.dat')
    testX, testY = loadData('../data/hw1_18_test.dat')

    # question 18
    # histogram = run(trainX, trainY, testX, testY)

    # question 19
    # histogram = run(trainX, trainY, testX, testY, pocket=False)

    # question 20
    histogram = run(trainX, trainY, testX, testY, pocketSize=100)

    print histogram[-1]
    plt.plot(range(1, 2001), histogram)
    plt.show()