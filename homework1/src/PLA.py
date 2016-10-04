import numpy as np
from matplotlib import pyplot as plt


def loadData(filename):
    X = []
    Y = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            row = [1.0] + [float(x) for x in line[:-1]]     # bias b = x_0 * w_0
            X.append(row)
            Y.append(int(line[-1]))

    return np.array(X), np.array(Y)


def sign(x):
    if x <= 0:
        return -1
    else:
        return 1


def shuffleData(X, Y):
    tmp = np.concatenate((X, Y.reshape((-1, 1))), axis=1)
    np.random.shuffle(tmp)

    return tmp[:, :-1], tmp[:, -1]


def train(trainX, trainY, learnRate=1):
    weight = np.zeros(trainX.shape[1])      # !!! rember to initialized as 0

    length = len(trainY)
    row = 0
    countUpdate = 0
    lastUpdate = 0
    flag = False

    while True:
        if row == length:
            if flag:
                break

            row = 0
            flag = True

        predictedY = sign(trainX[row].dot(weight))
        if predictedY != trainY[row]:
            flag = False

            weight += learnRate * trainY[row] * trainX[row]
            countUpdate += 1
            lastUpdate = row

        row += 1

    return countUpdate, lastUpdate


def randomCycle(round, X, Y, learnRate= 1):
    sum = 0.0
    histogram = []
    for i in range(1, round+1):
        trainX, trainY = shuffleData(X, Y)
        countUpdate, lastUpdate = train(trainX, trainY, learnRate)

        print "round " + str(i) + ", updated " + str(countUpdate) + " times."
        sum += countUpdate
        histogram.append(sum/i)

    return histogram


if __name__ == '__main__':
    trainX, trainY = loadData('../data/hw1_15_train.dat')

    # question 15
    # countUpdate, lastUpdate = train(trainX, trainY)
    # print countUpdate
    # print lastUpdate

    # question 16
    # histogram = randomCycle(2000, trainX, trainY)

    # question 17
    histogram = randomCycle(2000, trainX, trainY, learnRate= 0.2)

    print histogram[-1]
    plt.plot(range(1, 2001), histogram)
    plt.show()
