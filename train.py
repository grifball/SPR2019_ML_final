import sys
import json
import numpy as np

# get all the training data
trainFile = open('./training.jsonl', 'r')
trainDatas = []
for line in trainFile:
    line = trainFile.readline()
    trainData = json.loads(line)
    trainDatas.append(trainData)
trainDatas = np.array(trainDatas)

# get all the test data
testFile = open('./testing.jsonl', 'r')
testDatas = []
for line in testFile:
    line = testFile.readline()
    testData = json.loads(line)
    testDatas.append(testData)
testDatas = np.array(trainDatas)

# "classify" each point in the test data (just check if their first import is the same)
total = 0
correct = 0
for testData in testDatas:
    # check if the test point has a label
    if testData['label'] != -1:
        total += 1
        # check if the test point has imports
        if len(testData['imports'].keys()) != 0:
            # iterate through the training data and check the dll imports
            for trainData in trainDatas:
                # check if the training point has a label and imports
                if trainData['label'] != -1 and len(trainData['imports'].keys()) != 0:
                    if sorted(trainData['imports'].keys())[0] == sorted(testData['imports'].keys())[0]:
                        # we're guessing the label of this training point because the test point had the same import
                        guess = trainData['label']
                        break
            if guess == testData['label']:
                correct += 1
accuracy = correct/total

print("correct",correct)
print("total",total)
print("accuracy",accuracy)
