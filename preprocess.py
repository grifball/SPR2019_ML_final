
file1 = open('./ember/train_features_5.jsonl', 'r')
file2 = open('./ember/train_features_0.jsonl', 'r')
trainFile = open('./training.jsonl', 'w')
testFile = open('./testing.jsonl', 'w')
for line in range(0,1000):
    trainFile.write(file1.readline())
    testFile.write(file2.readline())
