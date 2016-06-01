
# coding: utf-8

# # Genome Imputation Project

# In[ ]:

import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as pp
from sklearn import preprocessing
from scipy.stats import mode

def readTrainFile(filename):
    return pd.read_csv(filename, sep=' ')

def createCrossValSet(original, divider):
    d = int(135*divider)
    train = original.iloc[:,0:d].copy()
    test = original.iloc[:,d:].copy()
    actual = original.iloc[:,d:].copy()
    for idx in range(len(test)):
        if idx % 2 == 1:
            test.iloc[idx,:] = 'x'
    return train, test, actual
    
def readTestFile(filename):
    return pd.read_csv(filename, sep=' ')

def imputeMode(train, test):
    counts = {0:0, 1:0, 2:0}
    for individual in range(len(train)):
        if train[individual] == 0:
            counts[0] = counts[0] + 1
        elif train[individual] == 1:
            counts[1] = counts[1] + 1
        elif train[individual] == 2:
            counts[2] = counts[2] + 1
    mode = max(counts.items(), key=operator.itemgetter(1))[0]
    return mode

def calculateAccuracy(actual, predicted):
    print("Calculating Accuracy...")
    if actual.shape != predicted.shape:
        print("Mismatch matrix dimensions")
        pass
    sampleSize = actual.shape[0]*actual.shape[1]
    correct = 0
    for SNP in range(len(actual)):
        for individual in range (len(actual.iloc[SNP])):
            if actual.iloc[SNP, individual] == predicted.iloc[SNP, individual]:
                correct = correct + 1
    return correct/sampleSize

def baselineImputation(train, test):
    # Simple Mean Imputation: take the mode of each SNP across all IND_#
    # This assumes SNPs are independent
    print("Calculating Baseline...")
    for SNP in range(len(train)):
        test.iloc[SNP,:] = test.iloc[SNP,:].replace('x', imputeMode(train.iloc[SNP,:], test))
        
def calculateDistance(train, ind):
    result = []
    for t in range(train.shape[1]):
        diff = 0
        for snp in range(train.shape[0]):
            if ind.iloc[snp] == 'x':
                continue
            elif train.iloc[snp, t] != ind.iloc[snp]:
                diff = diff + 1
        result.append(diff)
    return np.array(result)
        
def fillMissing(train, closest, ind):
    for snp in range(ind.shape[0]):
        if ind.iloc[snp] == 'x':
            ind.iloc[snp] = int(mode(train.iloc[snp, closest])[0])
#             print(ind.iloc[snp])

def knnImputation(train, test, k):
    # for every individual, find k nearest neighbor and take the majority value of its neighbors
    print("Imputing...")
    for ind in range(test.shape[1]):
        dist = calculateDistance(train, test.iloc[:,ind])
        closest = dist.argsort()[-k:]
        fillMissing(train, closest, test.iloc[:,ind])
        
def plotAccuracy(res_b, res_k, sets):
    pp.plot(sets, res_b, 'k', label='Baseline')
    pp.plot(sets, res_k[0], 'b', label='k=1')
    pp.plot(sets, res_k[1], 'g', label='k=3')
    pp.plot(sets, res_k[2], 'c', label='k=5')
    pp.plot(sets, res_k[3], 'm', label='k=7')
    pp.plot(sets, res_k[4], 'y', label='k=9')
    pp.plot(sets, res_k[5], 'r', label='k=11')
    pp.title("Baseline vs Local Imputation")
    pp.xlabel("% of Original Set as Train Set")
    pp.ylabel("Accuracy")
    pp.axis([0,1,0,1])
    pp.legend()
    pp.savefig('results.png')
    
if __name__ == "__main__":
#     trainSNPs = readTrainFile('imputation_training.txt')
#     testSNPs = readTestFile('imputation_test.txt')
    original = pd.read_csv('imputation_training.txt', sep=' ')
    sets = [.25, .5, .75]
    neighbors = [1, 3, 5, 7, 9, 11]
    baseline_results = []
    knn_results = [[],[],[],[],[],[]]
    f = open('results.txt', 'a')
    for trainCount in sets:
        train, test_b, actual = createCrossValSet(original, trainCount)
        for idx, k in enumerate(neighbors):
            test_k = test_b.copy()
            knnImputation(train, test_k, k)
            res_k = calculateAccuracy(actual, test_k)
            knn_results[idx].append(res_k)
            f.write(str(trainCount) + ' SNPs KNN Accuracy: ' + str(k) + ': ' + str(res_k) + '\n')
        baselineImputation(train, test_b)
        res_b = calculateAccuracy(actual, test_b)
        baseline_results.append(res_b)
        f.write(str(trainCount)+' SNPs Baseline Accuracy: '+str(res_b) + '\n')
        
    plotAccuracy(baseline_results, knn_results, sets)


# In[ ]:




# In[ ]:




# In[ ]:

# 0.25 SNPs Baseline Accuracy: 0.8062029411764706
# 0.5 SNPs Baseline Accuracy: 0.808364705882353
# 0.75 SNPs Baseline Accuracy: 0.8076941176470588

