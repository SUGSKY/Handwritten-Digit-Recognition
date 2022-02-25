import random
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce
import math

g_dataset = {}
g_test_good = {}
g_test_bad = {}
NUM_ROWS = 32
NUM_COLS = 32
DATA_TRAINING = 'digit-training.txt'    
DATA_TESTING = 'digit-testing.txt'
DATA_PREDICT = 'digit-predict.txt'

# kNN parameter
KNN_NEIGHBOR = 7

##########################
##### Load Data  #########
##########################

# Convert next digit from input file as a vector 
# Return (digit, vector) or (-1, '') on end of file
def read_digit(p_fp):
    # read entire digit (inlude linefeeds)
    bits = p_fp.read(NUM_ROWS * (NUM_COLS + 1))
    if bits == '':
        return -1,bits
    # convert bit string as digit vector
    vec = [int(b) for b in bits if b != '\n']
    val = int(p_fp.readline())
    return val,vec

# Parse all digits from training file
# and store all digits (as vectors) 
# in dictionary g_dataset
def load_data(p_filename=DATA_TRAINING):
    global g_dataset
    # Initial each key as empty list 
    g_dataset = defaultdict(list)
    with open(p_filename) as f:
        while True:
            val,vec = read_digit(f)
            if val == -1:
                break
            g_dataset[val].append(vec)

##########################
##### kNN Models #########
##########################

# Given a digit vector, returns
# the k nearest neighbor by vector distance
def knn(p_v, size=KNN_NEIGHBOR):
    nn = []
    for d,vectors in g_dataset.items():
        for v in vectors:
            dist = round(distance(p_v,v),2)
            nn.append((dist,d))
    nn.sort()
    return nn[:size]
    
# Based on the knn Model (nearest neighhor),
# return the target value
def knn_by_most_common(p_v):
    nn = knn(p_v)
    return nn[0][1]

##########################
##### Prediction  ########
##########################

# Make prediction based on kNN model
# Parse each digit from the predict file
# and print the predicted balue
def predict(p_filename=DATA_PREDICT):
    print('-' * 40)
    print('Results of predictions')
    print('-' * 40)
    with open(p_filename) as f:
        while True:
            val,vec = read_digit(f)
            if val == -1:
                break
            predict = knn_by_most_common(vec)
            print('predict value: ',predict)

##########################
##### Accuracy   #########
##########################

# Compile an accuracy report by
# comparing the data set with every
# digit from the testing file 
def validate(p_filename=DATA_TESTING):
    global g_test_bad, g_test_good
    g_test_bad = defaultdict(int)
    g_test_good = defaultdict(int)
    with open(p_filename) as f:
        while True:
            actual,vec = read_digit(f)
            if actual == -1:
                break
            predict = knn_by_most_common(vec)
            if actual == predict:
                g_test_good[actual] += 1
            else:
                g_test_bad[actual] += 1
                
##########################
##### Data Models ########
##########################

# Randomly select X samples for each digit
def data_by_random(size=25):
    for digit in g_dataset.keys():
        g_dataset[digit] = random.sample(g_dataset[digit],size)

##########################
##### Vector     #########
########################## 

# Return distance between vectors v & w
 
#Hamming Distance
def distance(v, w):
    return sum(l1 != l2 for l1, l2 in zip(v, w))

#Euclidean Distance
# def distance(v, w):
#     d = [(v_i - w_i)**2 for v_i,w_i in zip(v,w)]
#     return sum(d)**0.5

##########################
##### Report     #########
##########################

# Show info for training data set
def show_info():  
    print('-' * 40)
    title = 'Training Info'
    t = title.center(40)
    print(t)
    print('-' * 40)
    totalSample = []
    for d in range(10):
        linenew1 = '{:>17} = {:<15}'.format(d, len(g_dataset[d]))
        print(linenew1)
        totalSample.append(len(g_dataset[d]))
    print('-' * 40)
    sampleSum = sum(totalSample)
    print('Total Sample =',sampleSum)
    print('-' * 40)
    
# Show test results
def show_test():
    right = []
    wrong = []
    print('-' * 40)
    title = 'Testing Info'
    t = title.center(40)
    print(t)
    print('-' * 40)
    for d in range(10):
        good = g_test_good[d]
        bad = g_test_bad[d]
        accuracy = (good/(good + bad)) * 100
        right.append(good)
        wrong.append(bad)
        linenew = '{:>8} = {:>5}, {:>3}, {:>5}%'.format(d, good, bad, round(accuracy,3))
        print(linenew)
    rightSum = sum(right)
    wrongSum = sum(wrong)
    totalAccuracy = rightSum + wrongSum
    accuracySum = round((rightSum/totalAccuracy)*100, 3)
    print('-' * 40)
    print('Accuracy =', accuracySum, '%')
    print('Correct/Total = {}/{}'.format(rightSum, (totalAccuracy)))

if __name__ == '__main__':
    start = datetime.now()
    print('Beginning of Validation @ ', start)  
    load_data()
    show_info()
    validate()
    show_test()
    stop = datetime.now()
    print('-' * 40)
    print('End of Training @ ', stop) 
    predict()
    
    print('-' * 40)
