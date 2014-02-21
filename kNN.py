#!/opt/python-2.7/bin/ python2.7
# -*- coding: utf-8 -*-
"""
kNN Text Classifier
Jared Kramer

This script takes training and test files (formatted as fond in the accompanying samples),
a k-val that specifies the desired number of neighbors, a similarity value that specifies
the similarity measure (1 for Euclidean, 2 for Cosine), and the name of an output file.
The confusion matrix is printed to stdout.

"""

import sys
from math import sqrt
from collections import defaultdict
from collections import Counter as counter
from operator import itemgetter

if len(sys.argv) > 1:
    # Command line arguments
    train, test, k_val, similarity, sys_output = sys.argv[1:]
else:
   # Debugging arguments
   train = "examples/train.vectors.txt"
   test = "examples/test.vectors.txt"
   k_val = 10
   similarity = 2
   sys_output = "sys_output"

#need to cast these for use with the shell script
k_val = int(k_val)
similarity = int(similarity)

if similarity not in {1, 2}:
   sys.exit("Similarity identifier not 1 or 2; please make sure the command is correct and try again")

#load the data and initialize data structures
train = open(train, 'r').read().strip().split('\n')
test = open(test, 'r').read().strip().split('\n')
sys_output = open(sys_output, 'w')
instance_number = 0
test_confusion_matrix = defaultdict(lambda: defaultdict(int))
labels = set()
lefts = defaultdict(float)
rights = defaultdict(float)

# Load the data
train = {i: (train[i].split()[0], {word.split(":")[0]: int(word.split(":")[1]) for word in train[i].split()[1:]}) for i in range(len(train))}
test = {i: (test[i].split()[0], {word.split(":")[0]: int(word.split(":")[1]) for word in test[i].split()[1:]}) for i in range(len(test))}

every_word = {word for i in train for word in train[i][1].keys()}
every_word = every_word.union({word for i in test for word in test[i][1].keys()})

# Pre-calculate squares for euclidian distance
counts = set([value for item in train.values() for value in set(item[1].values())]).union(set([value for item in test.values() for value in set(item[1].values())]))
squares = {i: i**2 for i in counts}
train_squares = {i: {word: squares[count] for word, count in train[i][1].items()} for i in train}
test_squares = {i: {word: squares[count] for word, count in test[i][1].items()} for i in test}

# Pre-calculate cosine sums
if similarity == 2:
    for doc in train:
        lefts[doc] = sqrt(sum(train_squares[doc].values()))
    for doc in test:
        rights[doc] = sqrt(sum(test_squares[doc].values()))

# calculate the euclidean distances, there are a couple optimizations here which I left in for posterity
def euclidian(train_counts, test_counts, train_features, test_features, shared, train_id, test_id):
    train_only = train_features - shared
    test_only = test_features - shared

    to_subtract = 0
    running_sum = 0

    # (X - Y)^2 = X^2 + Y^2 - 2xy

    #### Original 65% 2:00
    for word in shared:
        running_sum += (test_counts[word] - train_counts[word])**2
    for word in test_only:
        running_sum += test_squares[test_id][word]
    for word in train_only:
        running_sum += train_squares[train_id][word]
    return running_sum # Ignore square root

    # Optimized 1 65% 2:03
#    for word in shared:
#        to_subtract += test_counts[word] * train_counts[word]
#        running_sum += test_squares[test_id][word] + train_squares[train_id][word]
#    for word in train_only:
#        running_sum += train_squares[train_id][word]
#    for word in test_only:
#        running_sum += test_squares[test_id][word]
#    return running_sum - 2*to_subtract # Ignore square root

    # Optimized 2 69% 1:59
#    for word in shared:
#        to_subtract += test_counts[word] * train_counts[word]
#        running_sum += train_squares[train_id][word]
#    for word in train_only:
#        running_sum += train_squares[train_id][word]
#    for word in test_only:
#        running_sum += test_squares[test_id][word]
#    return running_sum - 2*to_subtract # Ignore square root

# calculate the cosine distance
def cosine(train_counts, test_counts, shared, train_id, test_id):
    top = float(sum([train_counts[w] * test_counts[w] for w in shared]))
    return top/(lefts[train_id]*rights[test_id])

# this method prints out the confusion matrix
def print_matrix(matrix):
    print "".join(["class_num=", str(len(labels)), " feat_num=", str(len(every_word))])
    print "".join(["Confusion matrix for the testing data:"])
    print "row is the truth, column is the system output\n"
    print "\t", " ".join([label for label in matrix])
    total_docs_classified = 0
    docs_classified_correctly = 0
    for label1 in labels:
        print label1,
        for label2 in labels:
            result = matrix[label1][label2]
            print result,
            if label1 == label2:
                docs_classified_correctly += result
            total_docs_classified += result
        print
    print "".join(["\n", "testing accuracy = ", str(float(docs_classified_correctly)/total_docs_classified), '\n\n'])

# Run KNN
for test_id in test: #get the info form the test instance
    test_instance = test[test_id]
    test_label = test_instance[0]
    labels.add(test_label)
    test_counts = test_instance[1]
    test_features = set(test_counts.keys())
    nearest = []
    for train_id in train: #check it against the info for every training instance
        train_instance = train[train_id]
        train_label = train_instance[0]
        labels.add(train_label)
        train_counts = train_instance[1]
        train_features = set(train_counts.keys())
        shared = test_features.intersection(train_features)
        if similarity == 1: # Euclidian distance measure
            nearest.append((train_label, euclidian(train_counts, test_counts, train_features, test_features, shared, train_id, test_id)))
        else: # Cosine distance measure
            nearest.append((train_label, cosine(train_counts, test_counts, shared, train_id, test_id)))
    if similarity == 1: # If Euclidian
        nearest = sorted(nearest, key=itemgetter(1))[:int(k_val)]
    else: # If cosine
        nearest = sorted(nearest, key=itemgetter(1), reverse=True)[:int(k_val)]
    projected_label = counter(item[0] for item in nearest).most_common(1)[0][0]
    test_confusion_matrix[test_label][projected_label] += 1
    sys_output.write("".join(["test:", str(instance_number), "\t", test_label, "\t"]))
    instance_number += 1
    # print the labels and votes to the sys_output
    votes = defaultdict(int)
    for tup in nearest:
        votes[tup[0]] += 1
    for c in labels:
        sys_output.write("\t"+c+"\t"+str(votes[c]))
    sys_output.write("\n")

#print the confusion matrix
print_matrix(test_confusion_matrix)

