kNN
===

k-Nearest-Neighbors Text Classifier
Jared Kramer

This script takes training and test files (formatted as in the accompanying samples), a k-value that specifies the desired number of neighbors, a similarity value that specifies the similarity measure (1 for Euclidean, 2 for Cosine), and the name of an output file. The confusion matrix is printed to stdout.

The classifier works by calculating the distances between a given test instance and every training instance using the specified similarity metric.  Note that Euclidean distance here is actually a dissimilarity measure, where as Cosine is a similarity measure. The labels of k nearest training neighbors each cast a vote and the test instance is assigned the label with the most votes.

As noted above, the training and testing data must be formatted as in the accompanying examples, though changing the code to fit other data formats would be trivial in most cases.

This code classifies 900 test instances in approximately 2 minutes with the below accuracies. The running time is all testing and the training step for kNN is non-existant.

k	  euclidean cosine
1	  0.636		  0.72
5	  0.646		  0.703
10	0.66		  0.663


Usage: The command line arguments are as follows: 
1 = training data
2 = testing data
3 = k-value (number of neighbors)
4 = similarity metric
5 = output file to write detailed system output 
