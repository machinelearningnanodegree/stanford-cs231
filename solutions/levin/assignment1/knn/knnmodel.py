import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n import data_utils
from assignment1.cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from sklearn.cross_validation import KFold


class KNNModel(object):
    def __init__(self):
       
        return
    def load_data(self):
        # Load the raw CIFAR-10 data.
        cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
        self.X_train, self.y_train, self.X_test, self.y_test = load_CIFAR10(cifar10_dir)
        
        # As a sanity check, we print out the size of the training and test data.
        print 'Training data shape: ', self.X_train.shape
        print 'Training labels shape: ', self.y_train.shape
        print 'Test data shape: ', self.X_test.shape
        print 'Test labels shape: ', self.y_test.shape
        return
    def visualize_data(self):
        # Visualize some examples from the dataset.
        # We show a few examples of training images from each class.
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7
        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(self.y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(self.X_train[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()
        return
    def subsample_reshape(self):
        # Subsample the data for more efficient code execution in this exercise
        self.num_training = 5000
        mask = range(self.num_training)
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]
        
        self.num_test = 500
        mask = range(self.num_test)
        self.X_test = self.X_test[mask]
        self.y_test = self.y_test[mask]
        
        # Reshape the image data into rows
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], -1))
        print self.X_train.shape, self.X_test.shape
        return
    def train(self):
        # Create a kNN classifier instance. 
        # Remember that training a kNN classifier is a noop: 
        # the Classifier simply remembers the data and does no further processing 
        self.classifier = KNearestNeighbor()
        self.classifier.train(self.X_train, self.y_train)
        return
    def predict(self):
        # Open cs231n/classifiers/k_nearest_neighbor.py and implement
        # compute_distances_two_loops.
        
        # Test your implementation:
        dists = self.classifier.compute_distances_two_loops(self.X_test)
        print dists.shape
        # We can visualize the distance matrix: each row is a single test example and
        # its distances to training examples
#         plt.imshow(dists, interpolation='none')
#         plt.show()
        
        # Now implement the function predict_labels and run the code below:
        # We use k = 1 (which is Nearest Neighbor).
        y_test_pred = self.classifier.predict_labels(dists, k=1)
        
        # Compute and print the fraction of correctly predicted examples
        num_correct = np.sum(y_test_pred == self.y_test)
        accuracy = float(num_correct) / self.num_test
        print 'Got %d / %d correct => accuracy: %f' % (num_correct, self.num_test, accuracy)
        
        # try k = 5
        y_test_pred = self.classifier.predict_labels(dists, k=5)
        num_correct = np.sum(y_test_pred == self.y_test)
        accuracy = float(num_correct) / self.num_test
        print 'Got %d / %d correct => accuracy: %f' % (num_correct, self.num_test, accuracy)
        self.dists = dists
        return
    def compute_distance_oneloop(self):
        # Now lets speed up distance matrix computation by using partial vectorization
        # with one loop. Implement the function compute_distances_one_loop and run the
        # code below:
        dists_one = self.classifier.compute_distances_one_loop(self.X_test)
        
        # To ensure that our vectorized implementation is correct, we make sure that it
        # agrees with the naive implementation. There are many ways to decide whether
        # two matrices are similar; one of the simplest is the Frobenius norm. In case
        # you haven't seen it before, the Frobenius norm of two matrices is the square
        # root of the squared sum of differences of all elements; in other words, reshape
        # the matrices into vectors and compute the Euclidean distance between them.
        difference = np.linalg.norm(self.dists - dists_one, ord='fro')
        print 'Difference was: %f' % (difference, )
        if difference < 0.001:
            print 'Good! The distance matrices are the same'
        else:
            print 'Uh-oh! The distance matrices are different'
        return
    def compute_distance_noloop(self):
        # Now lets speed up distance matrix computation by using partial vectorization
        # with one loop. Implement the function compute_distances_one_loop and run the
        # code below:
        dists_two = self.classifier.compute_distances_no_loops(self.X_test)
        
        # To ensure that our vectorized implementation is correct, we make sure that it
        # agrees with the naive implementation. There are many ways to decide whether
        # two matrices are similar; one of the simplest is the Frobenius norm. In case
        # you haven't seen it before, the Frobenius norm of two matrices is the square
        # root of the squared sum of differences of all elements; in other words, reshape
        # the matrices into vectors and compute the Euclidean distance between them.
        difference = np.linalg.norm(self.dists - dists_two, ord='fro')
        print 'Difference was: %f' % (difference, )
        if difference < 0.001:
            print 'Good! The distance matrices are the same'
        else:
            print 'Uh-oh! The distance matrices are different'
        return
    def time_function(self, f, *args):
        """
        Call a function f with args and return the time (in seconds) that it took to execute.
        """
        import time
        tic = time.time()
        f(*args)
        toc = time.time()
        return toc - tic
    def compare_vectorization_speed(self):
        # you should see significantly faster performance with the fully vectorized implementation
        two_loop_time = self.time_function(self.classifier.compute_distances_two_loops, self.X_test)
        print 'Two loop version took %f seconds' % two_loop_time
        
        one_loop_time = self.time_function(self.classifier.compute_distances_one_loop, self.X_test)
        print 'One loop version took %f seconds' % one_loop_time
        
        no_loop_time = self.time_function(self.classifier.compute_distances_no_loops, self.X_test)
        print 'No loop version took %f seconds' % no_loop_time
        return
    def do_cross_validation(self):
        num_folds = 5
        k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
        
        X_train_folds = []
        y_train_folds = []
        X_val_folds = []
        y_val_folds = []
        kf = KFold(self.y_train.shape[0], n_folds=num_folds)
        for train_index, val_index in kf:
            X_train, X_val = self.X_train[train_index], self.X_train[val_index]
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]
            X_train_folds.append(X_train)
            y_train_folds.append(y_train)
            X_val_folds.append(X_val)
            y_val_folds.append(y_val)
        k_to_accuracies = {}
        self.classifier = KNearestNeighbor()
        for k in k_choices:
            for n_fold in range(num_folds):
                self.classifier.train(X_train_folds[n_fold], y_train_folds[n_fold])
                dists = self.classifier.compute_distances_no_loops(X_val_folds[n_fold])
                y_val_pred = self.classifier.predict_labels(dists, k=k)
                num_correct = np.sum(y_val_pred == y_val_folds[n_fold])
                accuracy = float(num_correct) / y_val_folds[n_fold].shape[0]
                if not k in k_to_accuracies:
                    k_to_accuracies[k] = []
                k_to_accuracies[k].append(accuracy)
                print "k = {}".format(k)
                print 'Got %d / %d correct => accuracy: %f' % (num_correct, y_val_folds[n_fold].shape[0], accuracy)
               
        
        # Print out the computed accuracies
        for k in sorted(k_to_accuracies):
            for accuracy in k_to_accuracies[k]:
                print 'k = %d, accuracy = %f' % (k, accuracy)
        self.plot_observation(k_choices, k_to_accuracies)
        return
    def plot_observation(self, k_choices, k_to_accuracies):
        # plot the raw observations
        for k in k_choices:
            accuracies = k_to_accuracies[k]
            plt.scatter([k] * len(accuracies), accuracies)
        
        # plot the trend line with error bars that correspond to standard deviation
        accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
        accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
        plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
        plt.title('Cross-validation on k')
        plt.xlabel('k')
        plt.ylabel('Cross-validation accuracy')
        max_index = np.argmax(accuracies_mean)
        print "Best k = {}, maximum value ={}".format(k_choices[max_index], accuracies_mean[max_index])
        plt.show()
        return
    def model_with_best_k(self):
        # Based on the cross-validation results above, choose the best value for k,   
        # retrain the classifier using all the training data, and test it on the test
        # data. You should be able to get above 28% accuracy on the test data.
        best_k = 10
        
        classifier = KNearestNeighbor()
        classifier.train(self.X_train, self.y_train)
        y_test_pred = classifier.predict(self.X_test, k=best_k)
        
        # Compute and display the accuracy
        num_correct = np.sum(y_test_pred == self.y_test)
        accuracy = float(num_correct) / self.num_test
        print 'Got %d / %d correct => accuracy: %f' % (num_correct, self.num_test, accuracy)
        return
    def run(self):
        self.load_data()
#         self.visualize_data() 
        self.subsample_reshape()
        self.train()
#         self.predict()
#         self.compute_distance_noloop()
#         self.compare_vectorization_speed()
#         self.do_cross_validation()
        self.model_with_best_k()
#         self.compute_distance_oneloop()
        
        return





if __name__ == "__main__":   
    obj= KNNModel()
    obj.run()