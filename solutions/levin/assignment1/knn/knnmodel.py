import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n import data_utils
from assignment1.cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor


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
        two_loop_time = self.time_function(self.classifier.compute_distances_two_loops, self.X_test)
        print 'Two loop version took %f seconds' % two_loop_time
        
        one_loop_time = self.time_function(self.classifier.compute_distances_one_loop, self.X_test)
        print 'One loop version took %f seconds' % one_loop_time
        
        no_loop_time = self.time_function(self.classifier.compute_distances_no_loops, self.X_test)
        print 'No loop version took %f seconds' % no_loop_time
        return
        # you should see significantly faster performance with the fully vectorized implementation
    def run(self):
        self.load_data()
#         self.visualize_data()
        self.subsample_reshape()
        self.train()
#         self.predict()
#         self.compute_distance_noloop()
        self.compare_vectorization_speed()
#         self.compute_distance_oneloop()
        
        return





if __name__ == "__main__":   
    obj= KNNModel()
    obj.run()