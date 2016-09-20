import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n import data_utils
from assignment1.cs231n.classifiers.softmax import softmax_loss_naive
from assignment1.cs231n.gradient_check import grad_check_sparse
import time



class Softmaxmodel(object):
    def __init__(self):
       
        return
    def get_CIFAR10_data(self,num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
        """
        Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
        it for the linear classifier. These are the same steps as we used for the
        SVM, but condensed to a single function.  
        """
        # Load the raw CIFAR-10 data
        cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
        # subsample the data
        mask = range(num_training, num_training + num_validation)
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = range(num_training)
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]
        mask = np.random.choice(num_training, num_dev, replace=False)
        X_dev = X_train[mask]
        y_dev = y_train[mask]
        
        # Preprocessing: reshape the image data into rows
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_val = np.reshape(X_val, (X_val.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
        
        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis = 0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        X_dev -= mean_image
        
        # add bias dimension and transform into columns
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
        X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
        print 'Train data shape: ', X_train.shape
        print 'Train labels shape: ', y_train.shape
        print 'Validation data shape: ', X_val.shape
        print 'Validation labels shape: ', y_val.shape
        print 'Test data shape: ', X_test.shape
        print 'Test labels shape: ', y_test.shape
        print 'dev data shape: ', X_dev.shape
        print 'dev labels shape: ', y_dev.shape
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_dev = X_dev
        self.y_dev = y_dev
        
        return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev
    
    def compute_loss(self):
        # First implement the naive softmax loss function with nested loops.
        # Open the file cs231n/classifiers/softmax.py and implement the
        # softmax_loss_naive function.

        # Generate a random softmax weight matrix and use it to compute the loss.
        W = np.random.randn(3073, 10) * 0.0001
        loss, grad = softmax_loss_naive(W, self.X_dev, self.y_dev, 0.0)
        
        # As a rough sanity check, our loss should be something close to -log(0.1).
        print 'loss: %f' % loss
        print 'sanity check: %f' % (-np.log(0.1))
        
        
        return
    def compute_gradient(self):
        # Complete the implementation of softmax_loss_naive and implement a (naive)
        # version of the gradient that uses nested loops.
        W = np.random.randn(3073, 10) * 0.0001
        loss, grad = softmax_loss_naive(W, self.X_dev, self.y_dev, 0.0)
        
        # As we did for the SVM, use numeric gradient checking as a debugging tool.
        # The numeric gradient should be close to the analytic gradient.
        
        f = lambda w: softmax_loss_naive(w, self.X_dev, self.y_dev, 0.0)[0]
        grad_numerical = grad_check_sparse(f, W, grad, 10)
        
        # similar to SVM case, do another gradient check with regularization
        loss, grad = softmax_loss_naive(W, self.X_dev, self.y_dev, 1e2)
        f = lambda w: softmax_loss_naive(w, self.X_dev, self.y_dev, 1e2)[0]
        grad_numerical = grad_check_sparse(f, W, grad, 10)
        return
    def run(self):
        self.get_CIFAR10_data()
        self.compute_loss()
#         self.compute_gradient()
     
        
        return





if __name__ == "__main__":   
    obj= Softmaxmodel()
    obj.run()