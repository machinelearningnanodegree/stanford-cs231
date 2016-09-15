import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n import data_utils
from sklearn.cross_validation import KFold


class SVModel(object):
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
    
    def split_data(self):
        # Split the data into train, val, and test sets. In addition we will
        # create a small development set as a subset of the training data;
        # we can use this for development so our code runs faster.
        num_training = 49000
        num_validation = 1000
        num_test = 1000
        num_dev = 500
        
        # Our validation set will be num_validation points from the original
        # training set.
        mask = range(num_training, num_training + num_validation)
        self.X_val = self.X_train[mask]
        self.y_val = self.y_train[mask]
        
        # Our training set will be the first num_train points from the original
        # training set.
        mask = range(num_training)
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]
        
        # We will also make a development set, which is a small subset of
        # the training set.
        mask = np.random.choice(num_training, num_dev, replace=False)
        self.X_dev = self.X_train[mask]
        self.y_dev = self.y_train[mask]
        
        # We use the first num_test points of the original test set as our
        # test set.
        mask = range(num_test)
        self.X_test = self.X_test[mask]
        self.y_test = self.y_test[mask]
        
        print 'Train data shape: ', self.X_train.shape
        print 'Train labels shape: ', self.y_train.shape
        print 'Validation data shape: ', self.X_val.shape
        print 'Validation labels shape: ', self.y_val.shape
        print 'Test data shape: ', self.X_test.shape
        print 'Test labels shape: ', self.y_test.shape
        return
    def reshape_data(self):
        # Preprocessing: reshape the image data into rows
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], -1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], -1))
        self.X_dev = np.reshape(self.X_dev, (self.X_dev.shape[0], -1))
        
        # As a sanity check, print out the shapes of the data
        print 'Training data shape: ', self.X_train.shape
        print 'Validation data shape: ', self.X_val.shape
        print 'Test data shape: ', self.X_test.shape
        print 'dev data shape: ', self.X_dev.shape
        return
    def substract_mean(self):
        # Preprocessing: subtract the mean image
        # first: compute the image mean based on the training data
        mean_image = np.mean(self.X_train, axis=0)
#         print mean_image[:10] # print a few of the elements
#         plt.figure(figsize=(4,4))
#         plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
#         plt.show()
        
        # second: subtract the mean image from train and test data
        self.X_train -= mean_image
        self.X_val -= mean_image
        self.X_test -= mean_image
        self.X_dev -= mean_image
        return
    def append_bias(self):
        # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
        # only has to worry about optimizing a single weight matrix W.
        self.X_train = np.hstack([self.X_train, np.ones((self.X_train.shape[0], 1))])
        self.X_val = np.hstack([self.X_val, np.ones((self.X_val.shape[0], 1))])
        self.X_test = np.hstack([self.X_test, np.ones((self.X_test.shape[0], 1))])
        self.X_dev = np.hstack([self.X_dev, np.ones((self.X_dev.shape[0], 1))])
        
        print self.X_train.shape, self.X_val.shape, self.X_test.shape, self.X_dev.shape
        return
    def preprocess(self):
        self.split_data()
        self.reshape_data()
        self.substract_mean()
        self.append_bias()
        return
    def run(self):
        self.load_data()
        self.preprocess()
#         self.visualize_data() 
        
     
        
        return





if __name__ == "__main__":   
    obj= SVModel()
    obj.run()