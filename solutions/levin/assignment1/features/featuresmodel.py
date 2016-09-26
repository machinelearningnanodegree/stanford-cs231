import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n import data_utils





class FeaturesModel(object):
    def __init__(self):
       
        return
    
    def get_CIFAR10_data(self, num_training=49000, num_validation=1000, num_test=1000):
        # Load the raw CIFAR-10 data
        cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
        # Subsample the data
        mask = range(num_training, num_training + num_validation)
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = range(num_training)
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        return
    def run(self):
        self.get_CIFAR10_data()
        
        return





if __name__ == "__main__":   
    obj= FeaturesModel()
    obj.run()