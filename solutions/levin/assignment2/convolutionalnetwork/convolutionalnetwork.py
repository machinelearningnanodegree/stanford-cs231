import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
import matplotlib.pyplot as plt
from assignment2.cs231n.layers import affine_forward
from assignment2.cs231n.layers import affine_backward
from assignment2.cs231n.layers import relu_forward
from assignment2.cs231n.layers import relu_backward
from assignment2.cs231n.layers import svm_loss
from assignment2.cs231n.layers import softmax_loss
from assignment2.cs231n.classifiers.fc_net import *
from assignment2.cs231n.data_utils import get_CIFAR10_data
from assignment2.cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from assignment2.cs231n.solver import Solver
from assignment2.cs231n.layer_utils import affine_relu_forward, affine_relu_backward
from assignment2.cs231n.data_utils import load_CIFAR10
from assignment2.cs231n.optim import sgd_momentum
from assignment2.cs231n.optim import rmsprop
from assignment2.cs231n.optim import adam
from assignment2.cs231n.fast_layers import *
import time




class ConvNet(object):
    def __init__(self):
       
        return
    
    def rel_error(self, x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))  
    def check_conv_naive_forward(self):
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)
        
        conv_param = {'stride': 2, 'pad': 1}
        out, _ = conv_forward_naive(x, w, b, conv_param)
        correct_out = np.array([[[[[-0.08759809, -0.10987781],
                                   [-0.18387192, -0.2109216 ]],
                                  [[ 0.21027089,  0.21661097],
                                   [ 0.22847626,  0.23004637]],
                                  [[ 0.50813986,  0.54309974],
                                   [ 0.64082444,  0.67101435]]],
                                 [[[-0.98053589, -1.03143541],
                                   [-1.19128892, -1.24695841]],
                                  [[ 0.69108355,  0.66880383],
                                   [ 0.59480972,  0.56776003]],
                                  [[ 2.36270298,  2.36904306],
                                   [ 2.38090835,  2.38247847]]]]])
        
        # Compare your output to ours; difference should be around 1e-8
        print 'Testing conv_forward_naive'
        print 'difference: ', self.rel_error(out, correct_out)
        return
    def check_max_pooling_naive_forward(self):
        x_shape = (2, 3, 4, 4)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
        pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
        
        out, _ = max_pool_forward_naive(x, pool_param)
        
        correct_out = np.array([[[[-0.26315789, -0.24842105],
                                  [-0.20421053, -0.18947368]],
                                 [[-0.14526316, -0.13052632],
                                  [-0.08631579, -0.07157895]],
                                 [[-0.02736842, -0.01263158],
                                  [ 0.03157895,  0.04631579]]],
                                [[[ 0.09052632,  0.10526316],
                                  [ 0.14947368,  0.16421053]],
                                 [[ 0.20842105,  0.22315789],
                                  [ 0.26736842,  0.28210526]],
                                 [[ 0.32631579,  0.34105263],
                                  [ 0.38526316,  0.4       ]]]])
        
        # Compare your output with ours. Difference should be around 1e-8.
        print 'Testing max_pool_forward_naive function:'
        print 'difference: ', self.rel_error(out, correct_out)
        return
    def check_max_pooling_naive_backward(self):
        x = np.random.randn(3, 2, 8, 8)
        dout = np.random.randn(3, 2, 4, 4)
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)
        
        out, cache = max_pool_forward_naive(x, pool_param)
        dx = max_pool_backward_naive(dout, cache)
        
        # Your error should be around 1e-12
        print 'Testing max_pool_backward_naive function:'
        print 'dx error: ', self.rel_error(dx, dx_num)
        return
    
    
    def get_CIFAR10_data(self, num_training=49000, num_validation=1000, num_test=1000):
        """
        Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
        it for the two-layer neural net classifier. These are the same steps as
        we used for the SVM, but condensed to a single function.  
        """
        # Load the raw CIFAR-10 data
        cifar10_dir = '../../assignment1/cs231n/datasets/cifar-10-batches-py'
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
        
        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        
        # Reshape data to rows
        X_train = X_train.reshape(num_training, -1)
        X_val = X_val.reshape(num_validation, -1)
        X_test = X_test.reshape(num_test, -1)
        print 'Train data shape: ', X_train.shape
        print 'Train labels shape: ', y_train.shape
        print 'Validation data shape: ', X_val.shape
        print 'Validation labels shape: ', y_val.shape
        print 'Test data shape: ', X_test.shape
        print 'Test labels shape: ', y_test.shape
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.data = {
                     'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val}
        return X_train, y_train, X_val, y_val,X_test,y_test
  
    
        return
    
    def run(self):
        self.get_CIFAR10_data()
#         self.check_conv_naive_forward()
#         self.check_max_pooling_naive_forward()
        self.check_max_pooling_naive_backward()
        
        return





if __name__ == "__main__":   
    obj= ConvNet()
    obj.run()