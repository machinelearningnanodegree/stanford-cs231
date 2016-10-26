import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
import matplotlib.pyplot as plt
from assignment2.cs231n.layers import affine_forward
from assignment2.cs231n.layers import affine_backward
from assignment2.cs231n.classifiers.fc_net import *
from assignment2.cs231n.data_utils import get_CIFAR10_data
from assignment2.cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from assignment2.cs231n.solver import Solver





class FullyConnectedNets(object):
    def __init__(self):
       
        return
    
    def rel_error(self, x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))  
    def test_affine_forward(self):
        # Test the affine_forward function

        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3
        
        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)
        
        x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)
        
        out, _ = affine_forward(x, w, b)
        correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                                [ 3.25553199,  3.5141327,   3.77273342]])
        
        # Compare your output with ours. The error should be around 1e-9.
        print 'Testing affine_forward function:'
        print 'difference: ', self.rel_error(out, correct_out)
        return
    def test_affine_backward(self):
        # Test the affine_backward function

        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)
        
        dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
        db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
        
        _, cache = affine_forward(x, w, b)
        dx, dw, db = affine_backward(dout, cache)
        
        # The error should be around 1e-10
        print 'Testing affine_backward function:'
        print 'dx error: ', self.rel_error(dx_num, dx)
        print 'dw error: ', self.rel_error(dw_num, dw)
        print 'db error: ', self.rel_error(db_num, db)
        return
    def run(self):
        self.test_affine_forward()
        self.test_affine_backward()
        return





if __name__ == "__main__":   
    obj= FullyConnectedNets()
    obj.run()