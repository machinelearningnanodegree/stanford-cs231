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
    def test_relu_forward(self):
        # Test the relu_forward function

        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        
        out, _ = relu_forward(x)
        correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                                [ 0.,          0.,          0.04545455,  0.13636364,],
                                [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])
        
        # Compare your output with ours. The error should be around 1e-8
        print 'Testing relu_forward function:'
        print 'difference: ', self.rel_error(out, correct_out)
        return
    def test_relu_backward(self):
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)
        
        dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)
        
        _, cache = relu_forward(x)
        dx = relu_backward(dout, cache)
        
        # The error should be around 1e-12
        print 'Testing relu_backward function:'
        print 'dx error: ', self.rel_error(dx_num, dx)
        return
    def test_affine_relu(self):
        x = np.random.randn(2, 3, 4)
        w = np.random.randn(12, 10)
        b = np.random.randn(10)
        dout = np.random.randn(2, 10)
        
        out, cache = affine_relu_forward(x, w, b)
        dx, dw, db = affine_relu_backward(dout, cache)
        
        dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
        db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)
        
        print 'Testing affine_relu_forward:'
        print 'dx error: ', self.rel_error(dx_num, dx)
        print 'dw error: ', self.rel_error(dw_num, dw)
        print 'db error: ', self.rel_error(db_num, db)
        return
    def test_loss_layer_propogation(self):
        num_classes, num_inputs = 10, 50
        x = 0.001 * np.random.randn(num_inputs, num_classes)
        y = np.random.randint(num_classes, size=num_inputs)
        
        dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
        loss, dx = svm_loss(x, y)
        
        # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
        print 'Testing svm_loss:'
        print 'loss: ', loss
        print 'dx error: ', self.rel_error(dx_num, dx)
        
        dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
        loss, dx = softmax_loss(x, y)
        
        # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
        print '\nTesting softmax_loss:'
        print 'loss: ', loss
        print 'dx error: ', self.rel_error(dx_num, dx)
        return
    def test_two_layer_implementation(self):
        N, D, H, C = 3, 5, 50, 7
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=N)
        
        std = 1e-2
        model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
        
        print 'Testing initialization ... '
        W1_std = abs(model.params['W1'].std() - std)
        b1 = model.params['b1']
        W2_std = abs(model.params['W2'].std() - std)
        b2 = model.params['b2']
        assert W1_std < std / 10, 'First layer weights do not seem right'
        assert np.all(b1 == 0), 'First layer biases do not seem right'
        assert W2_std < std / 10, 'Second layer weights do not seem right'
        assert np.all(b2 == 0), 'Second layer biases do not seem right'
        
        print 'Testing test-time forward pass ... '
        model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
        model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
        model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
        model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
        X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
        scores = model.loss(X)
        correct_scores = np.asarray(
          [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
           [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
           [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
        scores_diff = np.abs(scores - correct_scores).sum()
        assert scores_diff < 1e-6, 'Problem with test-time forward pass'
        
        print 'Testing training loss (no regularization)'
        y = np.asarray([0, 5, 1])
        loss, grads = model.loss(X, y)
        correct_loss = 3.4702243556
        assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
        
        model.reg = 1.0
        loss, grads = model.loss(X, y)
        correct_loss = 26.5948426952
        assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'
        
        for reg in [0.0, 0.7]:
            print 'Running numeric gradient check with reg = ', reg
            model.reg = reg
            loss, grads = model.loss(X, y)
            
            for name in sorted(grads):
                f = lambda _: model.loss(X, y)[0]
                grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
                print '%s relative error: %.2e' % (name, self.rel_error(grad_num, grads[name]))
        return
    
    def run(self):
#         self.test_affine_forward()
#         self.test_affine_backward()
#         self.test_relu_forward()
#         self.test_relu_backward()
#         self.test_affine_relu()
#         self.test_loss_layer_propogation()
        self.test_two_layer_implementation()
        return





if __name__ == "__main__":   
    obj= FullyConnectedNets()
    obj.run()