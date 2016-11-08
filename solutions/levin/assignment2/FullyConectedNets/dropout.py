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
import time




class Dropout(object):
    def __init__(self):
       
        return
    
    def rel_error(self, x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))  
    
    
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
        
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_val = X_val
#         self.y_val = y_val
#         self.X_test = X_test
#         self.y_test = y_test
        self.data = {
                     'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val}
        return X_train, y_train, X_val, y_val,X_test,y_test
  
    
        return
    def check_dropout_forward(self):
        x = np.random.randn(500, 500) + 10

        for p in [0.3, 0.6, 0.75]:
            out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
            out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})
            
            print 'Running tests with p = ', p
            print 'Mean of input: ', x.mean()
            print 'Mean of train-time output: ', out.mean()
            print 'Mean of test-time output: ', out_test.mean()
            print 'Fraction of train-time output set to zero: ', (out == 0).mean()
            print 'Fraction of test-time output set to zero: ', (out_test == 0).mean()
            print
        return
    def check_dropout_backward(self):
        x = np.random.randn(10, 10) + 10
        dout = np.random.randn(*x.shape)
        
        dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}
        out, cache = dropout_forward(x, dropout_param)
        dx = dropout_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)
        
        print 'dx relative error: ', self.rel_error(dx, dx_num)
        return
    def check_fullconn_withdropout(self):
        N, D, H1, H2, C = 2, 15, 20, 30, 10
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=(N,))
        
        for dropout in [0, 0.25, 0.5]:
            print 'Running check with dropout = ', dropout
            model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                  weight_scale=5e-2, dtype=np.float64,
                                  dropout=dropout, seed=123)
        
            loss, grads = model.loss(X, y)
            print 'Initial loss: ', loss
        
            for name in sorted(grads):
                f = lambda _: model.loss(X, y)[0]
                grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
                print '%s relative error: %.2e' % (name, self.rel_error(grad_num, grads[name]))
                print
        return
    def experiment_regularization(self):
        # Train two identical nets, one with dropout and one without
        data = self.data
        num_train = 500
        small_data = {
          'X_train': data['X_train'][:num_train],
          'y_train': data['y_train'][:num_train],
          'X_val': data['X_val'],
          'y_val': data['y_val'],
        }
        
        solvers = {}
        dropout_choices = [0, 0.75]
        for dropout in dropout_choices:
            model = FullyConnectedNet([500], dropout=dropout)
            print dropout
            
            solver = Solver(model, small_data,
                            num_epochs=25, batch_size=100,
                            update_rule='adam',
                            optim_config={
                              'learning_rate': 5e-4,
                            },
                            verbose=True, print_every=100)
            solver.train()
            solvers[dropout] = solver
        # Plot train and validation accuracies of the two models

        train_accs = []
        val_accs = []
        for dropout in dropout_choices:
            solver = solvers[dropout]
            train_accs.append(solver.train_acc_history[-1])
            val_accs.append(solver.val_acc_history[-1])
        
        plt.subplot(3, 1, 1)
        for dropout in dropout_choices:
            plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
        plt.title('Train accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(ncol=2, loc='lower right')
          
        plt.subplot(3, 1, 2)
        for dropout in dropout_choices:
            plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
        plt.title('Val accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(ncol=2, loc='lower right')
        
        plt.gcf().set_size_inches(15, 15)
        plt.show()
        return
    def run(self):
        self.get_CIFAR10_data()
#         self.check_dropout_forward()
#         self.check_dropout_backward()
#         self.check_fullconn_withdropout()
        self.experiment_regularization()
        return





if __name__ == "__main__":   
    obj= Dropout()
    obj.run()