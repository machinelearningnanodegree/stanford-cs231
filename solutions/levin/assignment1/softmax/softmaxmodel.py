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
from assignment1.cs231n.classifiers.softmax import softmax_loss_vectorized
import time
from assignment1.cs231n.classifiers.linear_classifier import Softmax



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
    def compute_vectorized_loss_grad(self):
        # Now that we have a naive implementation of the softmax loss function and its gradient,
        # implement a vectorized version in softmax_loss_vectorized.
        # The two versions should compute the same results, but the vectorized version should be
        # much faster.
        W = np.random.randn(3073, 10) * 0.0001
        tic = time.time()
        loss_naive, grad_naive = softmax_loss_naive(W, self.X_dev, self.y_dev, 0.00001)
        toc = time.time()
        print 'naive loss: %e computed in %fs' % (loss_naive, toc - tic)
        
        
        tic = time.time()
        loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, self.X_dev, self.y_dev, 0.00001)
        toc = time.time()
        print 'vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)
        
        # As we did for the SVM, we use the Frobenius norm to compare the two versions
        # of the gradient.
        grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
        print 'Loss difference: %f' % np.abs(loss_naive - loss_vectorized)
        print 'Gradient difference: %f' % grad_difference
        return
    def parameter_tuning(self):
        learning_rates = [1e-7, 5e-8]
        regularization_strengths = [5e4, 1e3]
        # results is dictionary mapping tuples of the form
        # (learning_rate, regularization_strength) to tuples of the form
        # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
        # of data points that are correctly classified.
        results = {}
        best_val = -1   # The highest validation accuracy that we have seen so far.
        best_model = None # The LinearSVM object that achieved the highest validation rate.
        num_iters = 1500
        for learning_rate in learning_rates:
            for regularization_strength in regularization_strengths:
                print "learning_rage {}, regularization_strength {}".format(learning_rate, regularization_strength)
                #train it
                model = Softmax()
                model.train(self.X_train, self.y_train, learning_rate=learning_rate, reg=regularization_strength,
                              num_iters=num_iters, verbose=True)
                #predict
                y_train_pred = model.predict(self.X_train)
                training_accuracy = np.mean(self.y_train == y_train_pred)
                y_val_pred = model.predict(self.X_val)
                validation_accuracy = np.mean(self.y_val == y_val_pred)
                results[(learning_rate,regularization_strength)] = training_accuracy, validation_accuracy
                if validation_accuracy > best_val:
                    best_val = validation_accuracy
                    best_model = model
                
        # Print out results.
        for lr, reg in sorted(results):
            train_accuracy, val_accuracy = results[(lr, reg)]
            print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                        lr, reg, train_accuracy, val_accuracy)
            
        print 'best validation accuracy achieved during cross-validation: %f' % best_val
        self.evaluate_bestmodel(best_model)
        self.visualize_bestmodel_weights(best_model)
        return
    def evaluate_bestmodel(self, best_model):
        # Evaluate the best svm on test set
        y_test_pred = best_model.predict(self.X_test)
        test_accuracy = np.mean(self.y_test == y_test_pred)
        print 'linear SVM on raw pixels final test set accuracy: %f' % test_accuracy
        return
    def visualize_bestmodel_weights(self,best_model):
        # Visualize the learned weights for each class.
        # Depending on your choice of learning rate and regularization strength, these may
        # or may not be nice to look at.
        w = best_model.W[:-1,:] # strip out the bias
        w = w.reshape(32, 32, 3, 10)
        w_min, w_max = np.min(w), np.max(w)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i in xrange(10):
            plt.subplot(2, 5, i + 1)
            
            # Rescale the weights to be between 0 and 255
            wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
            plt.imshow(wimg.astype('uint8'))
            plt.axis('off')
            plt.title(classes[i])
        plt.show()
        return
    def run(self):
        self.get_CIFAR10_data()
        self.parameter_tuning()
#         self.compute_loss()
#         self.compute_gradient()
#         self.compute_vectorized_loss_grad()
     
        
        return





if __name__ == "__main__":   
    obj= Softmaxmodel()
    obj.run()