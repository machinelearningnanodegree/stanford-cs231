import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n import data_utils
from assignment1.cs231n.classifiers.linear_svm import svm_loss_naive
from assignment1.cs231n.gradient_check import grad_check_sparse
import time
from assignment1.cs231n.classifiers.linear_svm import svm_loss_vectorized
from assignment1.cs231n.classifiers.linear_classifier import LinearSVM


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
    def compute_loss_grad_naive(self):
        # Evaluate the naive implementation of the loss we provided for you
        # generate a random SVM weight matrix of small numbers
        W = np.random.randn(3073, 10) * 0.0001 
        
        loss, grad = svm_loss_naive(W, self.X_dev, self.y_dev, 0.00001)
        print 'loss: %f' % (loss, )
        #Once you've implemented the gradient, recompute it with the code below
        # and gradient check it with the function we provided for you
        
        # Compute the loss and its gradient at W.
        loss, grad = svm_loss_naive(W, self.X_dev, self.y_dev, 0.0)
        
        # Numerically compute the gradient along several randomly chosen dimensions, and
        # compare them with your analytically computed gradient. The numbers should match
        # almost exactly along all dimensions.
        
        f = lambda w: svm_loss_naive(w, self.X_dev, self.y_dev, 0.0)[0]
        grad_numerical = grad_check_sparse(f, W, grad)
        
        # do the gradient check once again with regularization turned on
        # you didn't forget the regularization gradient did you?
        loss, grad = svm_loss_naive(W, self.X_dev, self.y_dev, 1e2)
        f = lambda w: svm_loss_naive(w, self.X_dev, self.y_dev, 1e2)[0]
        grad_numerical = grad_check_sparse(f, W, grad)
        return
    def vectorize_loss_computation(self):
        # Next implement the function svm_loss_vectorized; for now only compute the loss;
        # we will implement the gradient in a moment.
        W = np.random.randn(3073, 10) * 0.0001
        tic = time.time()
        loss_naive, grad_naive = svm_loss_naive(W, self.X_dev, self.y_dev, 0.00001)
        toc = time.time()
        print 'Naive loss: %e computed in %fs' % (loss_naive, toc - tic)
        
        
        tic = time.time()
        loss_vectorized, _ = svm_loss_vectorized(W, self.X_dev, self.y_dev, 0.00001)
        toc = time.time()
        print 'Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)
        
        # The losses should match but your vectorized implementation should be much faster.
        print 'difference: %f' % (loss_naive - loss_vectorized)
        return
    def vectorize_grad_computation(self):
        # Complete the implementation of svm_loss_vectorized, and compute the gradient
        # of the loss function in a vectorized way.
        
        # The naive implementation and the vectorized implementation should match, but
        # the vectorized version should still be much faster.
        W = np.random.randn(3073, 10) * 0.0001
        tic = time.time()
        _, grad_naive = svm_loss_naive(W, self.X_dev, self.y_dev, 0.00001)
        toc = time.time()
        print 'Naive loss and gradient: computed in %fs' % (toc - tic)
        
        tic = time.time()
        _, grad_vectorized = svm_loss_vectorized(W, self.X_dev, self.y_dev, 0.00001)
        toc = time.time()
        print 'Vectorized loss and gradient: computed in %fs' % (toc - tic)
        
        # The loss is a single number, so it is easy to compare the values computed
        # by the two implementations. The gradient on the other hand is a matrix, so
        # we use the Frobenius norm to compare them.
        difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
        print 'difference: %f' % difference
        return
    def use_gradient_descent(self):
        # In the file linear_classifier.py, implement SGD in the function
        # LinearClassifier.train() and then run it with the code below.
        
        svm = LinearSVM()
        tic = time.time()
        loss_hist = svm.train(self.X_train, self.y_train, learning_rate=2e-7, reg=1e4,
                              num_iters=1500, verbose=True)
#         loss_hist = svm.train(self.X_train, self.y_train, learning_rate=1e-7, reg=5e4,
#                               num_iters=1500, verbose=True)
        toc = time.time()
        print 'That took %fs' % (toc - tic)
        # A useful debugging strategy is to plot the loss as a function of
        # iteration number:
        plt.plot(loss_hist)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.show()
        # Write the LinearSVM.predict function and evaluate the performance on both the
        # training and validation set
        y_train_pred = svm.predict(self.X_train)
        print 'training accuracy: %f' % (np.mean(self.y_train == y_train_pred), )
        y_val_pred = svm.predict(self.X_val)
        print 'validation accuracy: %f' % (np.mean(self.y_val == y_val_pred), )
        return
    def parameter_tuning(self):
        learning_rates = [1e-7, 2e-7]
        regularization_strengths = [5e4, 1e4]
        # results is dictionary mapping tuples of the form
        # (learning_rate, regularization_strength) to tuples of the form
        # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
        # of data points that are correctly classified.
        results = {}
        best_val = -1   # The highest validation accuracy that we have seen so far.
        best_svm = None # The LinearSVM object that achieved the highest validation rate.
        num_iters = 1500
        for learning_rate in learning_rates:
            for regularization_strength in regularization_strengths:
                print "learning_rage {}, regularization_strength {}".format(learning_rate, regularization_strength)
                #train it
                svm = LinearSVM()
                svm.train(self.X_train, self.y_train, learning_rate=learning_rate, reg=regularization_strength,
                              num_iters=num_iters, verbose=True)
                #predict
                y_train_pred = svm.predict(self.X_train)
                training_accuracy = np.mean(self.y_train == y_train_pred)
                y_val_pred = svm.predict(self.X_val)
                validation_accuracy = np.mean(self.y_val == y_val_pred)
                results[(learning_rate,regularization_strength)] = training_accuracy, validation_accuracy
                if validation_accuracy > best_val:
                    best_val = validation_accuracy
                    best_svm = svm
                
        # Print out results.
        for lr, reg in sorted(results):
            train_accuracy, val_accuracy = results[(lr, reg)]
            print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                        lr, reg, train_accuracy, val_accuracy)
            
        print 'best validation accuracy achieved during cross-validation: %f' % best_val
        self.visualize_crossvaliaton(results)
        self.evaluate_bestsvm(best_svm)
        self.visualize_bestsvm_weights(best_svm)
        return
    def evaluate_bestsvm(self, best_svm):
        # Evaluate the best svm on test set
        y_test_pred = best_svm.predict(self.X_test)
        test_accuracy = np.mean(self.y_test == y_test_pred)
        print 'linear SVM on raw pixels final test set accuracy: %f' % test_accuracy
        return
    def visualize_bestsvm_weights(self,best_svm):
        # Visualize the learned weights for each class.
        # Depending on your choice of learning rate and regularization strength, these may
        # or may not be nice to look at.
        w = best_svm.W[:-1,:] # strip out the bias
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
    def visualize_crossvaliaton(self, results):
        # Visualize the cross-validation results
        import math
        x_scatter = [math.log10(x[0]) for x in results]
        y_scatter = [math.log10(x[1]) for x in results]
        
        # plot training accuracy
        marker_size = 100
        colors = [results[x][0] for x in results]
        plt.subplot(2, 1, 1)
        plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
        plt.colorbar()
        plt.xlabel('log learning rate')
        plt.ylabel('log regularization strength')
        plt.title('CIFAR-10 training accuracy')
        
        # plot validation accuracy
        colors = [results[x][1] for x in results] # default size of markers is 20
        plt.subplot(2, 1, 2)
        plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
        plt.colorbar()
        plt.xlabel('log learning rate')
        plt.ylabel('log regularization strength')
        plt.title('CIFAR-10 validation accuracy')
        plt.show()
        return
    def run(self):
        self.load_data()
        self.preprocess()
        self.parameter_tuning()
#         self.use_gradient_descent()
#         self.compute_loss_grad_naive()
#         self.vectorize_loss_computation()
#         self.vectorize_grad_computation()
#         self.visualize_data() 
        
     
        
        return





if __name__ == "__main__":   
    obj= SVModel()
    obj.run()