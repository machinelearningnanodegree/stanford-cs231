import sys
import os
from astropy.units import ys

sys.path.insert(0, os.path.abspath('..'))

import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n import data_utils
from assignment1.cs231n.classifiers.neural_net import TwoLayerNet
from assignment1.cs231n.gradient_check import eval_numerical_gradient
from assignment1.cs231n.vis_utils import visualize_grid




class TwoLayerNetModel(object):
    def __init__(self):
       
        return
    def rel_error(self, x, y):
            """ returns relative error """
            return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    def init_model_data(self):
        self.input_size = 4
        self.hidden_size = 10
        self.num_classes = 3
        self.num_inputs = 5
        
        np.random.seed(0)
        self.net = TwoLayerNet(self.input_size, self.hidden_size, self.num_classes, std=1e-1)
        self.X, self.y = self.init_toy_data()
        return
    def compute_scores(self):
        scores = self.net.loss(self.X)
        print 'Your scores:'
        print scores
        print
        print 'correct scores:'
        correct_scores = np.asarray([
          [-0.81233741, -1.27654624, -0.70335995],
          [-0.17129677, -1.18803311, -0.47310444],
          [-0.51590475, -1.01354314, -0.8504215 ],
          [-0.15419291, -0.48629638, -0.52901952],
          [-0.00618733, -0.12435261, -0.15226949]])
        print correct_scores
        print
        
        # The difference should be very small. We get < 1e-7
        print 'Difference between your scores and correct scores:'
        print np.sum(np.abs(scores - correct_scores))
        return
    def compute_loss(self):
        loss, _ = self.net.loss(self.X, self.y, reg=0.1)
        correct_loss = 1.30378789133
        
        # should be very small, we get < 1e-12
        print 'Difference between your loss and correct loss:'
        print np.sum(np.abs(loss - correct_loss))
        return
    def compute_gradient(self):
        loss, grads = self.net.loss(self.X, self.y, reg=0.1)
        # these should all be less than 1e-8 or so
        for param_name in grads:
            f = lambda W: self.net.loss(self.X, self.y, reg=0.1)[0]
            param_grad_num = eval_numerical_gradient(f, self.net.params[param_name], verbose=False)
            print '%s max relative error: %e' % (param_name, self.rel_error(param_grad_num, grads[param_name]))
        return
    def init_toy_data(self):
        np.random.seed(1)
        X = 10 * np.random.randn(self.num_inputs, self.input_size)
        y = np.array([0, 1, 2, 2, 1])
        return X, y
    
    def train_ontoydata(self):
        stats = self.net.train(self.X, self.y, self.X, self.y,
                    learning_rate=1e-1, reg=1e-5,
                    num_iters=100, verbose=True)
        
        print 'Final training loss: ', stats['loss_history'][-1]
        
        # plot the loss history
        plt.plot(stats['loss_history'])
        plt.xlabel('iteration')
        plt.ylabel('training loss')
        plt.title('Training Loss history')
        plt.show()
        return
    def get_CIFAR10_data(self, num_training=49000, num_validation=1000, num_test=1000):
        """
        Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
        it for the two-layer neural net classifier. These are the same steps as
        we used for the SVM, but condensed to a single function.  
        """
        # Load the raw CIFAR-10 data
        cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
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
        
        return
    def train_network(self):
        self.input_size = 32 * 32 * 3
        self.hidden_size = 300
        self.num_classes = 10
        self.net = TwoLayerNet(self.input_size, self.hidden_size, self.num_classes)
        
        # Train the network
        num_iters = 2000
        reg=0
        learning_rate_decay=0.95
        learning_rate=1e-3
        batch_size=200
        stats = self.net.train(self.X_train, self.y_train, self.X_val, self.y_val,
                    num_iters=num_iters, batch_size=batch_size,
                    learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                    reg=reg, verbose=True)
        
        # Predict on the validation set
        val_acc = (self.net.predict(self.X_val) == self.y_val).mean()
        train_acc = (self.net.predict(self.X_train) == self.y_train).mean()
        print 'Train accuracy:{}, Validation accuracy:{}'.format(train_acc, val_acc)
#         self.debug_training(stats)
#         self.show_net_weights(self.net)
        return
    def show_net_weights(self, net):
        W1 = net.params['W1']
        W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
        plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
        plt.gca().axis('off')
        plt.show()
        return
    def param_tuning(self):
        input_size = 32 * 32 * 3
        hidden_size = 200
        num_classes = 10
        best_val = -1
        
        
        # Train the network
        num_iters = 1800
        batch_size=200
        # hyperparameters
        learning_rates = [8e-4]
        regs = [5e-2]
        learning_rate_decays = [0.95]
        for lr in learning_rates:
            for reg in regs:
                for decay in learning_rate_decays:
                    print("learning rate: {}, regulation: {}, decay: {}".format(lr, reg, decay))
                    net = TwoLayerNet(input_size, hidden_size, num_classes)
                    net.train(self.X_train, self.y_train, self.X_val, self.y_val,
                                      num_iters=num_iters,
                                      batch_size=batch_size,
                                      learning_rate=lr,
                                      learning_rate_decay= decay,
                                      reg=reg,
                                      verbose=False)

        
                    # Predict on the validation set
                    val_acc = (net.predict(self.X_val) == self.y_val).mean()
                    train_acc = (net.predict(self.X_train) == self.y_train).mean()
                    if val_acc > best_val:
                        best_net = net
                        best_val = val_acc
                    print 'Train accuracy:{}, Validation accuracy:{}'.format(train_acc, val_acc)
        print 'Best accuracy:{}'.format(best_val)
#         self.show_net_weights(best_net)
        return
    def debug_training(self, stats):
        # Plot the loss function and train / validation accuracies
        plt.subplot(2, 1, 1)
        plt.plot(stats['loss_history'])
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        
        plt.subplot(2, 1, 2)
        plt.plot(stats['train_acc_history'], label='train')
        plt.plot(stats['val_acc_history'], label='val')
        plt.title('Classification accuracy history')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Clasification accuracy')
        plt.show()
        return
      
    def run(self):
        self.init_model_data()
        self.compute_scores()
        self.compute_loss()
        self.compute_gradient()
        
#         self.train_ontoydata()
#         self.get_CIFAR10_data()
#         self.train_network()
#         self.param_tuning()
        
        return





if __name__ == "__main__":   
    obj= TwoLayerNetModel()
    obj.run()