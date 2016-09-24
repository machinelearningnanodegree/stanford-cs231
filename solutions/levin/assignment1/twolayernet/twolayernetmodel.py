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
                    num_iters=110, verbose=True)
        
        print 'Final training loss: ', stats['loss_history'][-1]
        
        # plot the loss history
        plt.plot(stats['loss_history'])
        plt.xlabel('iteration')
        plt.ylabel('training loss')
        plt.title('Training Loss history')
        plt.show()
        return
    def run(self):
        self.init_model_data()
        self.compute_scores()
        self.compute_loss()
        self.compute_gradient()
        self.train_ontoydata()
        
        return





if __name__ == "__main__":   
    obj= TwoLayerNetModel()
    obj.run()