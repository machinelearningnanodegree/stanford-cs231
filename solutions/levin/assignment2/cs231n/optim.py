import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as w giving the gradient of the
        loss with respect to w.
    - config: A dictionary containing hyperparameter values such as learning rate,
        momentum, etc. If the update rule requires caching values over many
        iterations, then config will also hold these cached values.

Returns:
    - next_w: The next point after the update.
    - config: The config dictionary to be passed to the next iteration of the
        update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
        Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
        average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    mu = config.get('momentum')
    learning_rate = config.get('learning_rate')
    
    next_w = None
    #############################################################################
    # TODO: Implement the momentum update formula. Store the updated value in     #
    # the next_w variable. You should also use and update the velocity v.             #
    #############################################################################
    #penalty on gradient direction change (jitter), encourgement on consistent gradient direction 
    v = mu * v + (-learning_rate * dw)
    next_w = w+ v
    #############################################################################
    #                                                         END OF YOUR CODE                                                            #
    #############################################################################
    config['velocity'] = v

    return next_w, config



def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
        gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))
    
    learning_rate = config.get('learning_rate')
    decay_rate = config.get('decay_rate')
    epsilon = config.get('epsilon')
    cache = config.get('cache')

    next_x = None
    #############################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x     #
    # in the next_x variable. Don't forget to update cache value stored in            #    
    # config['cache'].                                                                                                                    #
    #############################################################################
    cache = decay_rate * cache + (1 - decay_rate) * (dx**2)
    next_x =x -learning_rate *dx/(np.sqrt(cache) + epsilon)
    config['cache'] = cache
    #############################################################################
    #                                                         END OF YOUR CODE                                                            #
    #############################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)
    
    learning_rate = config.get('learning_rate')
    beta1 = config.get('beta1')
    beta2 = config.get('beta2')
    epsilon = config.get('epsilon')
    m = config.get('m')
    v = config.get('v')
    t = config.get('t') + 1
    
    next_x = None
    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in     #
    # the next_x variable. Don't forget to update the m, v, and t variables         #
    # stored in config.                                                                                                                 #
    #############################################################################
#     for t in xrange(0, t_count):
    m = beta1*m + (1-beta1)*dx
    v = beta2*v + (1-beta2)*(dx**2)
    mb = m/(1 - beta1 ** t)
    vb = v/(1 - beta2 ** t)
    next_x =x - learning_rate * mb / (np.sqrt(vb) + epsilon)
    
    
    
    config['m'] = m
    config['v'] = v
    config['t'] = t
    #############################################################################
    #                                                         END OF YOUR CODE                                                            #
    #############################################################################
    
    return next_x, config

    
    
    
