from __future__ import division
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    n_samples = X.shape[0]
    n_classes = W.shape[1]
    dW = np.zeros_like(W)

    #############################################################################
    # Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # for each image in the data set
    for i in xrange(n_samples):
        true_class = y[i]
        scores = X[i].dot(W)

        # adjust for numerical stability
        scores -= np.max(scores)

        # calculate something
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(exp_scores)
        prob = exp_scores/sum_exp_scores

        # update loss
        loss -= np.log(prob[true_class])

        for j in xrange(n_classes):
            if j == true_class:
                dW[:, j] += (prob[true_class] - 1) * X[i, :]
                continue
            dW[:, j] += prob[j] * X[i, :]

    loss /= n_samples
    dW /= n_samples

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    n_samples = X.shape[0]
    n_classes = W.shape[1]

    #############################################################################
    # Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    scores = X.dot(W)
    scores -= np.max(scores)

    exp_scores = np.exp(scores)
    exp_class_scores = np.choose(y, exp_scores.T)

    mat_reference = np.tile(exp_class_scores, n_classes).reshape(n_classes, n_samples).T
    truth_table = np.equal(mat_reference, exp_scores)

    sum_exp_scores = np.sum(exp_scores, axis=1)
    prob = exp_class_scores/sum_exp_scores

    losses = - np.log(prob)
    losses /= n_samples
    losses += 0.5 * reg * np.sum(W * W)
    loss += np.sum(losses)

    gradients = exp_scores.T/sum_exp_scores
    gradients.T[truth_table] = (gradients.T[truth_table] - 1)

    dW += np.dot(X.T, gradients.T)
    dW /= n_samples
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW
