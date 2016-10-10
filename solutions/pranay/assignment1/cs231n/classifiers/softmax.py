import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
    # Initialize the loss and gradient to zero.
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0

    for i in xrange(num_train):
        scores = W.dot(X[:, i])
        scores -= np.max(scores)
        normalized = np.exp(scores) / np.sum(np.exp(scores))
        loss -= np.log(normalized[y[i]])
        for j in xrange(num_classes):
            if j == y[i]:
                dW[y[i], :] += (normalized[y[i]] - 1) * X[:, i]  # this is really a sum over j != y_i
                continue
            dW[j, :] += (normalized[j]) * X[:, i]  # sums each contribution of the x_i's

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Same with gradient
    dW /= num_train

    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)

    # Gradient regularization that carries through per https://piazza.com/class/i37qi08h43qfv?cid=118
    dW += reg * W

    return loss, dW
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    scores = W.dot(X)

    scores -= np.max(scores)  ##subtracting from max to speed it up

    normalized = np.exp(scores) / np.sum(np.exp(scores), axis=0)

    true_class_prob = np.choose(y, normalized)

    loss -= np.log(true_class_prob)

    loss = np.average(loss)  # average over the num_classes

    loss += 0.5 * reg * np.sum(W * W)  # add regularization

    truth_table = np.equal(normalized, true_class_prob)

    normalized[truth_table] = normalized[truth_table] - 1

    dW += np.dot(normalized, X.T)

    dW /= num_train

    dW += reg * W
    return loss, dW