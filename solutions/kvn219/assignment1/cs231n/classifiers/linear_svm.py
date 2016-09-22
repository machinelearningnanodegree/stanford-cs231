from __future__ import division
import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    delta = 1.0

    # for each image in the dataset
    for i in xrange(num_train):

        # calculate the score for the i_th image
        scores = X[i].dot(W)

        # save the score of the true class
        class_score = scores[y[i]]

        # for each possible class
        for j in xrange(num_classes):

            # if its the true class
            if j == y[i]:

                # skip below and continue the loop
                continue

            # find the margin at (i,j)
            margin = scores[j] - class_score + delta

            # if the margin is greater than 0
            if margin > 0:
                loss += margin

                # subtract the i_th row from the column of the true class
                dW[:, y[i]] -= X[i, :]

                # add the i_th row to the column the loop is in
                dW[:, j] += X[i, :]
                # each time you loop through the set of columns

    # get the avg loss and gradient
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    dW += reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    n_classes = W.shape[1]
    n_samples = X.shape[0]
    delta = 1.0

    #############################################################################
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    scores = X.dot(W)
    class_scores = np.choose(y, scores.T).reshape(n_samples, 1)

    ##mat_reference = np.tile(class_scores, n_classes)
    ##Commenting out the above line because with numpy broadcasting you won't need that
    margins = np.maximum(scores - class_scores + delta, 0)

    loss = np.sum(margins)/n_samples
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # create a truth matrix to keep tract of true classes,
    # using float32 to get better precision (makes a big difference)
    truth_table = np.array(margins != 0.0, dtype=np.float32)

    # count the number of margins that are greater than 0 and
    # subtract 1 from the sum of counts to account for the true class
    count = np.sum(truth_table, axis=1) - 1.0

    # substitute the correct classes with the count_of_num_classes
    truth_table[np.arange(n_samples), y] = - count

    # use the inner product to transform the weights into (n_attributes, n_classes)
    dW += np.dot(X.T, truth_table)/n_samples

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    dW += reg * W
    return loss, dW
