import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    total_count = 0 #this is for finding the gradient wrto the correct class yi
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        total_count += 1
        dW[:,j] += X[i]
    dW[:,y[i]] -=  total_count * X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  dW += reg * W #diferentiating the regualrization loss


  return loss, dW




def svm_loss_vectorized(W, X, y, reg):

  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  """

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W) # N X C matrix

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  correct_class_scores = scores[np.arange(num_train),y].reshape(num_train,1) # (N,1)
  margin_matrix = scores - correct_class_scores + 1 # (N,C)
  margin_matrix[np.arange(num_train),y] = 0 #for the correct class loss would be zero for sure

  # lets do the max that is margin > 0 thing
  margin_more_than_zero = np.maximum(np.zeros((num_train,num_classes)),margin_matrix)

  #now summing all of it
  loss = np.sum(margin_more_than_zero)
  loss = float(loss)/num_train
  loss = loss + 0.5 * reg * np.sum(W * W) # reg is lamda hyperparameter 


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #for the gradient wrto yi we have to count all margin > 0 for each class
  #lets make a binary matrix which has 1 if the value > 0 or 0 for value == 0
  binary_matrix = margin_more_than_zero #(N,C)

  binary_matrix[margin_more_than_zero > 0] = 1

  #now we will add all the values rows wise this is for 
  count_non_zero = binary_matrix.sum(axis = 1)

  binary_matrix[np.arange(num_train),y] = - 1 * count_non_zero
  dW = np.dot(X.T,binary_matrix)
  dW /= num_train
  dW += reg * W  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
