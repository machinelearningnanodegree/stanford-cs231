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
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_dimensions = X.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    score = X[i].dot(W)

    #the stabilty trick http://cs231n.github.io/linear-classify/
    score = score - np.max(score)  

    loss = loss - score[y[i]] + np.log(np.exp(score).sum())

    #for the gradient of softmax http://cs231n.github.io/neural-networks-case-study/#grad 

    dscores = np.exp(score)/np.exp(score).sum()
    dscores[y[i]] -= 1 #that is it
    #now make it again (1,C)
    dscores = dscores.reshape(1,num_classes)


    #and now 
    dW += np.dot(X[i].reshape(num_dimensions,1),dscores)


    #now the regularization 
  loss = loss + 0.5 * reg * np.sum(W * W)
  loss = loss/num_train

  dW = dW/num_train
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_dimensions = X.shape[1]
  
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #copy paste from http://cs231n.github.io/neural-networks-case-study/#loss
  score = X.dot(W) # (N,C)
  score = score - np.amax(score,axis = 1,keepdims = True)

  score = np.exp(score) 

  probs = score/np.sum(score,axis = 1, keepdims = True)

  loss = -1*np.log(probs[np.arange(num_train),y]).sum()/num_train

  loss = loss + 0.5 * reg * np.sum(W * W)

  #http://cs231n.github.io/neural-networks-case-study/#grad

  dscores = probs #(N,C)
  dscores[range(num_train),y] -= 1
  dscores = dscores / num_train
  dW = np.dot(X.T,dscores)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

