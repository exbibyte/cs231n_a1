from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #-log(e^(y_i+c) / sum{j} e^(y_j+c)) = -(y_i+c) + log(sum{j} e^(y_j+c))
    
    num_data = X.shape[0]
    num_feature = X.shape[1]

    cache = np.dot(X,W)
    
    c = -np.amax(cache)
    
    dloss = 1

    d_0 = dloss * 1/X.shape[0]
    d_1 = -1.0 *d_0
    
    for i in xrange(num_data):

        loss_temp = 0
        denom =0
        for k in xrange(W.shape[1]):
            h = 0
            for j in xrange(num_feature): #inner product
                h += X[i,j] * W[j,k] #d_4
            denom += np.exp(h+c) #d_3
        
        loss_temp = np.log(denom) #d_2

        d_2 = d_0 * 1/denom
        
        #for computing derivatives
        for k in xrange(W.shape[1]):
            h = 0
            for j in xrange(num_feature): #inner product
                h += X[i,j] * W[j,k]
            d_3 = d_2 * np.exp(h+c)
            
            for j in xrange(num_feature): #inner product
                dW[j,k] += d_3 * X[i,j]
            
        loss_temp_2 = 0
        for j in xrange(num_feature):
            loss_temp_2 += (X[i,j] * W[j,y[i]])
            dW[j,y[i]] += d_1 * X[i,j]
            
        loss_temp_2 += c

        loss_temp_2 = -1.0 * loss_temp_2 #d_1

        loss += loss_temp + loss_temp_2

    loss *= 1/X.shape[0] #d_0

    loss += reg * np.sum(W * W)

    dW += 2.0 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    cache = np.dot(X,W)
        
    c = -np.amax(cache)

    denom = np.sum(np.exp(np.dot(X,W) + c), axis=1) #d_0, d_1

    loss_total = -(np.diag(np.dot(X,W)[:,y]) + c) + np.log(denom) #d_loss_total

    loss = 1/X.shape[0] * np.sum(loss_total) + reg * np.sum(W * W)

    #derivatives

    dW += 2.0 * reg * W
    
    d_loss_total = 1/X.shape[0]

    d_0 = np.divide(d_loss_total, denom)

    d_1 = np.expand_dims(d_0,axis=1) * np.exp(np.dot(X,W) + c)
    
    dW += np.dot(X.T,d_1)

    for i in xrange(X.shape[0]):
        dW[:,y[i]] += d_loss_total * -1.0 * X[i,:]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
