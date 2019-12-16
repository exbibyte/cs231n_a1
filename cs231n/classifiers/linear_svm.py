from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #init back prop
    dloss = 1.0
    dtemp2 = dloss * 1/num_train #(2)
        
    for i in range(num_train):
        
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        dtemp5 = X[i,:]
        
        for j in range(num_classes):
            if j == y[i]:
                continue
            
            dtemp3 = 1; #takes care of path going through scores[j]
            dtemp4 = -1.0; #takes care of path going through correct_class_score
            
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            
            if margin > 0:
                
                #commit backprop value for max function wrt. W
                
                dW[:,j] += dtemp2 * dtemp3 * dtemp5
                dW[:,y[i]] += dtemp2 * dtemp4 * dtemp5 #y[i] constant as j varies
                
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train #(2)

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) #(1)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += dloss * reg * 2*W; #(1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ref = np.dot(X,W)[range(X.shape[0]),y]
    
    temp = np.dot(X,W) - np.expand_dims(ref, axis=1) + 1
    temp[range(X.shape[0]), y]=0

    gt_zero = (temp>0)
    
    loss = reg * np.sum(W*W) + 1/X.shape[0] * np.sum(np.sum(np.clip(temp, 0, None), axis=1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dloss = 1.0
    
    dW += dloss * reg * 2.0 * W

    #take care of clipping here
    dtemp = dloss * 1/X.shape[0] * 1.0 * gt_zero #dim: N x C
    
    dW += np.dot(X.T, dtemp) #dim: X.T: (D x N), dtemp: (N x C)

    dtemp2 = dloss * 1/X.shape[0] * -1.0 * gt_zero

    
    #don't need to take care of temp[range(X.shape[0]), y]=0 step since derivative of loss cancels out wrt.W since derivative of delta (1) wrt W is 0

    dtemp3 = np.sum(dtemp2,axis=1)

    dtemp4 = X * np.expand_dims(dtemp3,axis=1);
    
    for i in range(X.shape[0]):
        j = y[i]
        dW[:,j] += dtemp4[i,:]
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
