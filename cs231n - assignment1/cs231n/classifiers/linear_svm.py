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
    for i in range(num_train):
        num_positive_margin = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                num_positive_margin += 1
        dW[:, y[i]] += -1 * num_positive_margin * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """

    # compute the loss
    all_scores = X @ W                                      # (N, C)
    all_cols = range(X.shape[0])                            # (N, )
    correct_class_scores = all_scores[all_cols, y]          # (N, )
    partial_loss = all_scores.T - correct_class_scores + 1  # (C, N)
    partial_loss[partial_loss < 0] = 0
    loss = np.sum(partial_loss) - X.shape[0] * 1
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)

    # compute the gradient
    partial_loss[partial_loss > 0] = 1
    partial_loss.T[range(partial_loss.shape[1]), y] = 0
    partial_loss.T[range(partial_loss.shape[1]), y] = -1 * np.sum(partial_loss.T, axis=1)
    dW = X.T @ partial_loss.T
    dW /= X.shape[0]
    dW += 2 * reg * W

    return loss, dW
