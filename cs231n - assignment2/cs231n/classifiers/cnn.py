from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.C, self.H, self.W = input_dim
        self.pool_size = 2

        hidden_layer_input = num_filters * (self.H // self.pool_size) * (self.W // self.pool_size)
        self.params['W1'] = weight_scale * np.random.randn(num_filters, self.C, filter_size, filter_size)         #(F,C,FS,FS)
        self.params['b1'] = np.zeros(num_filters)                                                                 #(F,)
        self.params['W2'] = weight_scale * np.random.randn(hidden_layer_input, hidden_dim)                        #(C*H/P*W/P,Hidden)
        self.params['b2'] = np.zeros(hidden_dim)                                                                  #(Hidden)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)                               #(Hidden, Class)
        self.params['b3'] = np.zeros(num_classes)                                                                 #(Class,)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # Run forward pass to compute scores.
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache2 = affine_relu_forward(out, W2, b2)
        out, cache3 = affine_forward(out, W3, b3)
        scores = out

        if y is None:
            return scores

        # Compute softmax loss then run backward pass.
        loss, grads = 0, {}
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        dout, dW3, db3 = affine_backward(dout, cache3)
        dout, dW2, db2 = affine_relu_backward(dout, cache2)
        dout, dW1, db1 = conv_relu_pool_backward(dout, cache1)
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        grads['W1'], grads['W2'], grads['W3'] = dW1, dW2, dW3
        grads['b1'], grads['b2'], grads['b3'] = db1, db2, db3

        return loss, grads
