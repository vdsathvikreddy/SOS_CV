from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
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

        # Initialize weights and biases
        C, H, W = input_dim
        # Layer 1: Convolutional layer
        # Input: (N, C, H, W), Filters: (num_filters, C, filter_size, filter_size)
        self.params['W1'] = np.random.normal(0.0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)

        # After conv (preserves H, W), after 2x2 max pool with stride 2, spatial dims are H/2, W/2
        H_pool = H // 2
        W_pool = W // 2
        # Layer 2: Affine layer
        # Input: (num_filters * H/2 * W/2), Output: hidden_dim
        self.params['W2'] = np.random.normal(0.0, weight_scale, (num_filters * H_pool * W_pool, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)

        # Layer 3: Affine layer
        # Input: hidden_dim, Output: num_classes
        self.params['W3'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        # Forward pass
        # Layer 1: conv - relu - 2x2 max pool
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

        # Flatten the output for the affine layer
        N = X.shape[0]
        out1_flat = out1.reshape(N, -1)  # (N, num_filters * H/2 * W/2)

        # Layer 2: affine - relu
        out2, cache2 = affine_relu_forward(out1_flat, W2, b2)

        # Layer 3: affine
        scores, cache3 = affine_forward(out2, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}
        # Compute the loss
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

        # Backward pass
        # Layer 3: affine
        dout2, grads['W3'], grads['b3'] = affine_backward(dscores, cache3)
        grads['W3'] += self.reg * W3

        # Layer 2: affine - relu
        dout1_flat, grads['W2'], grads['b2'] = affine_relu_backward(dout2, cache2)
        grads['W2'] += self.reg * W2

        # Reshape for conv layer
        dout1 = dout1_flat.reshape(out1.shape)

        # Layer 1: conv - relu - 2x2 max pool
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout1, cache1)
        grads['W1'] += self.reg * W1

        return loss, grads