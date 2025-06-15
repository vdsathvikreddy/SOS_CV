from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *

class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deterministic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize parameters
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params[f'W{i+1}'] = np.random.normal(0.0, weight_scale, (dims[i], dims[i+1]))
            self.params[f'b{i+1}'] = np.zeros(dims[i+1])
            if self.normalization in ['batchnorm', 'layernorm'] and i < self.num_layers - 1:
                self.params[f'gamma{i+1}'] = np.ones(dims[i+1])
                self.params[f'beta{i+1}'] = np.zeros(dims[i+1])

        # Dropout parameters
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # Batchnorm/layernorm parameters
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None

        # Forward pass
        X = X.reshape(X.shape[0], -1)  # (N, input_dim)
        caches = []
        h = X
        for i in range(self.num_layers - 1):
            W, b = self.params[f'W{i+1}'], self.params[f'b{i+1}']
            if self.normalization == 'batchnorm':
                gamma, beta = self.params[f'gamma{i+1}'], self.params[f'beta{i+1}']
                h, cache = affine_bn_relu_forward(h, W, b, gamma, beta, self.bn_params[i])
            elif self.normalization == 'layernorm':
                gamma, beta = self.params[f'gamma{i+1}'], self.params[f'beta{i+1}']
                h, cache = affine_ln_relu_forward(h, W, b, gamma, beta, self.bn_params[i])
            else:
                h, cache = affine_relu_forward(h, W, b)
            if self.use_dropout:
                h, cache_dropout = dropout_forward(h, self.dropout_param)
                cache = (cache, cache_dropout)
            else:
                cache = (cache, None)
            caches.append(cache)

        # Final layer: affine only
        W, b = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        scores, cache_affine = affine_forward(h, W, b)
        caches.append((cache_affine, None))  # No dropout/norm

        # Test mode: return scores
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        # Backward pass
        # Softmax loss
        loss, dscores = softmax_loss(scores, y)

        # L2 regularization
        for i in range(self.num_layers):
            W = self.params[f'W{i+1}']
            loss += 0.5 * self.reg * np.sum(W * W)

        # Backpropagation
        dh = dscores
        for i in range(self.num_layers - 1, -1, -1):
            cache, cache_dropout = caches[i]
            if i == self.num_layers - 1:
                dh, grads[f'W{i+1}'], grads[f'b{i+1}'] = affine_backward(dh, cache)
            else:
                if self.use_dropout:
                    dh = dropout_backward(dh, cache_dropout)
                if self.normalization == 'batchnorm':
                    dh, grads[f'W{i+1}'], grads[f'b{i+1}'], grads[f'gamma{i+1}'], grads[f'beta{i+1}'] = affine_bn_relu_backward(dh, cache)
                elif self.normalization == 'layernorm':
                    dh, grads[f'W{i+1}'], grads[f'b{i+1}'], grads[f'gamma{i+1}'], grads[f'beta{i+1}'] = affine_ln_relu_backward(dh, cache)
                else:
                    dh, grads[f'W{i+1}'], grads[f'b{i+1}'] = affine_relu_backward(dh, cache)
            grads[f'W{i+1}'] += self.reg * self.params[f'W{i+1}']

        return loss, grads