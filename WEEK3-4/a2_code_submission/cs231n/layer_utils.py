from .layers import *
from .fast_layers import *  # Add this import to ensure fast implementations are available

def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer for affine -> batchnorm -> ReLU
    """
    a, fc_cache = affine_forward(x, w, b)
    an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for affine -> batchnorm -> ReLU
    """
    fc_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dan, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
    """
    Convenience layer for affine -> layernorm -> ReLU
    """
    a, fc_cache = affine_forward(x, w, b)
    an, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
    out, relu_cache = relu_forward(an)
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache

def affine_ln_relu_backward(dout, cache):
    """
    Backward pass for affine -> layernorm -> ReLU
    """
    fc_cache, ln_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = layernorm_backward(dan, ln_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)  # Use fast implementation
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)  # Use fast implementation
    return dx, dw, db

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """
    Convenience layer that performs a convolution, batchnorm, and ReLU
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)  # Use fast implementation
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

def conv_bn_relu_backward(dout, cache):
    """
    Backward pass for the conv-bn-relu convenience layer
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)  # Use fast implementation
    return dx, dw, db, dgamma, dbeta

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)  # Use fast implementation
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)  # Use fast implementation
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)  # Use fast implementation
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)  # Use fast implementation
    return dx, dw, db