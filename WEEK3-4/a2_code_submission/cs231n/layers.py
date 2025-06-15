from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    x_flat = x.reshape(N, -1)  # (N, D)
    out = x_flat @ w + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    x_flat = x.reshape(N, -1)  # (N, D)
    dx = (dout @ w.T).reshape(x.shape)  # (N, D) -> (N, d_1, ..., d_k)
    dw = x_flat.T @ dout  # (D, N) @ (N, M) -> (D, M)
    db = np.sum(dout, axis=0)  # (N, M) -> (M,)
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x = cache
    dx = dout * (x > 0)
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0)  # (D,)
        sample_var = np.var(x, axis=0)  # (D,)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)  # (N, D)
        out = gamma * x_hat + beta  # (N, D)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, x_hat, sample_mean, sample_var, gamma, eps)
    elif mode == "test":
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        cache = None
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, x_hat, sample_mean, sample_var, gamma, eps = cache
    N, D = x.shape
    dgamma = np.sum(dout * x_hat, axis=0)  # (D,)
    dbeta = np.sum(dout, axis=0)  # (D,)
    dx_hat = dout * gamma  # (N, D)
    std_inv = 1.0 / np.sqrt(sample_var + eps)
    dx = dx_hat * std_inv  # (N, D)
    dvar = np.sum(dx_hat * (x - sample_mean) * -0.5 * (sample_var + eps)**(-1.5), axis=0)  # (D,)
    dmean = np.sum(dx_hat * -std_inv, axis=0)  # (D,)
    dx += (dvar * 2 * (x - sample_mean) / N + dmean / N)  # (N, D)
    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    x, x_hat, sample_mean, sample_var, gamma, eps = cache
    N = x.shape[0]
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = (gamma / np.sqrt(sample_var + eps)) * (
        dout - np.mean(dout, axis=0) - x_hat * np.mean(dout * x_hat, axis=0)
    ) / N
    return dx, dgamma, dbeta

def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    N, D = x.shape
    mean = np.mean(x, axis=1, keepdims=True)  # (N, 1)
    var = np.var(x, axis=1, keepdims=True)  # (N, 1)
    x_hat = (x - mean) / np.sqrt(var + eps)  # (N, D)
    out = gamma * x_hat + beta  # (N, D)
    cache = (x, x_hat, mean, var, gamma, eps)
    return out, cache

def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, x_hat, mean, var, gamma, eps = cache
    N, D = x.shape
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = (gamma / np.sqrt(var + eps)) * (
        dout - np.mean(dout, axis=1, keepdims=True) - 
        x_hat * np.mean(dout * x_hat, axis=1, keepdims=True)
    ) / D
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p  # Inverted dropout
        out = x * mask
    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modify the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    x_slice = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    x_slice = x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
                    dw[f] += dout[n, f, i, j] * x_slice
                    dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += dout[n, f, i, j] * w[f]
                    db[f] += dout[n, f, i, j]
    
    dx = dx_padded[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_padded
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    x_slice = x[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width]
                    out[n, c, i, j] = np.max(x_slice)
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    x_slice = x[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width]
                    mask = x_slice == np.max(x_slice)
                    dx[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width] += dout[n, c, i, j] * mask
    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, C, H, W = x.shape
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
    out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # (N, C, H, W)
    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # (N, C, H, W)
    return dx, dgamma, dbeta

def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Forward pass for spatial group normalization.
    
    Inputs:
    - x: Input data, shape (N, C, H, W)
    - gamma: Scale parameter, shape (1, C, 1, 1)
    - beta: Shift parameter, shape (1, C, 1, 1)
    - G: Number of groups
    - gn_param: Dictionary with optional parameters (e.g., eps)
    
    Returns:
    - out: Normalized output, shape (N, C, H, W)
    - cache: Tuple for backward pass (x, x_hat, gamma, G, mu, var, eps)
    """
    eps = gn_param.get('eps', 1e-5)
    N, C, H, W = x.shape
    
    # Step 1: Reshape to group channels: (N, G, C//G, H, W)
    x_reshaped = x.reshape(N, G, C//G, H, W)
    
    # Step 2: Compute mean and variance over (C//G, H, W) for each (N, G)
    mu = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)  # (N, G, 1, 1, 1)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)  # (N, G, 1, 1, 1)
    
    # Step 3: Normalize
    x_hat_reshaped = (x_reshaped - mu) / np.sqrt(var + eps)  # (N, G, C//G, H, W)
    x_hat = x_hat_reshaped.reshape(N, C, H, W)  # (N, C, H, W)
    
    # Step 4: Scale and shift
    out = gamma * x_hat + beta  # (N, C, H, W)
    
    # Step 5: Cache for backward pass
    cache = (x, x_hat, gamma, G, mu, var, eps)
    return out, cache

def spatial_groupnorm_backward(dout, cache):
    """
    Backward pass for spatial group normalization.
    
    Inputs:
    - dout: Upstream derivatives, shape (N, C, H, W)
    - cache: A tuple of (x, x_hat, gamma, G, mu, var, eps) from the forward pass
    
    Returns:
    - dx: Gradient w.r.t. x, shape (N, C, H, W)
    - dgamma: Gradient w.r.t. gamma, shape (1, C, 1, 1)
    - dbeta: Gradient w.r.t. beta, shape (1, C, 1, 1)
    """
    x, x_hat, gamma, G, mu, var, eps = cache
    N, C, H, W = dout.shape
    
    # Step 1: Compute dgamma and dbeta by summing over (N, H, W) for each channel
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
    
    # Step 2: Reshape data to group channels
    # Each group has C//G channels, so reshape to (N, G, C//G, H, W)
    dout_reshaped = dout.reshape(N, G, C//G, H, W)
    x_reshaped = x.reshape(N, G, C//G, H, W)
    x_hat_reshaped = x_hat.reshape(N, G, C//G, H, W)
    gamma_reshaped = gamma.reshape(1, G, C//G, 1, 1)  # (1, G, C//G, 1, 1)
    
    # Step 3: Compute dx_hat (gradient w.r.t. normalized x)
    dx_hat = dout_reshaped * gamma_reshaped  # (N, G, C//G, H, W)
    
    # Step 4: Compute dx using the chain rule for group normalization
    # For group norm, dx = (gamma / sqrt(var + eps)) * (dout - mean(dout) - x_hat * mean(dout * x_hat))
    group_size = (C//G) * H * W
    mu_reshaped = mu.reshape(N, G, 1, 1, 1)  # (N, G, 1, 1, 1)
    var_reshaped = var.reshape(N, G, 1, 1, 1)  # (N, G, 1, 1, 1)
    
    # Compute terms for dx
    dx_hat_reshaped = dx_hat.reshape(N, G, C//G, H, W)
    dx = (gamma_reshaped / np.sqrt(var_reshaped + eps)) * (
        dx_hat_reshaped
        - np.mean(dx_hat_reshaped, axis=(2, 3, 4), keepdims=True)
        - x_hat_reshaped * np.mean(dx_hat_reshaped * x_hat_reshaped, axis=(2, 3, 4), keepdims=True)
    )
    
    # Step 5: Reshape dx back to (N, C, H, W)
    dx = dx.reshape(N, C, H, W)
    
    return dx, dgamma, dbeta

def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    N, C = x.shape
    correct_class_scores = x[np.arange(N), y]  # (N,)
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)  # (N, C)
    margins[np.arange(N), y] = 0  # Exclude correct class
    loss = np.sum(margins) / N
    dx = (margins > 0).astype(float)  # (N, C)
    dx[np.arange(N), y] = -np.sum(dx, axis=1)  # Correct class gradient
    dx /= N
    return loss, dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    N, C = x.shape
    scores = x - np.max(x, axis=1, keepdims=True)  # Numerical stability
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    probs = np.clip(probs, 1e-10, 1.0)  # Clip for stability
    correct_class_probs = probs[np.arange(N), y]
    loss = -np.sum(np.log(correct_class_probs)) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx