import numpy as np
import tensorflow as tf


def mean_tile(input_, n_output):
    input_ = tf.reduce_mean(input_, axis=1, keep_dims=True)
    input_data = tf.tile(input_, [1, n_output])
    return input_data


# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_output):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializer
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # only one encoder layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_output], initializer=w_init)
        b0 = tf.get_variable('b0', [n_output], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        z_I = tf.nn.elu(h0)

        # Two methods to calculate mean and std, the results may depends on the dataset, try yourself.
        # method 1: the calculated mean and std
        # mean, std = tf.nn.moments(z, axes=[1], keep_dims=True)
        # mean = tf.tile(mean, [1, n_output])
        # stddev = 1e-6 + tf.nn.softplus(std)
        # stddev = tf.tile(stddev, [1, n_output])

        # method 2: mean, std under a number of distributions
        mean = tf.layers.dense(z_I, units=n_output)  # presentation the z_I
        stddev = tf.layers.dense(z_I, units=n_output)  # presentation the z_I

    return mean, stddev, z_I


# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):
    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        # only one layer decoder
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_output], initializer=w_init)
        b0 = tf.get_variable('b0', [n_output], initializer=b_init)
        y = tf.matmul(z, w0) + b0
    return y


# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
    # encoding
    mu, sigma, z_I = gaussian_MLP_encoder(x_hat, dim_z)
    print 'mu shape', mu.shape, sigma.shape

    # sampling by re-parameterization technique
    z_s = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    print 'mu, sigma, z_s, mu[0], sigma[0], z_s[0]', mu, sigma, z_s, mu[0], sigma[0], z_s[0]
    # decoding
    y = bernoulli_MLP_decoder(z_s, n_hidden, dim_img, keep_prob)

    # second loss
    logvar_encoder = tf.log(1e-8 + tf.square(sigma))
    KL_divergence = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu, 2) - tf.exp(logvar_encoder),
                                         reduction_indices=1)

    loss_recog = tf.reduce_sum(tf.pow(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=x_hat), 2),
                               reduction_indices=1)

    loss = tf.reduce_mean(loss_recog + KL_divergence)
    # loss = tf.reduce_mean(loss_recog)  # No KL
    return y, z_s, loss, loss_recog, KL_divergence, z_I, mu, sigma


def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y


def maxminnorm(data):
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_norm