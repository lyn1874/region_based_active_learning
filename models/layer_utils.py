"""
Created on Tue Nov 21 10:09:39 2017
This file is utilized to denote different layers, there are conv_layer, conv_layer_enc, max_pool, up_sampling
@author: s161488
"""
import numpy as np
import tensorflow as tf
import math


def conv_layer_renew(bottom, name, shape, training_state, strides=(1, 1), activation_func=tf.nn.relu, padding='same',
                     dilation_rate=(1, 1), bias_state=True):
    """
    This function is a simplified version of convolutional layer
    bottom: Input, dtype tf.float32, shape [Batch_Size, Height, Width, Num_Input_Channel]
    shape: shape[0:2] the filter size
    shape[-1]: output channel size
    load_pretrain: a boolean variable. If True, then load the parameter from DeepLab If False,
        initialize with random truncated normal distribution
    dilation_rate: default to be [1,1]
    training_state: Since we first fix the "downsampling" path, only train the gaussin filter.
        Then after it's kind of fixed, we retrain the whole network.
    so it's a boolean variable. 
    activation_func: it could be relu, or None
    """
    with tf.variable_scope(name) as scope:
        w_init = tf.truncated_normal_initializer(stddev=1)
        b_init = tf.constant_initializer(0.0)
        output = tf.cond(training_state,
                         lambda: tf.layers.conv2d(bottom, filters=shape[1], kernel_size=shape[0], strides=strides,
                                                  padding=padding,
                                                  dilation_rate=dilation_rate, activation=activation_func,
                                                  use_bias=bias_state, kernel_initializer=w_init,
                                                  bias_initializer=b_init, trainable=True, name=scope.name),
                         lambda: tf.layers.conv2d(bottom, filters=shape[1], kernel_size=shape[0], strides=strides,
                                                  padding=padding,
                                                  dilation_rate=dilation_rate, activation=activation_func,
                                                  use_bias=bias_state, kernel_initializer=w_init,
                                                  bias_initializer=b_init, trainable=False, name=scope.name,
                                                  reuse=True))

    return output


def get_deconv_layer_weight(shape):
    """
    Args:
        shape: 4d shape. [kernel_size, kernel_size, output_channel, input_channel]
        
    Returns:
        the initialized deconvolution filter which performs as a bilinear upsampling.
    
    Source:
        https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py#L245
    """
    width = shape[0]
    height = shape[0]
    f = math.ceil(width / 2.0)
    c = (2.0 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(width):
        for y in range(height):
            bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return init


def deconv_layer_renew(bottom, filter_shape, output_channel, name, strides, training_state, padding='same',
                       bilinear_initialization=False):
    with tf.variable_scope(name) as scope:
        if bilinear_initialization is True:
            w_shape = [filter_shape, filter_shape, output_channel,
                       bottom.shape.as_list()[-1]]  # change to be bilinear upsampling!
            w_init = get_deconv_layer_weight(w_shape)
            print("The initialization of the deconvolution kernel is bilinear")
        else:
            w_init = tf.truncated_normal_initializer(stddev=0.1)
            b_init = tf.constant_initializer(0.0)
        output = tf.cond(training_state,
                         lambda: tf.layers.conv2d_transpose(bottom, output_channel, filter_shape, strides=strides,
                                                            padding=padding, activation=tf.nn.relu, use_bias=True,
                                                            kernel_initializer=w_init, bias_initializer=b_init,
                                                            trainable=True, name=scope.name),
                         lambda: tf.layers.conv2d_transpose(bottom, output_channel, filter_shape, strides=strides,
                                                            padding=padding, activation=tf.nn.relu, use_bias=True,
                                                            kernel_initializer=w_init, bias_initializer=b_init,
                                                            trainable=False, name=scope.name, reuse=True))
    return output
