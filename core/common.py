#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True, activate_type="mish"):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True:
            if activate_type == "mish":
                conv = conv * tf.nn.tanh(tf.nn.softplus(conv))
            elif activate_type == "leaky":
                conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output


def se_resnet_block(input_data, input_channel, filter_num1, filter_num2, ratio, trainable, name):

    res_short_cut = input_data

    with tf.variable_scope(name):

        # Convolution
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        # Squeeze-and-Excitation
        se_short_cut = input_data
        squeeze = tf.reduce_mean(input_data, axis=[1,2])
        excitation = tf.layers.dense(inputs=squeeze, use_bias=False, units=input_channel//ratio)
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(inputs=excitation, use_bias=False, units=input_channel)
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,input_channel])
        scale = se_short_cut * excitation

        se_resnet_output = scale + res_short_cut

    return se_resnet_output