#! /usr/bin/env python
# coding=utf-8


import core.common as common
import tensorflow as tf



def cspdarknet(input_data, trainable):

    with tf.variable_scope('cspdarknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        route = input_data
        route = common.convolutional(route, filters_shape=(1, 1, 64, 64), trainable=trainable, name='conv2')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 64, 64), trainable=trainable, name='conv3')

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(1, 1, 64, 64), trainable=trainable, name='conv6')
        input_data = tf.concat([input_data, route], axis=-1)
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 128, 64), trainable=trainable, name='conv7')

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 128),
                                          trainable=trainable, name='conv8', downsample=True)

        route = input_data
        route = common.convolutional(route, filters_shape=(1, 1, 128, 64), trainable=trainable, name='conv9')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 128, 64), trainable=trainable, name='conv10')

        for i in range(2):
            input_data = common.residual_block(input_data,  64,  64, 64, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(1, 1, 64, 64), trainable=trainable, name='conv15')
        input_data = tf.concat([input_data, route], axis=-1)
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 128, 128), trainable=trainable, name='conv16')

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv17', downsample=True)

        route = input_data
        route = common.convolutional(route, filters_shape=(1, 1, 256, 128), trainable=trainable, name='conv18')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 256, 128), trainable=trainable, name='conv19')

        for i in range(8):
            input_data = common.residual_block(input_data, 128, 128, 128, trainable=trainable, name='residual%d' %(i+3))

        input_data = common.convolutional(input_data, filters_shape=(1, 1, 128, 128), trainable=trainable, name='conv36')
        input_data = tf.concat([input_data, route], axis=-1)
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 256, 256), trainable=trainable, name='conv37')

        route_1 = input_data

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv38', downsample=True)

        route = input_data
        route = common.convolutional(route, filters_shape=(1, 1, 512, 256), trainable=trainable, name='conv39')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 512, 256), trainable=trainable, name='conv40')

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 256, 256, trainable=trainable, name='residual%d' %(i+11))

        input_data = common.convolutional(input_data, filters_shape=(1, 1, 256, 256), trainable=trainable, name='conv57')
        input_data = tf.concat([input_data, route], axis=-1)
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 512, 512), trainable=trainable, name='conv58')

        route_2 = input_data

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv59', downsample=True)

        route = input_data
        route = common.convolutional(route, filters_shape=(1, 1, 1024, 512), trainable=trainable, name='conv60')
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 1024, 512), trainable=trainable, name='conv61')

        for i in range(4):
            input_data = common.residual_block(input_data, 512, 512, 512, trainable=trainable, name='residual%d' %(i+19))

        input_data = common.convolutional(input_data, filters_shape=(1, 1, 512, 512), trainable=trainable, name='conv70')
        input_data = tf.concat([input_data, route], axis=-1)
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 1024, 1024), trainable=trainable, name='conv71')


        input_data = common.convolutional(input_data, filters_shape=(1, 1, 1024, 512), trainable=trainable, name='conv72', activate_type="leaky")
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024), trainable=trainable, name='conv73', activate_type="leaky")
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 1024, 512), trainable=trainable, name='conv74', activate_type="leaky")

        input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), \
                                tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1), \
                                tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)

        input_data = common.convolutional(input_data, filters_shape=(1, 1, 2048, 512), trainable=trainable, name='conv75', activate_type="leaky")
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024), trainable=trainable, name='conv76', activate_type="leaky")
        input_data = common.convolutional(input_data, filters_shape=(1, 1, 1024, 512), trainable=trainable, name='conv77', activate_type="leaky")

        return route_1, route_2, input_data


