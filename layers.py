from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

class Layer(object):
    def __init__(self, name="input"):
        self.name = name

class InputLayer(Layer):
    def __init__(self, input, name="input"):
        self.name = name
        self.input = None
        self.output = input

class DropoutLayer(Layer):
    def __init__(self, input, p=0.5, name="dropout"):
        self.name = name
        self.input = input
        self.output = tf.nn.dropout(self.input.output, p, name=self.name)

class DropoutLayer(Layer):
    def __init__(self, input, p=0.5, name="dropout"):
        self.name = name
        self.input = input
        self.output = tf.nn.dropout(self.input.output, p, name=self.name)

class NonlinearityLayer(Layer):
    def __init__(self, input, fun, name="nonlinearity"):
        self.name = name
        self.input = input
        self.output = fun(self.input.output)

class BiasLayer(Layer):
    def __init__(self, input, name="bias"):
        self.name = name
        self.input = input

        last_shape = self.input.output.get_shape().as_list()
        biases = tf.Variable(tf.constant(0.1, shape=[last_shape[-1]]), name="biases")
        self.output = tf.add(self.input.output, biases, name=self.name)

class DenseLayer(Layer):
    def __init__(self, input, hidden_units, name="dense"):
        self.name = name
        self.input = input

        last_shape = self.input.output.get_shape().as_list()
        size = 1
        for v in last_shape[1:]:
            size *= v

        flat = tf.reshape(self.input.output, [-1, size], name="flat")
        last_shape = flat.get_shape().as_list()

        weights = tf.Variable(
            tf.truncated_normal([last_shape[1], hidden_units], stddev=1.0 / math.sqrt(float(size))),
            name='weights'
        )
        self.output = tf.matmul(flat, weights, name=self.name)

class Conv2dLayer(Layer):
    def __init__(self, input, filter_size, num_filters, strides=[1, 1, 1, 1], padding="SAME", name="conv2d"):
        if type(filter_size) is not tuple:
            filter_size = (filter_size, filter_size)

        self.name = name
        self.input = input

        last_shape = self.input.output.get_shape().as_list()
        size = 1
        for v in last_shape[1:]:
            size *= v

        channels = last_shape[3]
        weights = tf.Variable(
            tf.truncated_normal([filter_size[0], filter_size[1], channels, num_filters], stddev=1.0 / math.sqrt(float(size))),
            name='weights'
        )
        self.output = tf.nn.conv2d(self.input.output, weights, strides=strides, padding=padding, name=self.name)

class MaxPoolingLayer(Layer):
    def __init__(self, input, ksize, strides, padding="SAME", name="maxpool"):
        self.name = name
        self.input = input
        self.output = tf.nn.max_pool(self.input.output, ksize, strides, padding, name=self.name)
