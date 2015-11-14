from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


class Layer(object):
  """
  Layer interface.
  A Layer has to have name: string, input: Layer and output tensorflow.Tensor.
  """

  def __init__(self, input, name="input"):
    self.name = name
    self.input = input
    self.output = None

  def get_dict(self, **args):
    if self.input is None:
      return {}
    else:
      return self.input.get_dict(**args)

class InputLayer(Layer):
  """
  An input Layer, just proxies its input to the output.
  """

  def __init__(self, input, name="input"):
    super(InputLayer, self).__init__(None, name)
    self.output = input

  def get_dict(self, **args):
    return {}

class DropoutLayer(Layer):
  """
  A layer that zeroes some values from a previous layer.
  """

  def __init__(self, input, p=0.5, name="dropout"):
    super(DropoutLayer, self).__init__(input, name)
    self.prob = p
    self.prob_var = tf.placeholder("float", shape=(), name="keep_prob")
    self.output = tf.nn.dropout(self.input.output, self.prob_var, name=self.name)

  def get_dict(self, **args):
    d = self.input.get_dict(**args)
    if d is None:
      d = {}

    if not args.get("dropout", True):
      d[self.prob_var] = 1.0
    else:
      d[self.prob_var] = self.prob
    return d

class NonlinearityLayer(Layer):
  """
  A layer that applies a function elementwise to the previous layer.
  """

  def __init__(self, input, fun, name="nonlinearity"):
    super(NonlinearityLayer, self).__init__(input, name)
    self.output = fun(self.input.output)

class BiasLayer(Layer):
  """
  A layer that adds a nicely initialized bias vector to the previous layer.
  """

  def __init__(self, input, name="bias"):
    super(BiasLayer, self).__init__(input, name)

    last_shape = self.input.output.get_shape().as_list()
    biases = tf.Variable(tf.constant(0.1, shape=[last_shape[-1]]), name="biases")
    self.output = tf.add(self.input.output, biases, name=self.name)

class MatrixLayer(Layer):
  """
  A layer that multiplies previous layer outputs to a nicely initialized matrix.
  """

  def __init__(self, input, hidden_units, name="matrix"):
    super(MatrixLayer, self).__init__(input, name)

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

class Conv2dApplyLayer(Layer):
  """
  A layer applies two-dimentional convolution.
  """

  def __init__(self, input, filter_size, num_filters, strides=[1, 1, 1, 1], padding="SAME", name="conv2d"):
    super(Conv2dApplyLayer, self).__init__(input, name)
    if type(filter_size) is not tuple:
      filter_size = (filter_size, filter_size)

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
  """
  A layer applies max-pooling.
  """

  def __init__(self, input, ksize, strides, padding="SAME", name="maxpool"):
    super(MaxPoolingLayer, self).__init__(input, name)
    self.output = tf.nn.max_pool(self.input.output, ksize, strides, padding, name=self.name)

class DenseLayer(Layer):
  """
  A fully connected layer.
  """

  def __init__(self, input, fun, hidden_units, name="dense"):
    super(DenseLayer, self).__init__(input, name)
    self.output = NonlinearityLayer(BiasLayer(MatrixLayer(input, hidden_units)), fun, name="dense")
