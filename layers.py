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

class ReshapeLayer(Layer):
  """
  A layer that reshapes previous layer outputs.
  """

  def __init__(self, input, shape, name="reshape"):
    super(ReshapeLayer, self).__init__(input, name)

    if shape == "flat":
      last_shape = self.input.output.get_shape().as_list()
      size = 1
      for v in last_shape[1:]:
        size *= v
      shape = [-1, size]
    else:
      shape = [-1] + shape

    self.output = tf.reshape(self.input.output, shape, self.name)

class MatrixLayer(Layer):
  """
  A layer that multiplies previous layer outputs to a nicely initialized matrix.
  """

  def __init__(self, input, hidden_units, name="matrix"):
    super(MatrixLayer, self).__init__(input, name)
    num_inputs = self.input.output.get_shape().as_list()[1]
    weights = tf.Variable(
      tf.truncated_normal([num_inputs, hidden_units], stddev=1.0 / math.sqrt(float(num_inputs))),
      name='weights'
    )
    self.output = tf.matmul(self.input.output, weights, name=self.name)

class Conv2dApplyLayer(Layer):
  """
  A layer applies two-dimentional convolution.
  """

  def __init__(self, input, filter_size, num_filters, strides=[1, 1, 1, 1], padding="SAME", name="conv2d_apply"):
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

class ConcatLayer(Layer):
  """
  A layer that concats previous layers outputs.
  """

  def __init__(self, inputs, dim=1, name="concat"):
    super(ConcatLayer, self).__init__(inputs, name)
    self.output = tf.concat(dim, map(lambda x: x.output, self.input), name=self.name)

  def get_dict(self, **args):
    d = {}
    for i in self.input:
      self._join_dicts(d, i.get_dict(**args))
    return d

  def _join_dicts(self, d1, d2):
    if d2 is None:
      return d1

    for k in d2:
        d1[k] = d2[k]
    return d1

class CompositeLayer(Layer):
  def __init__(self, input, name="composite"):
    super(CompositeLayer, self).__init__(input, name)
    self.network = None

  def get_dict(self, **args):
    return self.network.get_dict(**args)

class DenseLayer(CompositeLayer):
  """
  A fully connected layer.
  """

  def __init__(self, input, hidden_units, fun=tf.nn.relu, dropout=None, name="dense"):
    super(DenseLayer, self).__init__(input, name)

    network = ReshapeLayer(input, "flat")
    network = MatrixLayer(network, hidden_units)
    network = BiasLayer(network)
    network = NonlinearityLayer(network, fun)
    if dropout is not None:
      network = DropoutLayer(network, dropout)
    self.network = network
    self.output = self.network.output

class Conv2dLayer(CompositeLayer):
  """
  A layer applies two-dimentional convolution with bias, nonlinearity, max-pooling and optional dropout.
  """

  def __init__(self, input, num_filters, filter_size, pool_size=None, pool_stride=None, fun=tf.nn.relu, dropout=None, name="conv2d"):
    super(Conv2dLayer, self).__init__(input, name)

    network = Conv2dApplyLayer(input, filter_size, num_filters)
    network = BiasLayer(network)
    network = NonlinearityLayer(network, fun)
    if pool_size is None or pool_stride is None:
      assert pool_size is None and pool_stride is None
    else:
      network = MaxPoolingLayer(network, [1, pool_size, pool_size, 1], [1, pool_stride, pool_stride, 1])
    if dropout is not None:
      network = DropoutLayer(network, dropout)
    self.network = network
    self.output = self.network.output
