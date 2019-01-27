import ops
import numpy as np
import tensorflow as tf

class LeNet:
    def __init__(self, height, width, channels):
        self.height = height
        self.width = width
        self.channels = channels
        self.learning_rate = 1e-4
        self.network = None
        self.loss = None
        self.optimizer = None

    def network(self, inputs, labels, is_training):
        self.network = ops.convolution(inputs, self.channels, 50, 5, 50,
                                       is_training=is_training, scope='conv1')

        self.network = ops.pooling(self.network, scope='pool1')

        self.network = ops.convolution(self.network, 50, 20, 5, 20,
                                       is_training=is_training, scope='conv2')

        self.network = ops.pooling(self.network, scope='pool2')

        self.network = ops.flatten(self.network, scope='flatten')

        self.network = ops.dense(self.network, self.network.get_shape().as_list()[1],
                                 200, scope='fc1')

        self.network = ops.dense(self.network, 200, 50, scope='fc2')

        self.network = ops.dense(self.network, 50, 10, activation=None, scope='fc3')

        self.loss = ops.loss(self.network, labels, scope='loss')

        if is_training:
            self.optimizer = ops.optimize(self.loss, self.learning_rate)


class AlexNet:
    def __init__(self, height, width, channels):
        self.height = height
        self.width = width
        self.channels = channels
        self.learning_rate = 1e-4
        self.network = None
        self.loss = None
        self.optimizer = None

    def network(self, inputs, labels, is_training):
        pass


class VGG19:
    def __init__(self, height, width, channels):
        self.height = height
        self.width = width
        self.channels = channels
        self.learning_rate = 1e-4
        self.network = None
        self.loss = None
        self.optimizer = None

    def network(self, inputs, labels is_training):
        self.network = 


class ResNet50:
    def __init__(self):
        pass


class GoogLeNet:
    def __init__(self):
        pass
