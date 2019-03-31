import ops
import numpy as np
import tensorflow as tf

class LeNet:
    def __init__(self, height, width, channels, learning_rate):
        self.height = height
        self.width = width
        self.channels = channels
        self.learning_rate = learning_rate
        self.network = None
        self.loss = None
        self.optimizer = None

    def build_model(self, inputs, labels, is_training=False):
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
            self.optimizer = ops.optimize(self.loss, self.learning_rate, scope='update')


class AlexNet:
    def __init__(self, height, width, channels, learning_rate):
        self.height = height
        self.width = width
        self.channels = channels
        self.learning_rate = learning_rate
        self.network = None
        self.loss = None
        self.optimizer = None
        self.image_size = 224

    def build_model(self, inputs, labels, is_training):
        # pad inputs to size 224x224x3 - NOTE: may change to bilinear upsampling
        pad = int((self.image_size - self.height) / 2)
        inputs = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

        # convolution with 11x11 kernel and stride 4 (new size: 55x55x96)
        self.network = ops.convolution(inputs, self.channels, 96, 11, 96, stride=4,
                                       padding='VALID', is_training=is_training, scope='conv1')

        # pooling with 3x3 kernel and stride 2 (new size: 27x27x96)
        self.network = ops.pooling(self.network, k_size=3, scope='pool1')

        # convolution with 5x5 kernel and stride 1 (new size: 27x27x256)
        self.network = ops.convolution(self.network, 96, 256, 5, 256,
                                       is_training=is_training, scope='conv2')

        # pooling with 3x3 kernel and stride 2 (new size: 13x13x256)
        self.network = ops.pooling(self.network, k_size=3, scope='pool2')

        # convolution with 3x3 kernel and stride 1 (new size: 13x13x384)
        self.network = ops.convolution(self.network, 256, 384, 3, 384, batch_norm=False,
                                       is_training=is_training, scope='conv3')

        # convolution with 3x3 kernel and stride 1 (new size: 13x13x384)
        self.network = ops.convolution(self.network, 384, 384, 3, 384, batch_norm=False,
                                       is_training=is_training, scope='conv4')

        # convolution with 3x3 kernel and stride 1 (new size: 13x13x256)
        self.network = ops.convolution(self.network, 384, 256, 3, 256, batch_norm=False,
                                       is_training=is_training, scope='conv5')

        # pooling with 3x3 kernel and stride 2 (new size: 6x6x256)
        self.network = ops.pooling(self.network, k_size=3, scope='pool3')

        # flatten (new size: 9216)
        self.network = ops.flatten(self.network, scope='flatten')

        # fully connected layer (new size: 4096)
        self.network = ops.dense(self.network, 9216, 4096, dropout=True, dropout_rate=0.2,
                                 is_training=is_training, scope='fc1')

        # fully connected layer (new size: 1024) -- Original Paper Size: 4096 (for ImageNet)
        self.network = ops.dense(self.network, 4096, 1024, dropout=True, dropout_rate=0.2,
                                 is_training=is_training, scope='fc2')

        # output layer (new size: 10) -- Original Paper Size: 1000 (for ImageNet)
        self.network = ops.dense(self.network, 1024, 10, activation=None,
                                 is_training=is_training, scope='fc3')

        self.loss = ops.loss(self.network, labels, scope='loss')

        if is_training:
            self.optimizer = ops.optimize(self.loss, self.learning_rate, scope='update')


class VGG19:
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


class ResNet50:
    def __init__(self):
        pass


class GoogLeNet:
    def __init__(self):
        pass
