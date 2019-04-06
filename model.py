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

        # convolution with 3x3 kernel and stride 1 (new size: 224x224x64)
        self.network = ops.convolution(inputs, self.channels, 64, 3, 64,
                                       is_training=is_training, scope='conv1')

        # convolution with 3x3 kernel and stride 1 (new size: 224x224x64)
        self.network = ops.convolution(self.network, 64, 64, 3, 64,
                                       is_training=is_training, scope='conv2')

        # pooling with 2x2 kernel and stride 2 (new size: 112x112x64)
        self.network = ops.pooling(self.network, scope='pool1')

        # convolution with 3x3 kernel and stride 1 (new size: 112x112x128)
        self.network = ops.convolution(self.network, 64, 128, 3, 128,
                                       is_training=is_training, scope='conv3')

        # convolution with 3x3 kernel and stride 1 (new size: 112x112x128)
        self.network = ops.convolution(self.network, 128, 128, 3, 128,
                                       is_training=is_training, scope='conv4')

        # pooling with 2x2 kernel and stride 2 (new size: 56x56x128)
        self.network = ops.pooling(self.network, scope='pool2')

        # convolution with 3x3 kernel and stride 1 (new size: 56x56x256)
        self.network = ops.convolution(self.network, 128, 256, 3, 256,
                                       is_training=is_training, scope='conv5')

        # 3 convolutions with 3x3 kernel and stride 1 (new size: 56x56x256)
        for idx in range(6, 9):
            self.network = ops.convolution(self.network, 256, 256, 3, 256,
                                          is_training=is_training, scope='conv' + str(idx))

        # pooling with 2x2 kernel and stride 2 (new size: 28x28x256)
        self.network = ops.pooling(self.network, scope='pool3')

        # convolution with 3x3 kernel and stride 1 (new size: 28x28x512)
        self.network = ops.convolution(self.network, 256, 512, 3, 512,
                                       is_training=is_training, scope='conv9')

        # 3 convolutions with 3x3 kernel and stride 1 (new size: 28x28x512)
        for idx in range(10, 13):
            self.network = ops.convolution(self.network, 512, 512, 3, 512,
                                          is_training=is_training, scope='conv' + str(idx))

        # pooling with 2x2 kernel and stride 2 (new size: 14x14x512)
        self.network = ops.pooling(self.network, scope='pool4')

        # 4 convolutions with 3x3 kernel and stride 1 (new size: 14x14x512)
        for idx in range(13, 17):
            self.network = ops.convolution(self.network, 512, 512, 3, 512,
                                          is_training=is_training, scope='conv' + str(idx))

        # pooling with 2x2 kernel and stride 2 (new size: 7x7x512)
        self.network = ops.pooling(self.network, scope='pool5')

        # flatten (new size: 25088)
        self.network = ops.flatten(self.network, scope='flatten')

        # fully connected layer (new size: 4096)
        self.network = ops.dense(self.network, 25088, 4096, dropout=True, dropout_rate=0.2,
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


class ResNet50:
    def __init__(self):
        pass


class GoogLeNet:
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
        pad = int((self.image_size - self.height) / 2)
        inputs = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

        # convolution with 7x7 kernel and stride 2 (new size: 112x112x64)
        self.network = ops.convolution(inputs, self.channels, 64, 7, 64, stride=2,
                                       padding='VALID', is_training=is_training, scope='conv1')

        # pooling with 3x3 kernel and stride 2 (new size: 56x56x64)
        self.network = ops.pooling(self.network, k_size=3, scope='pool1')

        # convolution with 1x1 kernel and stride 1 (new size: 56x56x192)
        self.network = ops.convolution(self.network, 64, 192, 1, 192, batch_norm=False,
                                       is_training=is_training, scope='conv2')

        # convolution with 3x3 kernel and stride 1 (new size: 56x56x192)
        self.network = ops.convolution(self.network, 192, 192, 3, 192,
                                       is_training=is_training, scope='conv3')

        # pooling with 3x3 kernel and stride 2 (new size: 28x28x192)
        self.network = ops.pooling(self.network, k_size=3, scope='pool2')

        # inception module (3a)
        self.network = self.inception_module(self.network,
                                             [[64, 96, 16], [128, 32, 32]],
                                             scope='incept1')

        # inception module (3b)
        self.network = self.inception_module(self.network,
                                             [[128, 128, 32], [192, 96, 64]],
                                             final_pool=True,
                                             scope='incept'+str(i))

        # inception module (4a)
        self.network = self.inception_module(self.network,
                                             [[192, 96, 16], [208, 48, 64]],
                                             scope='incept'+str(i))

        # auxiliary classifier
        if is_training:
            aux_loss1 = self.aux_classifier(self.network, labels, 512, is_training,
                                            scope='auxclass1')

        # inception module (4b)
        self.network = self.inception_module(self.network,
                                             [[160, 112, 24], [224, 64, 64]],
                                             scope='incept'+str(i))

        # inception module (4c)
        self.network = self.inception_module(self.network,
                                             [[128, 128, 24], [256, 64, 64]],
                                             scope='incept'+str(i))

        # inception module (4d)
        self.network = self.inception_module(self.network,
                                             [[112, 144, 32], [288, 64, 64]],
                                             scope='incept'+str(i))

        # auxiliary classifier
        if is_training:
            aux_loss2 = self.aux_classifier(self.network, labels, 528, is_training,
                                            scope='auxclass2')

        # inception module (4e)
        self.network = self.inception_module(self.network,
                                             [[256, 160, 32], [320, 128, 128]],
                                             final_pool=True,
                                             scope='incept'+str(i))

        # inception module (5a)
        self.network = self.inception_module(self.network,
                                             [[256, 160, 32], [320, 128, 128]],
                                             scope='incept'+str(i))

        # inception module (5b)
        self.network = self.inception_module(self.network,
                                             [[384, 192, 48], [384, 128, 128]],
                                             scope='incept'+str(i))

        # pooling with 7x7 kernel and stride 1 (new size: 1x1x1024)
        with tf.variable_scope('final_pool', reuse=tf.AUTO_REUSE):
            self.network = tf.nn.avg_pool(self.network, 7, 1 'VALID', scope='pool')

        # flatten (new size: 1024)
        self.network = ops.flatten(self.network, scope='flatten')

        # fully connected layer (new size: 1024)
        self.network = ops.dense(self.network, 1024, 1024, dropout=True, dropout_rate=0.4,
                                 is_training=is_training, scope='fc1')

        # output layer (new size: 10) -- Original Paper Size: 1000 (for ImageNet)
        self.network = ops.dense(self.network, 1024, 10, activation=None,
                                 is_training=is_training, scope='fc2')

        loss = ops.loss(self.network, labels, scope='loss')

        if is_training: # if training use auxiliary classifiers as well
            self.loss = loss + aux_loss1 + aux_loss2
            self.optimizer = ops.optimize(self.loss, self.learning_rate, scope='update')
        else:
            self.loss = loss

    def inception_module(self, inputs, output_sizes, final_pool=False, scope=None):
        """
            Inception module allowing for deeper networks with fewer parameters.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            input_channels = tf.shape(inputs)[3]

            net1 = ops.convolution(inputs, input_channels, output_sizes[0, 0], 1,
                                   output_sizes[0, 0], is_training=is_training,
                                   scope='net1')

            net2 = ops.convolution(inputs, input_channels, output_sizes[0, 1], 1,
                                   output_sizes[0, 1], is_training=is_training,
                                   scope='net2a')
            net2 = ops.convolution(net2, output_sizes[0, 1], output_sizes[1, 0],
                                   3, output_sizes[1, 0], is_training=is_training,
                                   scope='net2b')

            net3 = ops.convolution(inputs, input_channels, output_sizes[0, 2], 1,
                                  output_sizes[0, 2], is_training=is_training,
                                  scope='net3a')
            net3 = ops.convolution(net3, output_sizes[0, 2], output_sizes[1, 1],
                                   5, output_sizes[1, 1], is_training=is_training,
                                   scope='net3b')

            net4 = ops.pooling(inputs, k_size=3, stride=1, padding='SAME', scope='pool')
            net4 = ops.convolution(net4, input_channels, output_sizes[1, 2], 1,
                                   output_sizes[1, 2], is_training=is_training,
                                   scope='net4')

            network = tf.concat([net1, net2, net3, net4], -1)

            if final_pool:
                network = ops.pooling(network, k_size=3, scope='outpool')

            return network

    def aux_classifier(self, inputs, labels, input_channels, is_training, scope=None):
        """
            Auxiliary Classifier used in Inception Module to help propagate
            gradients backward.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # pooling layer with 5x5 kernel and stride 3 (new size: 4x4xC)
            network = tf.nn.avg_pool(inputs, 5, 3, 'VALID', name='pool')

            # convolution with 1x1 kernel and stride 1 (new size: 4x4x128)
            network = ops.convolution(network, input_channels, 128, 1, 128, batch_norm=False,
                                      is_training=is_training, scope='auxconv')

            # flatten (new size: 2048)
            network = ops.flatten(network, scope='flatten')

            # fully connected layer (new size: 1024)
            network = ops.dense(network, 2048, 1024, dropout=True, dropout_rate=0.7,
                                is_training=is_training, scope='fc1')

            # output layer (new size: 10) -- Original Paper Size: 1000 (for ImageNet)
            network = ops.dense(network, 1024, 10, activation=None, is_training=is_training,
                                scope='fc2')

            # loss of auxiliary classifier
            loss = ops.loss(network, labels, scope='auxloss')

            return loss
