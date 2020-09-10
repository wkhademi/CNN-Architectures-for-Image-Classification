import numpy as np
import tensorflow as tf

def convolution(inputs,
                input_size,
                output_size,
                weight_size,
                stride=1,
                padding='SAME',
                weight_init=tf.glorot_uniform_initializer(),
                bias_init = tf.constant_initializer(0.0),
                batch_norm=True,
                activation=tf.nn.relu,
                dropout=False,
                dropout_rate=0.3,
                is_training=True,
                scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', shape=[weight_size, weight_size, input_size, output_size],
                                  dtype=tf.float32, initializer=weight_init)

        conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME',
                            name='conv2d')

        biases = tf.get_variable('biases', shape=[bias_size], dtype=tf.float32,
                                 initializer=bias_init)

        conv = tf.nn.bias_add(conv, biases, name='conv2d_preact')

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=is_training, name='conv2d_batchnorm')

        if activation:
            conv = activation(conv, 'conv2d_act')

        if dropout:
            conv = tf.layers.dropout(conv, rate=dropout_rate, training=is_training,
                                     name='conv2d_dropout')

    return conv


def pooling(inputs,
            k_size=2,
            stride=2,
            padding='SAME',
            scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        pool = tf.nn.max_pool(inputs, ksize=[1, k_size, k_size, 1],
                              strides=[1, stride, stride, 1], padding=padding, name='pool')

    return pool


def flatten(inputs,
            scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        shape = inputs.get_shape().as_list()
        size = shape[1] * shape[2] * shape[3]

        flatten = tf.reshape(inputs, (-1, size), name='flatten')

    return flatten


def dense(inputs,
          input_size,
          output_size,
          weight_init=tf.glorot_uniform_initializer(),
          bias_init=tf.constant_initializer(0.0),
          batch_norm=True,
          activation=tf.nn.relu,
          dropout=False,
          dropout_rate=0.3,
          is_training=True,
          scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', shape=[input_size, output_size],
                                  dtype=tf.float32, initializer=weight_init)

        biases = tf.get_variable('biases', shape=[output_size], dtype=tf.float32,
                                 initializer=bias_init)

        dense = tf.matmul(inputs, weights) + biases

        if batch_norm:
            dense = tf.layers.batch_normalization(dense, training=is_training, name='dense_batchnorm')

        if activation:
            dense = activation(dense, 'dense_act')

        if dropout:
            dense = tf.layers.dropout(dense, rate=dropout_rate, training=is_training,
                                      name='dense_dropout')

    return dense


def loss(logits,
         labels,
         scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        labels = tf.cast(labels, tf.int32)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits,
                                                                   name='loss')
        loss = tf.reduce_mean(cross_entropy)

    return loss


def accuracy(logits,
             labels,
             scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        labels = tf.argmax(labels, output_type=tf.int32)
        pred_labels = tf.argmax(tf.nn.softmax(logits), output_type=tf.int32)
        predicted_correct = tf.equal(labels, pred_labels)
        accuracy = tf.reduce_mean(tf.cast(predicted_correct, tf.float32))

        return accuracy


def optimize(loss,
             lr,
             scope=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return optimize
