import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from utils import shuffle_data, get_batch
from model import LeNet, AlexNet, VGG19, ResNet50, GoogLeNet

class Train:
    def __init__(self, model, images, labels, conf):
        self.model = model
        self.images = images
        self.labels = labels
        self.conf = conf
        self.image_shape = images.shape
        self.steps_per_epoch = int(self.image_shape[0] / float(self.conf.batch_size))
        self.max_steps = int(self.steps_per_epoch * self.conf.epochs)

    def train_model(self):
        """
            Train an image classification model with a certain architecture.
        """
        # get existing or create new checkpoint path
        if self.conf.load_model is not None:
            checkpoint = 'checkpoints/' + self.conf.load_model
        else:
            checkpoint_name = datetime.now().strftime('%d%m%Y-%H%M')
            checkpoint = 'checkpoints/' + checkpoint_name

            try:
                os.makedirs(checkpoint)
            except os.error:
                print('Error: Failed to make new checkpoint directory')
                sys.exit(1)

        # build graph for specific architecture
        graph = tf.Graph()
        with graph.as_default():
            inputs = tf.placeholder(tf.float32, shape=(self.conf.batch_size, self.image_shape[1], self.image_shape[2], self.image_shape[3]),
                                    name='inputs')
            labels = tf.placeholder(tf.float32, shape=(self.conf.batch_size, self.labels.shape[1]), name='outputs')

            model = self.model(self.image_shape[1],
                               self.image_shape[2],
                               self.image_shape[3],
                               self.conf.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                model.build_model(inputs, labels, is_training=True)

            summary_op = tf.summary.merge_all()
            writer = tf.summary.FileWriter(checkpoint, graph=graph)
            saver = tf.train.Saver(max_to_keep=2)

        with tf.Session(graph=graph) as sess:
            if self.conf.load_model is not None: # restore graph and last saved training step
                ckpt = tf.train.get_checkpoint_state(checkpoint)
                meta_graph_path = ckpt.model_checkpoint_path + '.meta'
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, tf.train.latest_checkpoint(checkpoint))
                start_step = int(meta_graph_path.split("-")[2].split(".")[0])
            else:
                sess.run(tf.global_variables_initializer())
                start_step = 1

            try:
                for step in range(start_step, self.max_steps): # train model for desired number of epochs
                    if step % self.steps_per_epoch == 0: # shuffle data every epoch
                        self.images, self.labels = shuffle_data(self.images, self.labels)

                    # fetch batch of images and their respective labels
                    image_batch, label_batch = get_batch(self.images, self.labels, self.conf.batch_size,
                                                         step, self.steps_per_epoch)

                    # update weights
                    loss, _ = sess.run([model.loss, model.optimizer],
                                        feed_dict={inputs: image_batch, labels: label_batch})

                    if step % self.conf.display_frequency == 0:
                        print('Step {}:'.format(step))
                        print('Training Loss: {}'.format(loss))

                    if step % self.conf.checkpoint_frequency == 0:
                        save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                        print("Model saved as {}".format(save_path))

            except KeyboardInterrupt: # save training progress upon user exit
                print('Saving models training progress to the `checkpoints` directory...')
                save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                print('Model saved as {}'.format(save_path))
                sys.exit(0)
