import numpy as np
import tensorflow as tf

class Test:
    def __init__(self, model, images, labels, conf):
        self.model = model
        self.images = images
        self.labels = labels
        self.conf = conf
        self.image_shape = images.shape

    def test_model(self):
        """
            Test an image classification model with a certain architecture.
        """
        # get existing checkpoint path
        checkpoint = 'checkpoints/' + self.conf.load_model

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

        with tf.Session(graph=graph) as sess:
            # restore graph
            ckpt = tf.train.get_checkpoint_state(checkpoint)
            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoint))

            losses = []
            accuracy = []

            for image, label in zip(self.images, self.labels):
                loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict={inputs: image, labels: label})
                losses.append(loss)
                accuracies.append(accuracy)

            loss = np.mean(losses)
            accuracy = np.mean(accuracy)

            print('Test Loss: {}'.format(loss))
            print('Test Accuracy: {}'.format(accuracy))
