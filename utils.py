import sys
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.utils import shuffle
from model import LeNet, AlexNet, VGG19, ResNet50, GoogLeNet


def load_cifar():
    """
        Load CIFAR10 dataset.
    """
    return tf.keras.datasets.cifar10.load_data()


def load_mnist():
    """
        Load MNIST dataset.
    """
    return tf.keras.datasets.mnist.load_data(path='mnist.npz')


# list of valid models
VALID_MODELS = {'LeNet': LeNet, 'AlexNet': AlexNet, 'VGG-19': VGG19,
                'ResNet-50': ResNet50, 'GoogLeNet': GoogLeNet}

# list of valid datasets
VALID_DATASETS = {'MNIST': load_mnist, 'CIFAR': load_cifar}


def fetch_dataset(dataset_name):
    """
        Fetch the images and labels for a specified dataset.

        Args:
            dataset_name: specifies the name of the dataset

        Returns:
            train_imgs: Training image set
            test_imgs: Testing image set
            train_labels: Corresponding training labels for training image set
            test_labels: Corresponding testing labels for testing image set
    """
    for name, load_dataset in VALID_DATASETS.iteritems():
        if dataset_name == name:
            train, test = load_dataset()

            # unpack tuples
            train_imgs = train[0]
            train_labels = train[1]
            test_imgs = test[0]
            test_labels = test[1]

            if len(train_imgs.shape) == 3: # expand dimension for MNIST dataset
                train_imgs = np.expand_dims(train_imgs, axis=-1)
                test_imgs = np.expand_dims(test_imgs, axis=-1)

            # normalize images
            train_imgs = (train_imgs - np.mean(train_imgs)) / 255.
            test_imgs = (test_imgs - np.mean(test_imgs)) / 255.

            # one hot encode labels
            with tf.Session() as sess:
                train_labels = sess.run(tf.one_hot(train_labels, 10))
                test_labels = sess.run(tf.one_hot(test_labels, 10))

            return train_imgs, test_imgs, train_labels, test_labels

    print("Error: invalid dataset passed in as an argument")
    sys.exit()


def fetch_model(model_name):
    """
        Fetch the specified model architecture.

        Args:
            model_name: Name of the desired model architecture

        Returns:
            model: Desired model object
    """
    for name, model in VALID_MODELS.iteritems():
        if model_name == name:
            return model

    print("Error: invalid model passed in as an argument")
    sys.exit()


def shuffle_data(images, labels):
    """
        Shuffle images and labels in a consistent manner.

        Args:
            images: Set of images
            labels: Set of labels

        Returns:
            images: Set of shuffled images
            labels: Set of shuffled labels
    """
    return shuffle(images, labels)


def get_batch(images, labels, batch_size, index, max_index):
    """
        Get a batch of images and labels from a the dataset.

        Args:
            images: Set of images
            labels: Set of labels
            batch_size: Number of images and labels to grab from the set
            index: Start index to grab images and labels from in the set
            max_index: Highest index the dataset goes up to

        Returns:
            image_batch: Batch of images from dataset
            label_batch: Corresponding batch of labels from dataset
    """
    start = (index % max_index) * batch_size
    end = start + batch_size

    image_batch = images[start:end]
    label_batch = labels[start:end]

    return image_batch, label_batch
