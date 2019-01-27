import tensorflow as tf
from main import VALID_MODELS, VALID_DATASETS
from model import LeNet, AlexNet, VGG19, ResNet50, GoogLeNet


def load_mnist():
    return tf.keras.datasets.cifar10.load_data()


def load_cifar():
    return tf.keras.datasets.mnist.load_data(path='mnist.npz')


def fetch_dataset(dataset_name):
    for name, load_dataset in VALID_DATASETS:
        if dataset_name is name:
            train, test = load_dataset()
            
            return train[0], test[0], train[1], test[1]


def fetch_model(model_name):
    for name, model in VALID_MODELS:
        if model_name is name:
            return model
