import sys
import argparse
import numpy as np
import tensorflow as tf
from test import Test
from train import Train
from model import LeNet, AlexNet, VGG19, ResNet50, GoogLeNet
from utils import fetch_dataset, fetch_model, load_mnist, load_cifar

VALID_MODELS = {'LeNet': LeNet, 'AlexNet': AlexNet, 'VGG-19': VGG19,
                'ResNet-50': ResNet50, 'GoogLeNet': GoogLeNet}

VALID_DATASETS = {'MNIST': load_mnist, 'CIFAR': load_cifar}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LeNet')
    parser.add_argument('--data', type=str, default='MNIST')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    conf = parser.parse_args()

    # get images and labels for specific dataset
    if conf['data'] in VALID_DATASETS:
        train_set, test_set, train_labels, test_labels = fetch_dataset(conf['data'])
    else:
        print("Error: invalid dataset passed in as an argument")
        sys.exit()

    # get model for specific CNN architecture
    if conf['model'] in VALID_MODELS:
        model = fetch_model(conf['model'])
    else:
        print("Error: invalid model passed in as an argument")
        sys.exit()

    # train or test with specific CNN architecture and dataset
    if conf['train']:
        trainer = Train(model, train_set, train_labels, conf['epochs'], conf['batch_size'])
        trainer.train_model()
        trainer.save_model()
    else:
        tester = Test(model, test_set, test_labels)
        tester.load_model()
        tester.test_model()
