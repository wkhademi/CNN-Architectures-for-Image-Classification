import argparse
import numpy as np
import tensorflow as tf
from test import Test
from train import Train
from utils import fetch_dataset, fetch_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LeNet')
    parser.add_argument('--data', type=str, default='MNIST')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--display_frequency', type=int, default=1000)
    parser.add_argument('--checkpoint_frequency', type=int, default=1000)
    conf = parser.parse_args()

    # get images and labels for specific dataset
    train_set, test_set, train_labels, test_labels = fetch_dataset(conf.data)

    # get model for specific CNN architecture
    model = fetch_model(conf.model)

    # train or test with specific CNN architecture and dataset
    if conf.train:
        trainer = Train(model, train_set, train_labels, conf)
        trainer.train_model()
    else:
        tester = Test(model, test_set, test_labels, conf)
        tester.test_model()
