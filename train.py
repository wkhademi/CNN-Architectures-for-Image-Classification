import numpy as np
import tensorflow as tf

class Train:
    def __init__(self, model, images, labels, epochs, batch_size):
        self.model = model
        self.images = images
        self.labels = labels
        self.epochs = epochs
        self.batch_size = batch_size

    def train_model(self):
        pass

    def save_model(self):
        pass
