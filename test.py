import numpy as np
import tensorflow as tf

class Test:
    def __init__(self, model, images, labels, conf):
        self.model = model
        self.images = images
        self.labels = labels
        self.conf = conf

    def test_model(self):
        pass
