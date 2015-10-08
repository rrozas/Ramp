import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=15,
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)