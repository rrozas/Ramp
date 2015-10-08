import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from itertools import repeat

def sample_from_rotation_x( x ):
    x_extends = []
    y_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    return np.array(x_extends)

def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)

class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        if criterion_smaller_is_better is True:
            self.best_valid = np.inf
        else:
            self.best_valid = -np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best {:s} was {:.6f} at epoch {}.".format(
                    self.criterion, self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)


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
            ('dropout1', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        on_epoch_finished = [EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
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
    max_epochs=100,dropout1_p=0.5,
    hidden4_W=init.GlorotUniform(gain='relu'),
    hidden5_W=init.GlorotUniform(gain='relu'),
    output_W=init.GlorotUniform(),
    batch_iterator_train=BatchIterator(batch_size=500)

)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = sample_from_rotation_x( X )
        #X = X.transpose((0, 3, 1, 2))
        return X

    def simple_preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        #X = sample_from_rotation_x( X )
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        y = sample_from_rotation_y(y)
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.simple_preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.simple_preprocess(X)
        return self.net.predict_proba(X)