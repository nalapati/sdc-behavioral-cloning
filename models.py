"""Model definitions, construction, testing, validation, training.

NOTE: We used parts of this code as a framework for the Udacity
      SDC Challenge 2, https://github.com/emef/sdc, however for this
      project I experimented with 3D convolutional networks.
"""
import logging
import os
import time

# Adds functionality to work with a dataset
from datasets import load_dataset
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.engine.topology import Merge
from keras.layers import (
    Activation, BatchNormalization, Dense, Dropout, Flatten,
    Input, SpatialDropout2D, SpatialDropout3D, merge)
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import (
    AveragePooling2D, Convolution2D, Convolution3D,
    MaxPooling2D, MaxPooling3D)
from keras.models import Model, Sequential
from keras.models import load_model as keras_load_model
from keras.regularizers import l2

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class SdcModel(object):
    """ Contains functions to train/evaluate/save models.
    """
    def __init__(self, model_config):
        """
        @param model_config - dictionary containing a model configuration.
        """
        self.model = keras_load_model(model_config['model_uri'])
        self.timesteps = model_config['timesteps']

    def fit(self, dataset, training_args, callbacks=None):
        """ This method constructs a training and validation generator
        and calls keras model fit_generator to train the model.

        @param dataset - See Dataset(datasets.py)
        @param training_args - Dict containing training params 
                               (epochs, batch_size, pctl_sampling)
        @param callbacks - Any keras callbacks to use in the training process (snapshots,
                           early exit) and so on.
        """
        batch_size = training_args.get('batch_size', 100)
        epochs = training_args.get('epochs', 5)
        pctl_sampling = training_args.get('pctl_sampling', False)
        validation_size = training_args.get(
            'validation_size', dataset.validation_generator(
                batch_size).get_size())
        epoch_size = training_args.get(
            'epoch_size', dataset.training_generator(
                batch_size).get_size())

        # display model configuration
        self.model.summary() 

        training_generator = dataset.training_generator(batch_size)
        validation_generator = dataset.validation_generator(batch_size)
        if self.timesteps:
            # Timesteps for the 3D model.
            training_generator = training_generator.with_timesteps(
                self.timesteps)
            validation_generator = validation_generator.with_timesteps(
                self.timesteps)

        if pctl_sampling:
            training_generator = training_generator.with_pctl_sampling()
        
        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=(callbacks or []))

    def evaluate(self, dataset):
        """
        @param dataset - See Dataset(dataset.py)
        """
        generator = dataset.testing_generator(32)
        if self.timesteps:
            generator = generator.with_timesteps(self.timesteps)
        return std_evaluate(self, generator)

    def predict_on_batch(self, batch):
        """
        @param batch - batch of input per model configuration.
        """
        return self.model.predict_on_batch(batch)

    def save(self, model_path):
        """
        @param model_path - path at which to save the model.
        @return - dict with a model configuration.
        """
        save_model(self.model, model_path)
        return {
            'model_uri': model_path
        }

    @classmethod
    def create(cls, creation_args):
        """
        @param creation_args - Dict containing params with which to create a model.
                               (input_shape, timesteps, model_uri).
        @return - model configuration dict to be used to construct an SDCModel.
        """
        # Only support sequential models
        timesteps = creation_args['timesteps']
        img_input = Input(shape=creation_args['input_shape'])

        layer = MaxPooling3D((1, 2, 2))(img_input)
        layer = Convolution3D(60, 5, 5, 5, init="he_normal", activation="relu", border_mode="same")(layer)
        layer = MaxPooling3D((2, 3, 3))(layer)
        layer = SpatialDropout3D(0.5)(layer)
        layer = BatchNormalization(axis=4)(layer)

        layer = Convolution3D(120, 3, 3, 3, init="he_normal", activation="relu", border_mode="same")(layer)
        layer = MaxPooling3D((2, 3, 2))(layer)
        layer = SpatialDropout3D(0.5)(layer)
        layer = BatchNormalization(axis=4)(layer)

        layer = Convolution3D(180, 3, 3, 3, init="he_normal", activation="relu", border_mode="same")(layer)
        layer = MaxPooling3D((2, 3, 2))(layer)
        layer = SpatialDropout3D(0.5)(layer)
        layer = BatchNormalization(axis=4)(layer)

        layer = Flatten()(layer)

        layer = Dense(256)(layer)
        layer = PReLU()(layer)
        layer = Dropout(0.5)(layer)

        layer = Dense(1, W_regularizer=l2(0.001))(layer)

        model = Model(input=img_input, output=layer)
        model.compile(
            loss='mean_squared_error',
            optimizer='adadelta',
            metrics=['rmse'])

        model.save(creation_args['model_uri'])
        return {
            'model_uri': creation_args['model_uri'],
            'timesteps': creation_args['timesteps']
        }

def std_evaluate(model, generator):
    """
    Evaluates a model on the dataset represented by the generator.

    @param model - SDCModel
    @param generator - generator generating (batch_size, X, y)
    @return - list of mse, rmse
    """
    size = generator.get_size()
    batch_size = generator.get_batch_size()
    n_batches = size / batch_size

    err_sum = 0.
    err_count = 0.
    for _ in np.arange(n_batches):
        X_batch, y_batch = generator.__next__()
        y_pred = model.predict_on_batch(X_batch)
        err_sum += np.sum((y_batch - y_pred) ** 2)
        err_count += len(y_pred)

    mse = err_sum / err_count
    return [mse, np.sqrt(mse)]

def save_model(model, model_path):
    """
    Save a keras model to a local path.

    @param model - keras model
    @param model_path - local path to write to
    """
    try: os.makedirs(os.path.dirname(model_path))
    except: pass

    json_string = model.to_json()
    model.save(model_path)
    with open(model_path.replace("h5", "json"), "w") as f:
        f.write(json_string)
        f.write("\n")

def rmse(y_true, y_pred):
    """Calculates RMSE
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

metrics.rmse = rmse

def train_model(args):
    """ Trains a model using the specified args.
    @param args - Dict (model_config, dataset_path, task_id)
    """
    logger.info('loading model with config %s', args)

    model = SdcModel(args['model_config'])
    dataset = load_dataset(args['dataset_path'])
    baseline_mse = dataset.get_baseline_mse()
    logger.info('baseline mse: %f, baseline rmse: %f' % (
        baseline_mse, np.sqrt(baseline_mse)))

    earlystop = EarlyStopping(monitor="val_rmse", patience=12, mode="min")
    model.fit(dataset, args['training_args'], False, [earlystop])
    output_model_path = os.path.join(
        args['model_path'], '%s.h5' % args['task_id'])

    output_config = model.save(output_model_path)
    logger.info('Wrote final model to %s', output_model_path)

    # assume evaluation is mse
    evaluation = model.evaluate(dataset)
    training_mse = evaluation[0]

    improvement = -(training_mse - baseline_mse) / baseline_mse
    logger.info('Evaluation: %s', evaluation)
    logger.info('Baseline MSE %.5f, training MSE %.5f, improvement %.2f%%',
                baseline_mse, training_mse, improvement * 100)
    logger.info('output config: %s' % output_config)

def generate_id():
    """
    @return - a task id under which to store a model."
    """
    return str(int(time.time()))

def main():
    logging.basicConfig(level=logging.INFO)
    train_model({
        "dataset_path": "/home/nalapati/udacity/sdc/udacity-p3/datasets/dataset_32",
        "model_path": "/home/nalapati/udacity/sdc/udacity-p3/models",
        "model_config": SdcModel.create({
            "input_shape": (10, 80, 320, 3),
            "model_uri": "/home/nalapati/models/" + generate_id() + ".h5",
            "timesteps": 10
        }),
        "task_id": str(int(time.time())),
        "training_args": {
            "batch_size": 32,
            "epochs": 50
        },
    })

if __name__ == '__main__':
    main()
