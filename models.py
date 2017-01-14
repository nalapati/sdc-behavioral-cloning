"""Model definitions, construction, testing, validation, training.

NOTE: We used parts of this code as a framework for the Udacity
      SDC Challenge 2, https://github.com/emef/sdc, I am repurposing
      the framework in a slightly different way that makes hyper-parameter
      tuning slightly easier.
 
      The idea is to have models created by configuration, and training
      parameters created by configuration, so you can queue up a bunch of
      configurations and peek at the stats post training.

      The final form would be using a library like hyperop to control
      sections of the NN + its parameters to search for the best performing
      combo.
"""
import logging
import os
import time

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

    def __init__(self, model_config):
        self.model = keras_load_model(model_config['model_uri'])
        self.timesteps = model_config.get('timesteps', 0)
        self.scale = model_config.get('scale', 1.0)

    def fit(self, dataset, training_args, callbacks=None, final=False):
        batch_size = training_args.get('batch_size', 100)
        epochs = training_args.get('epochs', 5)
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
            training_generator = training_generator.with_timesteps(
                self.timesteps).with_scale(self.scale)
            validation_generator = validation_generator.with_timesteps(
                self.timesteps).with_scale(self.scale)
        
        history = self.model.fit_generator(
            training_generator,
            validation_data=validation_generator,
            samples_per_epoch=epoch_size,
            nb_val_samples=validation_size,
            nb_epoch=epochs,
            verbose=1,
            callbacks=(callbacks or []))

    def evaluate(self, dataset):
        generator = dataset.testing_generator(32)
        if self.timesteps:
            generator = generator.with_timesteps(self.timesteps)
        return std_evaluate(self, generator)

    def predict_on_batch(self, batch):
        return self.model.predict_on_batch(batch) / self.scale

    def save(self, model_path):
        save_model(self.model, model_path)
        return {
            'model_uri': model_path,
            'timesteps': self.timesteps,
            'scale': self.scale
        }

    @classmethod
    def create(cls, creation_args):
        # Only support sequential models
        timesteps = creation_args.get('timesteps', 0)
        scale = creation_args.get('scale', 1.0)
        img_input = Input(shape=creation_args['input_shape'])

        layer = Convolution2D(24, 5, 5, init="he_normal", border_mode="valid")(img_input)
        layer = Activation("relu")(layer)
        layer = MaxPooling2D((3, 3))(layer)
        layer = SpatialDropout2D(0.5)(layer)
        layer = BatchNormalization(axis=3)(layer)

        layer = Convolution2D(36, 5, 5, init="he_normal", border_mode="valid")(layer)
        layer = Activation("relu")(layer)
        layer = MaxPooling2D((3, 3))(layer)
        layer = SpatialDropout2D(0.5)(layer)
        layer = BatchNormalization(axis=3)(layer)
       
        layer = Convolution2D(48, 3, 3, init="he_normal", border_mode="valid")(layer)
        layer = Activation("relu")(layer)
        layer = MaxPooling2D((2, 2))(layer)
        layer = SpatialDropout2D(0.5)(layer)
        layer = BatchNormalization(axis=3)(layer)

        layer = Convolution2D(64, 3, 3, init="he_normal", border_mode="valid")(layer)
        layer = Activation("relu")(layer)
        layer = MaxPooling2D((2, 2))(layer)
        layer = SpatialDropout2D(0.5)(layer)
        layer = BatchNormalization(axis=3)(layer)

        layer = Flatten()(layer)

        layer = Dense(256)(layer)
        layer = PReLU()(layer)
        layer = Dropout(0.5)(layer)

        layer = Dense(100)(layer)
        layer = PReLU()(layer)
        layer = Dropout(0.5)(layer)

        layer = Dense(50)(layer)
        layer = PReLU()(layer)
        layer = Dropout(0.5)(layer)

        nn = Dense(1, W_regularizer=l2(0.0001))(layer)

        model = Model(input=img_input, output=nn)
        model.compile(
            loss='mean_squared_error',
            optimizer='adadelta',
            metrics=['rmse'])

        model.save(creation_args['model_uri'])
        return {
            'model_uri': creation_args['model_uri'],
            'timesteps': timesteps,
            'scale': scale
        }

def std_evaluate(model, generator):
    """
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
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

metrics.rmse = rmse

def train_model(args):
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
    return str(int(time.time()))

def main():
    logging.basicConfig(level=logging.INFO)
    train_model({
        "dataset_path": "/home/nalapati/udacity/sdc/udacity-p3/datasets/dataset_4",
        "model_path": "/home/nalapati/udacity/sdc/udacity-p3/models",
        "model_config": SdcModel.create({
            "input_shape": (160, 320, 3),
            "model_uri": "/home/nalapati/models/" + generate_id() + ".h5",
            "scale": 1.0
        }),
        "task_id": str(int(time.time())),
        "training_args": {
            "batch_size": 32,
            "epochs": 50
        },
    })

if __name__ == '__main__':
    main()
