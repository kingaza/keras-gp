"""
Utility functions for running experiments, saving results, etc.
"""
import os
import sys
from time import time, localtime
import warnings

import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard

from kgp.metrics import root_mean_squared_error as RMSE


def train(model, data,
          epochs=100,
          batch_size=128,
          callbacks=None,
          tensorboard=False,
          checkpoint=None,
          checkpoint_monitor='val_loss',
          verbose=1,
          **fit_kwargs):
    """Train the model on the data.

    Arguments:
    ----------
        model : Model
            Assumes the model has been already compiled.
        data : dict
        epochs : uint (default: 100)
        batch_size : uint (default: 128)
        callbacks : list (default: None)
        checkpoint : str (default: None)
        verbose : uint (default: 1)

    Returns:
    --------
        history : training history
    """
    X_train, y_train = data['train']
    X_test, y_test = data['test']
    validation_data = data['valid'] if 'valid' in data else None
    callbacks = callbacks or []

    logs_folder = os.path.join('./', 'logs')
    if not os.path.isdir(logs_folder):
        os.makedirs(logs_folder)    

    t = localtime(time())
    s_format = '{:0>4d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}'
    time_stamp = s_format.format(t.tm_year, t.tm_mon, t.tm_mday, 
                                 t.tm_hour, t.tm_min, t.tm_sec)

    log_folder = os.path.join(logs_folder, time_stamp)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)
        
    if tensorboard is True:
        callbacks += [
            TensorBoard(log_dir=log_folder)
        ]

    if checkpoint is not None:
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)

        checkpoint_file = os.path.join(log_folder, '{}.h5'.format(checkpoint))
        callbacks += [
            ModelCheckpoint(checkpoint_file,
                            monitor=checkpoint_monitor,
                            save_weights_only=True,
                            save_best_only=True)
        ]

    # Train the model
    if verbose:
        sys.stdout.write("Training...\n")
        sys.stdout.flush()

    history = model.fit(X_train, y_train, validation_data=validation_data,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, verbose=verbose,
                        **fit_kwargs)

    if verbose:
        sys.stdout.write('Done.\n')

    # Test the model
    if checkpoint is not None:
        if os.path.isfile(checkpoint_file):
            model.load_weights(checkpoint_file)
        else:
            warnings.warn('Checkpoint file was specified, but no models were '
                          'saved by the monitor. Make sure the validation '
                          'dataset is specified and the monitoring channel '
                          'is set correctly.')

    return history

