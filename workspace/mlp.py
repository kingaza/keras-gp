"""
MLP regression on Kin40k data.
Attains about 0.14 RMSE within 150 epochs.
Reference: https://arxiv.org/abs/1511.02222
"""
from __future__ import print_function

import os

import numpy as np
np.random.seed(42)

# Keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping

# Model assembling and executing
from kgp.utils.experiment import train

# Metrics
from kgp.metrics import root_mean_squared_error as RMSE


def load_data(delay=1, test_size=0.33, shuffle=False):
    
    assert (test_size>0 and test_size<1)
    assert (delay>0)
	
    data = np.load("data/resp.npy")
    n_train = int(len(data) * (1-test_size))
    x_train, y_train = data[:n_train,:520], data[:n_train,520+delay-1]
    x_test, y_test = data[n_train:,:520], data[n_train:,520+delay-1]

    if shuffle:
		#Shuffle data. 
	    tmp = np.arange(len(x_train))
	    np.random.shuffle(tmp)
	    x_train,y_train = x_train[tmp],y_train[tmp]
		
	    tmp = np.arange(len(x_test))
	    np.random.shuffle(tmp)
	    x_test, y_test = x_test[tmp],y_test[tmp]	
    
    return [[x_train, y_train], [x_test, y_test]]


def standardize_data(X_train, X_test, X_valid):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid


def assemble_mlp(input_shape, output_shape):
    """Assemble a simple MLP model.
    """
    inputs = Input(shape=input_shape)
    hidden = Dense(1024, activation='relu', name='dense1')(inputs)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(512, activation='relu', name='dense2')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(64, activation='relu', name='dense3')(hidden)
    hidden = Dropout(0.25)(hidden)
    hidden = Dense(2, activation='relu', name='dense4')(hidden)
    outputs = Dense(1, activation='linear')(hidden)
    return Model(inputs=inputs, outputs=outputs)


def main():
    # Load data
    (X_train,y_train), (X_test,y_test) = load_data(delay=2, shuffle=True)
    X_valid, y_valid = X_test, y_test

    X_train, X_test, X_valid = standardize_data(X_train, X_test, X_valid)
    data = {
        'train': (X_train, y_train),
        'valid': (X_valid, y_valid),
        'test': (X_test, y_test),
    }

    # Model & training parameters
    input_shape = data['train'][0].shape[1:]
    output_shape = data['train'][1].shape[1:]
    batch_size = 128
    epochs = 100

    # Construct & compile the model
    model = assemble_mlp(input_shape, output_shape)
    model.compile(optimizer=Adam(1e-4), loss='mse')

    # Load saved weights (if exist)
    if os.path.isfile('checkpoints/mlp.h5'):
        model.load_weights('checkpoints/mlp.h5', by_name=True)    

    # Callbacks
    # callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=10),
    # ]
    callbacks = []

    # Train the model
    history = train(model, data, callbacks=callbacks,
                    checkpoint='mlp', checkpoint_monitor='val_loss',
                    epochs=epochs, batch_size=batch_size, verbose=1)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test RMSE:', rmse_predict)


if __name__ == '__main__':
    main()
