"""
LSTM regression on Actuator data.
"""
from __future__ import print_function

import numpy as np
np.random.seed(42)

# Keras
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, assemble
from kgp.utils.experiment import train

# Metrics
from kgp.metrics import root_mean_squared_error as RMSE


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


def load_data(delay=1, test_size=0.33, shuffle=False):
    
    assert (test_size>0 and test_size<1)
    assert (delay>0)
	
    data = np.load("data/resp.npy")
    data = data[:,:,None]
    n_train = int(len(data) * (1-test_size))
    x_train, y_train = data[:n_train,:520,:], data[:n_train,520+delay-1,:]
    x_test, y_test = data[n_train:,:520,:], data[n_train:,520+delay-1,:]

    if shuffle:
		#Shuffle data. 
	    tmp = np.arange(len(x_train))
	    np.random.shuffle(tmp)
	    x_train,y_train = x_train[tmp],y_train[tmp]
		
	    tmp = np.arange(len(x_test))
	    np.random.shuffle(tmp)
	    x_test, y_test = x_test[tmp],y_test[tmp]	
    
    return [[x_train, y_train], [x_test, y_test]]

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
    input_shape = list(data['train'][0].shape[1:])
    output_shape = list(data['train'][1].shape[1:])
    batch_size = 16
    epochs = 100

    nn_params = {
        'H_dim': 32,
        'H_activation': 'tanh',
        'dropout': 0.1,
    }

    # Retrieve model config
    configs = load_NN_configs(filename='lstm.yaml',
                              input_shape=input_shape,
                              output_shape=output_shape,
                              params=nn_params)

    # Construct & compile the model
    model = assemble('LSTM', configs['1H'])
    model.compile(optimizer=Adam(1e-1), loss='mse')

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

    # Train the model
    history = train(model, data, callbacks=[],
                    checkpoint='lstm_actuator', checkpoint_monitor='val_loss',
                    epochs=epochs, batch_size=batch_size, verbose=2)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test RMSE:', rmse_predict)


if __name__ == '__main__':
    main()
