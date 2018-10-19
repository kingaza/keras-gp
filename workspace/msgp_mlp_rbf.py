"""
MSGP-MLP regression on Kin40k data.
Reference: https://arxiv.org/abs/1511.02222

This example showcases semi-stochastic training
of GP-MLP model from scratch. Note that the original
paper used full-batch pretraining-finetuning scheme.
"""
from __future__ import print_function

import os

import numpy as np
np.random.seed(42)

# Keras
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping

# KGP
from kgp.models import Model
from kgp.layers import GP

# Model assembling and executing
from kgp.utils.experiment import train

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE



def load_resp_data(n_length=260, start=1, stop=2, delay=1, test_size=0.33, shuffle=False):
    
    assert (start < stop)
    assert (0 < test_size < 1)
    assert (delay>0)

    data_files = ['{:0>3d}.npy'.format(n) for n in np.arange(start,stop+1)]  
    data_list = [np.load('resp_data/{}'.format(f)) for f in data_files]
    data = np.concatenate(data_list)
    
    n_train = int(len(data) * (1-test_size))
    x_train, y_train = data[:n_train,:n_length], data[:n_train,n_length+delay-1]
    x_test, y_test = data[n_train:,:n_length], data[n_train:,n_length+delay-1]

    if shuffle:
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


def assemble_mlp(input_shape, output_shape, batch_size, nb_train_samples):
    """Assemble a simple MLP model.
    """
    inputs = Input(shape=input_shape)
    hidden = Dense(1024, activation='relu', name='dense1')(inputs)
    hidden = BatchNormalization(name='batchnorm1')(hidden)
    hidden = Dropout(rate=0.02, name='dropout1')(hidden)
    hidden = Dense(1024, activation='relu', name='dense2')(hidden)
    hidden = BatchNormalization(name='batchnorm2')(hidden)
    hidden = Dense(128, activation='relu', name='dense3')(hidden)
    hidden = BatchNormalization(name='batchnorm3')(hidden)
    hidden = Dense(16, activation='relu', name='dense4')(hidden)
    hidden = BatchNormalization(name='batchnorm4')(hidden)    
    hidden = Dense(2, activation='relu', name='dense5')(hidden)
    hidden = BatchNormalization(name='batchnorm5')(hidden)

    gp = GP(hyp={
                'lik': np.log(0.3),
                'mean': [],
                'cov': [[0.5], [1.0]],
            },
            inf='infGrid', dlik='dlikGrid',
            opt={'cg_maxit': 2000, 'cg_tol': 1e-6},
            mean='meanZero', cov='covSEiso',
            update_grid=1,
            grid_kwargs={'eq': 1, 'k': 70.},
            batch_size=batch_size,
            nb_train_samples=nb_train_samples)
    outputs = [gp(hidden)]
    return Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':    
# def main():

    # Load data
    (X_train,y_train), (X_test,y_test) = load_resp_data(n_length=260, start=1, stop=10, 
                                                        delay=2, shuffle=True)
    l = int(len(X_test)/2)
    # X_train, X_test, X_valid = standardize_data(X_train, X_test, X_valid)
    data = {
        'train': (X_train, y_train),
        'valid': (X_test[:l], y_test[:l]),
        'test': (X_test[l:], y_test[l:]),
    }
    print('Loaded Data...')
    print('  train:', data['train'][0].shape)
    print('  valid:', data['valid'][0].shape)
    print('  test:',  data['test'][0].shape)

    # Model & training parameters
    input_shape = data['train'][0].shape[1:]
    output_shape = data['train'][1].shape[1:]
    batch_size = 2**10
    epochs = 5000

    # Construct & compile the model
    model = assemble_mlp(input_shape, output_shape, batch_size,
                         nb_train_samples=len(X_train))
    opt = Adam(lr=1e-4)
    loss = [gen_gp_loss(gp) for gp in model.output_layers]
    model.compile(optimizer=opt, loss=loss)

    # Load saved weights (if exist)
    # if os.path.isfile('checkpoints/msgp_mlp.h5'):
    #     model.load_weights('checkpoints/msgp_mlp.h5', by_name=True)

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_mse', patience=50)]

    # Train the model
    history = train(model, data, gp_n_iter=5,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks, tensorboard=True,
                    checkpoint='msgp_mlp', checkpoint_monitor='val_mse', 
                    verbose=1)

    model.summary()                    

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    
    rmse_predict = RMSE(y_test, y_preds)
    print('Test RMSE:', rmse_predict)

    rmse_delay = RMSE(X_test[:,-1], y_preds)
    rmse_rel = rmse_predict / rmse_delay
    print('Relative RMSE:', rmse_rel)
    
# if __name__ == '__main__':    
#    main()
