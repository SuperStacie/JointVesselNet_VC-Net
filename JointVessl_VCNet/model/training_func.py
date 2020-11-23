"""
Author: Yifan Wang
Date created: 11/09/2020
This file is partially refered to https://github.com/ellisdg/3DUnetCNN
"""
import math
from functools import partial

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model


K.set_image_dim_ordering('th')


def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,          
                  learning_rate_patience=10, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):                                                             
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True,save_weights_only=True))   #default save best wgt only monitored on val_loss
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def fit_vcnet(model,model_file,X,Y,batch_size,epochs,val_data,initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                learning_rate_patience=10, early_stopping_patience=None):
    print('initial_lrate',initial_learning_rate)
    print('learning_rate_patience',learning_rate_patience)
    model.fit(x=X,y=Y,batch_size=batch_size,epochs=epochs,validation_data=val_data,
                                                callbacks=get_callbacks(model_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience))
