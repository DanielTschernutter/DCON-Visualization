import time
import streamlit as st
import tensorflow as tf
import numpy as np
import keras_tuner as kt

from config import Config
from tensorflow.python.keras import regularizers
from DCON import DCON
from webapp.preprocessing import preprocess_dataset

class Baseline_Hypermodel(kt.HyperModel):
    def __init__(self, n, N):
        self.n = n
        self.N = N
        
    def build(self, hp):    
    
        # hyperparameter grid (see .fit for hyperparameter grid for batch_size)
        lr=hp.Choice("lr",[0.0005, 0.001, 0.005])
        beta_1=hp.Choice("beta_1",[0.9, 0.99])
        reg_param=hp.Choice("reg_param",[0.0, 0.001, 0.01])
        
        if reg_param==0.0:
            model = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.N, activation='relu', input_dim=self.n),
                    tf.keras.layers.Dense(1, use_bias=False)
                    ])
        else:
            model = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.N, activation='relu', input_dim=self.n, kernel_regularizer=regularizers.l2(reg_param), bias_regularizer=regularizers.l2(reg_param)),
                    tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=regularizers.l2(reg_param))
                    ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999)
        
        model.compile(loss='mse',optimizer=optimizer)
          
        return model
        
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [64, 128, args[1].shape[0]]),
            **kwargs,
        )

@st.cache_data(show_spinner=False)
def train_and_evaluate_DCON(config: Config, ind: int, options: dict):
    
    # Get Data
    X_Train,X_Val,X_Test,Y_Train,Y_Val,Y_Test = preprocess_dataset(config, ind)
    n = X_Train.shape[1]
    m = X_Train.shape[0]

    ###### Train DCON ######
    model = DCON(n_hidden=options['n_hidden'], n_inputs=n)

    if options['early_stopping']:
        start_DCON = time.time()
        model.fit(X_Train, Y_Train,
                n_epochs=options['n_epochs'],
                max_num_DC_iterations=options['max_num_DC_iterations'],
                verbose=0,
                reg_param=options['reg_param'],
                random_seed=config.RANDOM_STATE,
                validation_data=(X_Val, Y_Val),
                patience_early_stopping=options['patience'])
        end_DCON = time.time()
    else:
        start_DCON = time.time()
        model.fit(X_Train, Y_Train,
                n_epochs=options['n_epochs'],
                max_num_DC_iterations=options['max_num_DC_iterations'],
                verbose=0,
                reg_param=options['reg_param'],
                random_seed=config.RANDOM_STATE)
        end_DCON = time.time()
    keras_model=model.get_keras()
    cpu_time = end_DCON - start_DCON

    ###### Evaluate ######
    test_loss=keras_model.evaluate(X_Test,Y_Test,verbose=0)
    train_loss=keras_model.evaluate(X_Train,Y_Train,verbose=0)

    output_dict = {'train_loss_DCON': train_loss,
                   'test_loss_DCON': test_loss,
                   'cpu_time_DCON': cpu_time,
                   'epochs_DCON': model.n_epochs,
                   'iterations': np.arange(1,model.n_epochs+1),
                   'mse_train': model.mse,
                   'mse_val': model.mse_val,
                   'norm_diffs': model.norm_diffs}

    return output_dict

@st.cache_data(show_spinner=False)
def train_and_evaluate_Keras(config: Config, ind: int, options: dict):
    
    # Get Data
    X_Train,X_Val,X_Test,Y_Train,Y_Val,Y_Test = preprocess_dataset(config, ind)
    n = X_Train.shape[1]
    m = X_Train.shape[0]
    
    ###### Train Keras ######
    tf.random.set_seed(config.RANDOM_STATE)
    tf.keras.utils.set_random_seed(config.RANDOM_STATE)
    tuner = kt.RandomSearch(Baseline_Hypermodel(n,options['n_hidden']),
                            objective='val_loss',
                            max_trials=5,
                            seed=config.RANDOM_STATE,
                            project_name=str(config.DATA_PATH)+"\KerasTuner",
                            overwrite=True)
    
    if options['early_stopping']:
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=options['patience'])
        start_Keras = time.time()
        tuner.search(X_Train, Y_Train,
                    epochs=options['n_epochs_Keras'],
                    validation_data=(X_Val, Y_Val),
                    verbose=0,
                    callbacks=[stop_early])
        end_Keras = time.time()
    else:
        start_Keras = time.time()
        tuner.search(X_Train, Y_Train,
                    epochs=options['n_epochs_Keras'],
                    validation_data=(X_Val, Y_Val),
                    verbose=0)
        end_Keras = time.time()

    cpu_time = end_Keras - start_Keras
    baseline_model = tuner.get_best_models()[0]
    best_hp = tuner.get_best_hyperparameters()[0]


    ###### Evaluate ######
    test_loss=baseline_model.evaluate(X_Test,Y_Test,verbose=0)
    train_loss=baseline_model.evaluate(X_Train,Y_Train,verbose=0)

    output_dict = {'train_loss_Keras': train_loss,
                   'test_loss_Keras': test_loss,
                   'cpu_time_Keras': cpu_time,
                   'keras_hyperparameters': best_hp.values}

    return output_dict