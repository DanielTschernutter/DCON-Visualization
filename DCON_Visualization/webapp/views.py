import streamlit as st

from config import Config

def dataset_navigation_bar(config: Config):
    #st.sidebar.title("Experiment Settings")
    tab1, tab2, tab3 = st.sidebar.tabs(["General settings","DCON","Adam"])

    # General settings
    with tab1:
        selected_dataset = st.selectbox("Dataset", config.DATASET_NAMES)
        n_hidden = st.number_input("Number of hidden neurons", 5, 30, value=10)
        selected_early_stopping = st.selectbox("Early Stopping", ["yes","no"], index=0)
        if selected_early_stopping == "yes":
            selected_early_stopping = True
        else:
            selected_early_stopping = False
        patience = st.number_input("Patience for early stopping", 1, 10, value=5, disabled=not selected_early_stopping)
    
    with tab2:
        reg_param = st.selectbox("Regularization", [0.1,0.01,0.001], index=1)
        n_epochs = st.slider("Maximum number of epochs", 1, 200, value=100, key="DCON_epochs")
    with tab3:
        n_epochs_Keras = st.slider("Maximum number of epochs", 1, 200, value=100, key="Keras_epochs")
        
    options = {'n_epochs': n_epochs,
               'n_hidden': n_hidden,
               'max_num_DC_iterations': 50,
               'reg_param': reg_param,
               'n_epochs_Keras': n_epochs_Keras,
               'early_stopping': selected_early_stopping,
               'patience': patience}

    return config.DATASET_NAMES.index(selected_dataset), options
