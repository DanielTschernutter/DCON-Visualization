import plotly.express as px
import pandas as pd

def create_line_plot_loss(results,options):
    if options['early_stopping']:
        
        df = pd.DataFrame(list(zip(results['iterations'],results['mse_train'],["Training"]*len(results['iterations'])))+
                          list(zip(results['iterations'],results['mse_val'],["Validation"]*len(results['iterations']))),
                          columns=["Epochs","MSE","Loss"])
        
        fig = px.line(df, x='Epochs', y="MSE",color="Loss", log_y=True)
        return fig
    
    else:

        df = pd.DataFrame(list(zip(results['iterations'],results['mse_train'])),
                          columns=["Epochs","MSE"])
        
        fig = px.line(df, x='Epochs', y="MSE", log_y=True)
        return fig

def create_line_plot_norm_diffs(results):
    df = pd.DataFrame(list(zip(results['iterations'],results['norm_diffs'])),
                        columns=["Epochs","Distance between iterates"])
    
    fig = px.line(df, x='Epochs', y="Distance between iterates", log_y=True)
    return fig

def create_bar_plot_train_loss(results):
    
    df = pd.DataFrame([[results['train_loss_DCON'],"DCON"],
                       [results['train_loss_Keras'],"Adam"]],
                      columns=["MSE","Algorithm"])

    fig = px.bar(df, x='Algorithm', y='MSE')
    
    return fig

def create_bar_plot_test_loss(results):
    
    df = pd.DataFrame([[results['test_loss_DCON'],"DCON"],
                       [results['test_loss_Keras'],"Adam"]],
                      columns=["MSE","Algorithm"])

    fig = px.bar(df, x='Algorithm', y='MSE')
    
    return fig

def create_bar_plot_cputime(results):
    
    df = pd.DataFrame([[results['cpu_time_DCON'],"DCON"],
                       [results['cpu_time_Keras'],"Adam"]],
                      columns=["CPU Time","Algorithm"])

    fig = px.bar(df, x='Algorithm', y='CPU Time')
    
    return fig