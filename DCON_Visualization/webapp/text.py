
def get_DCON_markdown():
    markdown_text=r'''
        #### DCON
        
        This application uses the DCON algorithm as introduced in
        > D. Tschernutter, M. Kraus, S. Feuerriegel (2022). A Globally Convergent Algorithm for Neural Network Parameter Optimization Based on Difference-of-Convex Functions. Working Paper
        '''
    return markdown_text

def get_Adam_markdown():
    markdown_text=r'''
        #### Adam

        As a benchmark it uses the Adam algorithm as introduced in
        > D. P. Kingma and J. Ba (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
        
        In particular, it uses the implementation in
        > F. Chollet, and others (2015). Keras. https://keras.io

        Hyperparamters are tuned using the Keras RandomSearch Tuner on the following grids
        | Hyperparameter | Grid |
        | ------ | ------ |
        | Learning rate | $\{0.0005,0.001,0.005\}$ |
        | First moment exponential decay rate $\beta_1$ | $\{0.9,0.99\}$ |
        | L2-Regularization factor | $\{0.0,0.001,0.01\}$ |
        | Batch size | $\{64,128,\text{full}\}$ |
        '''
    return markdown_text

def get_Dataset_markdown():
    markdown_text=r'''
        #### Datasets

        All experiments use datasets from the UCI Machine Learning Repository
        > D. Dua, C. Graff (2017). UCI Machine Learning Repository. http://archive.ics.uci.edu/ml

        They are downloaded when first used and saved locally into a data folder.
        '''
    return markdown_text