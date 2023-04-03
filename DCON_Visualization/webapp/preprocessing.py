import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from config import Config

def preprocess_dataset(config: Config, ind: int) -> pd.DataFrame:
    
    file_name = str(config.DATA_PATH)+"\Dataset"+str(ind+1)+"\dataframe.pkl"

    if os.path.exists(file_name):
        df = pd.read_pickle(file_name)
    else:
        return None
    
    if ind==0:
        try:
            # Drop unused columns
            df.drop(['VENDOR'],axis=1, inplace=True)
            df.drop(['MODEL'],axis=1, inplace=True)
            df.drop(['ERP'],axis=1, inplace=True)
            
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='PRP']=covariate_scaler.fit_transform(df.loc[:,df.columns!='PRP'])
            df[['PRP']]=target_scaler.fit_transform(df[['PRP']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='PRP'])
            Y=np.ravel(np.array(df[['PRP']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,
                                                                Y,
                                                                random_state=config.RANDOM_STATE,
                                                                train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train,
                                                            Y_Train,
                                                            random_state=config.RANDOM_STATE,
                                                            train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None
    
    elif ind==1:
        try:
            # Label encoding of month
            df['month'].replace({'jan': 1,
                                'feb': 2,
                                'mar': 3,
                                'apr': 4,
                                'may': 5,
                                'jun': 6,
                                'jul': 7,
                                'aug': 8,
                                'sep': 9,
                                'oct': 10,
                                'nov': 11,
                                'dec': 12},inplace=True)
            
            # Label encoding of day
            df['day'].replace({'mon': 1,
                            'tue': 2,
                            'wed': 3,
                            'thu': 4,
                            'fri': 5,
                            'sat': 6,
                            'sun': 7},inplace=True)
            
            # Log transform of area as recommended
            df['area']=np.log(1+df['area'])
            
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='area']=covariate_scaler.fit_transform(df.loc[:,df.columns!='area'])
            df[['area']]=target_scaler.fit_transform(df[['area']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='area'])
            Y=np.ravel(np.array(df[['area']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,
                                                                Y,
                                                                random_state=config.RANDOM_STATE,
                                                                train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train,
                                                            Y_Train,
                                                            random_state=config.RANDOM_STATE,
                                                            train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None

    elif ind==2:
        try:
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='Annual Return.1']=covariate_scaler.fit_transform(df.loc[:,df.columns!='Annual Return.1'])
            df[['Annual Return.1']]=target_scaler.fit_transform(df[['Annual Return.1']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='Annual Return.1'])
            Y=np.ravel(np.array(df[['Annual Return.1']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=config.RANDOM_STATE, train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=config.RANDOM_STATE, train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None
        
    elif ind==3:
        try:
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='residuary_resistance']=covariate_scaler.fit_transform(df.loc[:,df.columns!='residuary_resistance'])
            df[['residuary_resistance']]=target_scaler.fit_transform(df[['residuary_resistance']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='residuary_resistance'])
            Y=np.ravel(np.array(df[['residuary_resistance']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=config.RANDOM_STATE, train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=config.RANDOM_STATE, train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None
        
    elif ind==4:
        try:
            # One hot encoding of Type
            df = pd.concat([df,pd.get_dummies(df['Type'], prefix='Type')],axis=1)
            df.drop(['Type'],axis=1, inplace=True)
            
            # Drop Nans    
            df = df.dropna()
            
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='Total Interactions']=covariate_scaler.fit_transform(df.loc[:,df.columns!='Total Interactions'])
            df[['Total Interactions']]=target_scaler.fit_transform(df[['Total Interactions']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='Total Interactions'])
            Y=np.ravel(np.array(df[['Total Interactions']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=config.RANDOM_STATE, train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=config.RANDOM_STATE, train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None
        
    elif ind==5:
        try:
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='V-10']=covariate_scaler.fit_transform(df.loc[:,df.columns!='V-10'])
            df[['V-10']]=target_scaler.fit_transform(df[['V-10']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='V-10'])
            Y=np.ravel(np.array(df[['V-10']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=config.RANDOM_STATE, train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=config.RANDOM_STATE, train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None
        
    elif ind==6:
        try:
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='Y house price of unit area']=covariate_scaler.fit_transform(df.loc[:,df.columns!='Y house price of unit area'])
            df[['Y house price of unit area']]=target_scaler.fit_transform(df[['Y house price of unit area']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='Y house price of unit area'])
            Y=np.ravel(np.array(df[['Y house price of unit area']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=config.RANDOM_STATE, train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=config.RANDOM_STATE, train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None
        
    elif ind==7:
        try:
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='quant_response']=covariate_scaler.fit_transform(df.loc[:,df.columns!='quant_response'])
            df[['quant_response']]=target_scaler.fit_transform(df[['quant_response']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='quant_response'])
            Y=np.ravel(np.array(df[['quant_response']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=config.RANDOM_STATE, train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=config.RANDOM_STATE, train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None
        
    elif ind==8:
        try:
            # Scalers
            covariate_scaler=RobustScaler()
            target_scaler=MinMaxScaler()
            
            # Scaling
            df.loc[:,df.columns!='quant_response']=covariate_scaler.fit_transform(df.loc[:,df.columns!='quant_response'])
            df[['quant_response']]=target_scaler.fit_transform(df[['quant_response']])
            
            # Create numpy arrays
            X=np.array(df.loc[:,df.columns!='quant_response'])
            Y=np.ravel(np.array(df[['quant_response']]))
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=config.RANDOM_STATE, train_size=0.8)
            X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=config.RANDOM_STATE, train_size=0.9)
            
            return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
        
        except:
            return None