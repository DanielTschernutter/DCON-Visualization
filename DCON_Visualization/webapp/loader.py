import requests
import os
import zipfile
import pandas as pd

from io import BytesIO
from config import Config

def save_dataset(config: Config, df: pd.DataFrame, ind: int):
    path = str(config.DATA_PATH)+"\Dataset"+str(ind+1)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name=str(config.DATA_PATH)+"\Dataset"+str(ind+1)+"\dataframe.pkl"
    df.to_pickle(file_name)

def load_dataset_from_UCL(config: Config, ind: int) -> pd.DataFrame:
    
    url = config.DATASET_URLS[ind]

    if ind==0:
        try:
            df = pd.read_table(url,
                               sep=",",
                               header=None,
                               names=["VENDOR","MODEL",
                                      "MYCT","MMIN",
                                      "MMAX","CACH",
                                      "CHMIN","CHMAX",
                                      "PRP","ERP"])
            
            save_dataset(config,df,ind)
        except:
            return False
        
        return True
    
    elif ind==1:
        try:
            df = pd.read_table(url, sep=",")
            
            save_dataset(config,df,ind)
        except:
            return False
        
        return True
    
    elif ind==2:
        try:
            sheets = ['1st period', '2nd period', '3rd period', '4th period']
            df = pd.concat(pd.read_excel(url, sheet_name=sheet, header=1) for sheet in sheets)
            df = df[[' Large B/P ', ' Large ROE ', ' Large S/P ', ' Large Return Rate in the last quarter ', 
                     ' Large Market Value ', ' Small systematic Risk', 'Annual Return.1']]
            
            save_dataset(config,df,ind)
        except:
            return False
        
        return True
    
    elif ind==3:
        try:
            df = pd.read_table(url,
                               delimiter=" ",
                               header=None,
                               names=["longitude",
                                      "prismatic_coefficient",
                                      "length_displacement",
                                      "beam_draught_ratio",
                                      "length_beam_ratio",
                                      "froude_number",
                                      "residuary_resistance"],
                               on_bad_lines='skip')
            # Delete prismatic_coefficient due to missing values
            df.drop(['prismatic_coefficient'], axis=1, inplace=True)
            
            save_dataset(config,df,ind)
        except:
            return False
        
        return True
    
    elif ind==4:
        try:
            # Create folder
            path = str(config.DATA_PATH)+"\Dataset"+str(ind+1)
            if not os.path.exists(path):
                os.makedirs(path)

            # download and unzip file
            r = requests.get(url, stream=True)
            with zipfile.ZipFile(BytesIO(r.content),"r") as zip_ref:
                zip_ref.extractall(path)
            
            # read csv file and delete unzipped files again
            csv_file_path = str(config.DATA_PATH)+"\Dataset"+str(ind+1)+"\dataset_Facebook.csv"
            df = pd.read_table(csv_file_path,delimiter=";")
            os.remove(csv_file_path)
            os.remove(str(config.DATA_PATH)+"\Dataset"+str(ind+1)+"\Facebook_metrics.txt")

            # Total interactions = comment + like + share
            df.drop(['comment'], axis=1, inplace=True)
            df.drop(['like'], axis=1, inplace=True)
            df.drop(['share'], axis=1, inplace=True)

            save_dataset(config,df,ind)
        except:
            return False
        
        return True
    
    elif ind==5:
        try:
            df = pd.read_excel(url, header=1)

            df.drop(['START YEAR', 'START QUARTER', 'COMPLETION YEAR', 'COMPLETION QUARTER'], axis=1, inplace=True)
            
            # Use time lag 4 that shows best results according to https://ascelibrary.org/doi/pdf/10.1061/%28ASCE%29CO.1943-7862.0001570
            df.drop(['V-{}'.format(x) for x in range(11,30)], axis=1, inplace=True)
            df.drop(['V-{}.{}'.format(x, y) for x in range(11,30) for y in [1, 2, 3]], axis=1, inplace=True)
            
            # V-9 = construction costs , V-10 = sales price ==> predict only sales price drop the other output variable
            df.drop(['V-9'], axis=1, inplace=True)

            save_dataset(config,df,ind)
        except:
            return False
        
        return True
    
    elif ind==6:
        try:
            df = pd.read_excel(url, header=0)
            
            # Drop numbering
            df.drop(['No'], axis=1, inplace=True)

            save_dataset(config,df,ind)
        except:
            return False
        
        return True
    
    elif ind==7:
        try:
            df = pd.read_table(url,
                               delimiter=";",
                               names=["CIC0",
                                      "SM1_Dz",
                                      "GATS1i",
                                      "NdsCH",
                                      "NdssC",
                                      "MLOGP",
                                      "quant_response"])
            
            save_dataset(config,df,ind)
        except:
            return False
        
        return True
    
    elif ind==8:
        try:
            df = pd.read_table(url,
                               delimiter=";",
                               names=["TPSA",
                                      "SAacc",
                                      "H-050",
                                      "MLOGP",
                                      "RDCHI",
                                      "GATS1p",
                                      "nN",
                                      "C-040",
                                      "quant_response"])
            
            save_dataset(config,df,ind)
        except:
            return False
        
        return True