"""
* Copyright <2019> <Thierry Rakotoarivelo> 
* https://github.com/thierryr/dpcopula_kendall
"""

import sys, os
sys.path.append(os.getcwd() + '/DPCopula/')

from DPCopula.synthetic import kendall_algorithm
from DPCopula.Database import Database
import pandas as pd
import numpy as np

def DPCopula(dataset, dp_eps=1):
    if not dp_eps: dp_eps = 10**3

    WD = os.getcwd()
    input_data = f'{WD}/temp/temp_{dataset}.csv'
    
    df = pd.read_csv(input_data, skipinitialspace=True)
    attr_dic = {i:df.iloc[:,i].unique().tolist() for i in range(0, len(df.columns))}
    
    db = Database()
    db.load_from_file(input_data, attr_dic)
    
    synthetic = kendall_algorithm(db, dp_eps/2, dp_eps/2)
    synthetic = pd.DataFrame(synthetic)
    synthetic.columns = df.columns
    return synthetic.apply(pd.to_numeric).dropna()