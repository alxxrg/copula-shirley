import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from category_encoders import WOEEncoder, GLMMEncoder, OneHotEncoder, OrdinalEncoder

from utility_tests import BinaryLabelCheck

def CEDecoder(decoder_dict):
    """Construct a custom decoder for to map the encoded data back to their natural scale (nominal values).

    Args:
        decoder_dict (dict): A nested dictionary of the form {col:{encoded_values:original_values}}.

    Returns:
        lambda function: A Decoder function.
    """
    dec_dict = decoder_dict.copy()
    def FindClosestVal(num, d):
        return d[num] if num in d else d[min(d.keys(), key=lambda k: abs(k-num))]
        
    def Decode(X_in, decoder_dict):
        X_out = X_in.copy()
        for column in decoder_dict.keys():
            X_out[column] = X_out[column].apply(FindClosestVal, d=decoder_dict[column])
        return X_out
    
    return lambda X: Decode(X, dec_dict)

def PreprocessData(df, train_idx=None, test_idx=None, cat_encoder=None, cat_attr=None, datetime_attr=None, enc_target=None):
    """Preprocess the categorical attributes to ordinal scale.

    Args:
        df (DataFrame): The data to be preprocessed.
        cat_encoder (str, optional): Which encoder to use: 
            'WOE'=Weight of Evidence, 'GLMM'=Generalized Linear Mixed Model, 'OHE'=One Hot, None=Ordinal.  Defaults to None.
        cat_attr (list, optional): List of the categorical attributes. If None, tries to infer from datatype. Defaults to None.
        datetime_attr (list, optional): List of the datetime attributes. Defaults to None.
        enc_target (str, optional): A target attribute for the WOE and GLMM encoders. 'WOE' target attribute needs to be binary, can't be None.
            'GLMM' target can be binary or numerical, if None pick a random numerical attribute. Defaults to None.

    Returns:
        Dataframe: An encoded DataFrame.
        Function: A Decoder function.
    """
    data = df.copy()
    
    if datetime_attr is not None:
        data[datetime_attr] = data[datetime_attr].astype('np.datetime64')
    if cat_attr is not None:
        data[cat_attr] = data[cat_attr].astype('category')
    else:
        cat_attr = list(set(data.columns) - set(data.select_dtypes(np.number).columns))
        data[cat_attr] = data[cat_attr].astype('category')
    
    if cat_encoder == 'WOE':
        if enc_target is None:
            raise TypeError("WOE Encoder target class can't be None.")
        if np.unique(data[enc_target]).size > 2:
            raise TypeError("WOE Encoder target class must be binary. Use binary class or use GLMM encoder.")
        encoder = WOEEncoder(cols=cat_attr)
        data = BinaryLabelCheck(data, enc_target)
        if (train_idx is not None) and (test_idx is not None):
            encoder.fit(data.iloc[test_idx], y=data.iloc[test_idx][enc_target])
            clean_data = encoder.transform(data.iloc[train_idx])
        else:
            encoder.fit(data, y=data[enc_target])
            clean_data = encoder.transform(data)
        decoder_dict = {col:{clean_data[col][i]:data[col][i] for i in clean_data.index} for col in cat_attr}
        decoder = CEDecoder(decoder_dict)
    elif cat_encoder == 'GLMM':
        if enc_target is None:
            warnings.warn("GLMM Encoder target attribute isn't declared. A random numerical attribute will be chosen as a target class.")
            enc_target = np.random.choice(data.select_dtypes(np.number).columns)
        encoder = GLMMEncoder(cols=cat_attr)
        if (train_idx is not None) and (test_idx is not None):
            encoder.fit(data.iloc[test_idx], y=data.iloc[test_idx][enc_target])
            clean_data = encoder.transform(data.iloc[train_idx])
        else:
            encoder.fit(data, y=data[enc_target])
            clean_data = encoder.transform(data)
        decoder_dict = {col:{clean_data[col][i]:data[col][i] for i in clean_data.index} for col in cat_attr}
        decoder = CEDecoder(decoder_dict)
    elif cat_encoder == 'OHE':
        encoder = OneHotEncoder(cols=cat_attr, use_cat_names=True)
        encoder.fit(data)
        clean_data = encoder.transform(data)
        decoder = encoder.inverse_transform
    else:
        encoder = OrdinalEncoder(cols=cat_attr)
        encoder.fit(data)
        clean_data = encoder.transform(data)
        decoder = encoder.inverse_transform
    
    return clean_data, decoder