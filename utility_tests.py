import warnings
from itertools import permutations
from math import isinf, isnan

import numpy as np
import pandas as pd

from scipy.stats import entropy, ks_2samp, spearmanr, SpearmanRConstantInputWarning
from sklearn.neighbors import KernelDensity
from sklearn.metrics import matthews_corrcoef, mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier


def KLDiv(sample_E, sample_T):
    """Estimate the Kullback-Leibler Divergence between the experimental samples sample_E and the theorical samples sample_T.

    TODO:
        Adjust when sparse density.

    Args:
        sample_E (1-D array): The experimental observations.
        sample_T (1-D array): The theorical or reference observations.

    Returns:
        float: The Kullback-Leibler Divergence.
    """
    x = np.unique(sample_E.append(sample_T))
    x = x.reshape((x.size, 1))
    
    P = sample_E.to_numpy().reshape((sample_E.size, 1))
    Q = sample_T.to_numpy().reshape((sample_T.size, 1))
    
    model = KernelDensity(bandwidth=2)
    model.fit(P)
    prob_P = np.exp(model.score_samples(x))
    model.fit(Q)
    prob_Q = np.exp(model.score_samples(x))
    
    return entropy(prob_P, prob_Q)

def KSDist(sample_E, sample_T):
    """Estimate the Kolmogorov-Smirnov distance between the experimental samples sample_E and the theorical samples sample_T.

    Args:
        sample_E (1-D array): The experimental observations.
        sample_T (1-D array): The theorical or reference observations.

    Returns:
        float: The Kolmogorov-Smorniv distance.
    """
    return ks_2samp(sample_E, sample_T)[0]

def KLDivDF(sample_E, sample_T):
    """Estimate the Kullback-Leibler Divergence between all the columns of the experimental samples sample_E
    and the theorical samples sample_T and return the mean.

    Args:
        sample_E (n-D array): The experimental observations.
        sample_T (n-D array): The theorical or reference observations.

    Returns:
        float: The mean Kullback-Leibler Divergence between all columns.
    """
    res = [KLDiv(sample_E[col], sample_T[col]) for col in sample_E.columns]
    res = [v for v in res if not isinf(v)]
    return np.nanmean(res)

def KSDistDF(sample_E, sample_T):
    """Estimate the Kolmogorov-Smirnov distance between all the columns of the experimental samples sample_E
    and the theorical samples sample_T and return the mean.

    Args:
        sample_E (n-D array): The experimental observations.
        sample_T (n-D array): The theorical or reference observations.

    Returns:
        float: The mean Kolmogorov-Smorniv distance between all columns.
    """
    res = [KSDist(sample_E[col], sample_T[col]) for col in sample_E.columns]
    return np.mean(res)

def BinaryLabelCheck(df, label):
    if not (np.unique(df[label]).size == 2):
        raise ValueError("More than two labels for the binary label.")

    df_ret = df.copy()
    if (np.min(df[label]) == 0) and (np.max(df[label]) == 1):
        return df
    else:
        scale = np.min(df[label])
        df_ret[label] = df_ret[label] - scale
        return df_ret

def MultiClassLabelCheck(synth, real, label):
    synth_ret, real_ret = synth.copy(), real.copy()

    min_val = np.min([np.min(real_ret[label]), np.min(synth_ret[label])])
    max_val = np.max([np.max(real_ret[label]), np.max(synth_ret[label])])

    missing_values = list( set(range(min_val, (max_val+1))) - set(np.unique(synth_ret[label])) )
    if len(missing_values):
        for value in missing_values:
            synth_ret = synth_ret.append(pd.Series(np.nan, index=synth_ret.columns), ignore_index=True)
            synth_ret[label].iloc[-1] = value

    synth_ret[label] = synth_ret[label] - min_val
    real_ret[label] = real_ret[label] - min_val
    return synth_ret, real_ret

def BinaryClassif(synth_sample, real_sample, label, n_cores=1):
    """Computes the Matthews Correlation Coefficient (MCC) of the binary classification of label.
    The XGBoost model is trained on synth_sample and tested on real_sample.

    Args:
        synth_sample (DataFrame): A DataFrame of synthetic observations.
        real_sample (DataFrame): A DataFrame of orginal (raw) observations.
        label (str): The name of the class (must be binary).
        n_cores (int, optional): The number of cores to use. Defaults to 1.

    Returns:
        float: The MCC value of the classification.
    """
    synth_sample = BinaryLabelCheck(synth_sample, label)
    real_sample = BinaryLabelCheck(real_sample, label)

    train_col = list(set(synth_sample.columns) - set([label]))
    
    X_test = real_sample[train_col]
    y_test = real_sample[label]
    
    X_train = synth_sample[train_col]
    y_train = synth_sample[label]
    
    model = XGBClassifier(n_estimators=512,
                          use_label_encoder=False,
                          max_depth=64,
                          verbosity=0,
                          objective='binary:logistic',
                          eval_metric='error',
                          maximize=False,
                          n_jobs=n_cores,
                         )
    y_pred = model.fit(X_train, y_train).predict(X_test)
    
    return matthews_corrcoef(y_test, y_pred)

def MultiClassif(synth_sample, real_sample, label, n_cores=1):
    """Computes the Matthews Correlation Coefficient (MCC) of the multiclass classification of label.
    The XGBoost model is trained on synth_sample and tested on real_sample.
    
    Args:
        synth_sample (DataFrame): A DataFrame of synthetic observations.
        real_sample (DataFrame): A DataFrame of orginal (raw) observations.
        label (str): The name of the class.
        n_cores (int, optional): The number of cores to use. Defaults to 1.

    Returns:
        float: The MCC value of the multiclass classification.
    """
    synth_sample, real_sample = MultiClassLabelCheck(synth_sample, real_sample, label)

    train_col = list(set(synth_sample.columns) - set([label]))
    
    X_test = real_sample[train_col]
    y_test = real_sample[label]
    
    X_train = synth_sample[train_col]
    y_train = synth_sample[label]
    
    model = XGBClassifier(n_estimators=512,
                          use_label_encoder=False,
                          max_depth=64,
                          verbosity=0,
                          objective = 'multi:softmax',
                          num_class = np.unique(y_train).size,
                          eval_metric = 'merror',
                          maximize=False,
                          n_jobs=n_cores,
                         )
    y_pred = model.fit(X_train, y_train).predict(X_test)
    
    return matthews_corrcoef(y_test, y_pred)

def LinearRegr(synth_sample, real_sample, label, n_cores=1):
    """Computes the Root Mean Square Error (RMSE) of the regression problem of label.
    The Linear Regression model is trained on synth_sample and tested on real_sample.
    
    Args:
        synth_sample (DataFrame): A DataFrame of synthetic observations.
        real_sample (DataFrame): A DataFrame of orginal (raw) observations.
        label (str): The name of the class (must be continuous).
        n_cores (int, optional): The number of cores to use. Defaults to 1.

    Returns:
        float: The RMSE value of the regression.
    """
    train_col = list(set(synth_sample.columns) - set([label]))
    
    X_test = real_sample[train_col]
    y_test = real_sample[label]
    
    X_train = synth_sample[train_col]
    y_train = synth_sample[label]
    
    model = LinearRegression(n_jobs=n_cores)
    y_pred = model.fit(X_train, y_train).predict(X_test)
    
    return np.sqrt(mean_squared_error(y_test, y_pred))

def LocalCorr(sample, ref_col, q_col, percentile=0.05):
    """Computes the Pearson Correlation Coefficient between the reference column ref_col and the query column q_col 
        in the lowest and the highest percentile of the data, in respect with the ref_col.
        Example: First find the rows of the first and last percentile of the values in ref_col.
        Select the first percentile rows in the sample data and compute the correlation coefficient between ref_col and q_col.
        Do the same for the last percentile rows.
    
    Args:
        sample (DataFrame): A DataFrame.
        ref_col (DataFrame): The reference column to use.
        q_col (str): The query column to use.
        percentile (float, optional): The size of the area to use for the computation of the correlation coefficients, with respect to ref_col. Defaults to 0.05.

    Returns:
        float: The Pearson Correlation Coefficient in the first 5 percentiles between ref_col and q_col.
        float: The Pearson Correlation Coefficient in the last 5 percentiles between ref_col and q_col.
    """
    cdf = ECDF(sample[ref_col])
    l = sample[sample[ref_col] <= cdf.inv(percentile)]
    h = sample[sample[ref_col] >= cdf.inv(1-percentile)]
    warnings.simplefilter(action='ignore', category=SpearmanRConstantInputWarning)
    return spearmanr(l[ref_col], l[q_col])[0], spearmanr(h[ref_col], h[q_col])[0]

def BestPairForLocalCorr(sample, percentile=0.05):
    """Find the pair of attributes with low correlation in the first percentiles and high correlation in the last percentiles or vice-versa.
    Returns the pair with the highest difference.
    
    Args:
        sample (DataFrame): A DataFrame.
        percentile (float, optional): The size of the area to use for the computation of the correlation coefficients. Defaults to 0.05.

    Returns:
        tuple: A pair of attributes.
    """
    d = {pair:LocalCorr(sample, pair[0], pair[1], percentile) for pair in permutations(sample.columns, r=2)}
    index = np.nanargmax([abs(abs(v[0]) - abs(v[1])) for v in d.values()])
    return list(d.keys())[index]

def GlobalCorr(synth_sample, real_sample):
    """Compute two scores (max and mean) from the Spearman correlation coefficients between the two correlation matrices of synth_sample and real_sample.
    If a column in synth_sample or real_sample is single-valued, add a small noise on one (randomly) value of the column.

    Args:
        synth_sample (DataFrame): A DataFrame of synthetic observations.
        real_sample (DataFrame): A DataFrame of orginal (raw) observations.

    Returns:
        float: The maximal difference between the correlation coefficients of synth_sample and real_sample.
        float: The mean difference between the correlation coefficients of synth_sample and real_sample.
    """
    delta = 1*10**(-12)
    synth, real = synth_sample.values, real_sample.values

    for col in range(synth.shape[1]):
        if np.std(synth[:, col]) == 0:
            rand_row = np.random.choice(synth.shape[0])
            synth[rand_row, col] = synth[rand_row, col] - delta
        if np.std(real[:, col]) == 0:
            rand_row = np.random.choice(real.shape[0])
            real[rand_row, col] = real[rand_row, col] - delta

    real_corr = spearmanr(real)[0]
    synth_corr = spearmanr(synth)[0]
    max_diff_corr = np.max(np.abs(real_corr - synth_corr))
    mean_diff_corr = np.mean(np.abs(real_corr - synth_corr))
    return max_diff_corr, mean_diff_corr




