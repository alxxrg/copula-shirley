import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from diffprivlib.mechanisms import GaussianAnalytic, GeometricTruncated, LaplaceTruncated

def SampleForVine(data, ratio, x):
    """Sample ratio*n_row from data and force at least two values per column on the sample x.

    Args:
        data (DataFrame): A DataFrame.
        ratio (float): The sampling ratio.
        x (int): 1 or 2, sample to force integrity on.

    Returns:
        DataFrame: First sample, with ratio*n_row number of rows.
        DataFrame: Second sample, with (1-ratio)*n_row number of rows.
    """
    resample = True
    while resample:
        sample_1 = data.sample(frac=ratio)
        sample_2 = data.drop(sample_1.index)
        if x == 1: vals = [np.unique(sample_1[col]) for col in sample_1.columns]
        else: vals = [np.unique(sample_2[col]) for col in sample_2.columns]
        if  np.min([len(v) for v in vals]) > 1:
            resample = False
    return sample_1, sample_2
    
def GetECDFs(ecdf_samples, epsilon=0.0, mechanism='Laplace', GS=2, delta=0.001):
    """Store differentially-private Empirical Cumulative Density Functions (ECDFs) in a Dictionary with the columns as the keys {col: ECFD}.
    When epsilon=0.0 returns a non-dp ECDF.

    Args:
        ecdf_samples (DataFrame): A DataFrame to estimate the ECDFs from. 
        epsilon (float, optional): The Differential Privacy bugdet (noise injection) parameter. Defaults to 0.0.
        mechanism (str, optional): The Differential Privacy random mechanism to sample noise from. Can be 'Laplace', 'Gaussian' or 'Geometric'. Defaults to 'Laplace'.
        GS (int, optional): The Global Sensitivity of the function for DP. This implementation uses the bounded version of the DP. Defaults to 2.
        delta (float, optional): The delta parameter to achieve (epsilon, delta)-DP with the Gaussian mechanism. Defaults to 0.001.

    Returns:
        dict: A Dictionary containing the ECDFs of each columns of the input DataFrame, with the columns names as the keys {col: ECFD}.
    """
    nobs = ecdf_samples.shape[0]
    if epsilon:
        if mechanism == 'Laplace':
            dp_mech = LaplaceTruncated(epsilon=epsilon, sensitivity=GS, lower=0, upper=nobs)
        elif mechanism == 'Gaussian':
            dp_mech = GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=GS)
        elif mechanism == 'Geometric':
            dp_mech = GeometricTruncated(epsilon=epsilon, sensitivity=GS, lower=0, upper=nobs)
        dp_ecdfs = {col:ECDF(ecdf_samples[col], dp_mech) for col in ecdf_samples.columns}
    else:
        dp_ecdfs = {col:ECDF(ecdf_samples[col]) for col in ecdf_samples.columns}
    return dp_ecdfs

def TransformToPseudoObservations(model_samples, dp_ecdfs):
    """Transform the samples into pseudo-observations via the Probability Integral Tranform (PIT).

    Args:
        model_samples (DataFrame): A DataFrame containing the training samples for the vine-copula model.
        dp_ecdfs (dict): A dictionary of ECDFs.

    Returns:
        matrix: (Noisy) pseudo-observations.
    """
    return np.array([dp_ecdfs[col].fct(model_samples[col]) for col in model_samples.columns]).T

def TransformToNatualScale(pseudo_samples, dp_ecdfs):
    """Transform the pseudo-observations back to the natural scale of the original data via the Inverse PIT.

    Args:
        pseudo_samples (DataFrame): A DataFrame containing the pseudo-observations generated from the vine-copula model.
        dp_ecdfs (dict): A dictionary of ECDFs.

    Returns:
        DataFrame: Samples with natural scale given by the ECDFs.
    """
    samples = pd.DataFrame(np.array([dp_ecdfs[col].inv(pseudo_samples[col]) for col in pseudo_samples.columns]).T)
    samples.columns = pseudo_samples.columns
    return samples

class ECDF(object):
    """Estimate the CDF of the input. If a DP mechanism is given, estimate a noisy CDF from histogram.

    Args:
        x (1-D array): An array containing observations to estimate de ECDF from.
        dpmechanism (object, optional): A diffprivlib.mechanisms class object to sample DP noise from. Defaults to None.

    Attributes:
        xs (1-D array): The x-axis values of the ECDF.
        yx (1-D array): The y-axis values of the ECDF.
        fct (function): The step function interpolated from the ECDF.
        inv (function): The step function of the inverse ECDF.
    """   
    def __init__(self, x, dpmechanism=None):
        def Histogram(X):
            h = {np.unique(X)[i]:np.sum(X == np.unique(X)[i]) for i in np.arange(np.unique(X).size)}
            k = np.fromiter(h.keys(), dtype=float)
            v = np.fromiter(h.values(), dtype=float)
            return k, v
        
        self.xs = np.array(x, copy=True)
        if dpmechanism is None:
            self.xs = np.reshape(self.xs, self.xs.size)
            self.xs = np.sort(self.xs)
            self.ys = (np.searchsorted(self.xs, self.xs, side='right') + 1)/self.xs.size
            self.ys = np.unique(self.ys)
            self.xs = np.unique(self.xs)
        else:
            self.xs, self.ys = Histogram(x)
            self.ys = np.fromiter([dpmechanism.randomise(np.int(self.ys[i])) for i in np.arange(self.ys.size)], self.ys.dtype, count=self.ys.size)
            self.ys[self.ys < 0.0] = 0.0
            self.ys = np.cumsum(self.ys)/np.sum(self.ys)
            
        self.ys[self.ys > 1.0] = 1.0
        self.fct = interp1d(self.xs, self.ys, kind='previous', fill_value=(0.0, 1.0), bounds_error=False, copy=True)
        self.inv = interp1d(self.ys, self.xs, kind='next', fill_value=(np.min(self.xs), np.max(self.xs)), bounds_error=False, copy=True)