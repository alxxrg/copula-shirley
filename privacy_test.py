import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def MembershipScoreFunction(row, epsilon, delta=10e-12):
    sum_over_col = 0
    n = len(row)
    for i in range(n):
        if (row[i] <= epsilon): sum_over_col = sum_over_col + np.log(row[i] + delta)
    sum_over_col = sum_over_col * (-1/n)
    return sum_over_col 

def MembershipEpsilon(dist_mat):
    return np.median(np.amin(dist_mat, axis=1))
    
def MembershipScores(dist_mat, epsilon):
    return  pd.Series(np.apply_along_axis(MembershipScoreFunction, axis=1, arr=dist_mat, epsilon=epsilon))

def MembershipInferenceAttack(train_sample, test_sample, synth_sample, n, metric = 'euclidean', n_iter = 50):
    M = min(train_sample.shape[0], test_sample.shape[0])
    M_training  = train_sample.sample(M)
    M_test = test_sample.sample(M)
    M_data = np.array(pd.concat([M_training, M_test]))
    
    iter_privacy_scores = []
    for i in range(n_iter):
        n_synth = np.array(synth_sample.sample(n))
        distance_matrix = cdist(M_data, n_synth, metric = metric)
        attack_epsilon = MembershipEpsilon(distance_matrix)
        profile_scores = MembershipScores(distance_matrix, attack_epsilon)
        M_highest_scores = pd.Series(profile_scores.sort_values(ascending=False).index[:M])
        iter_privacy_scores.append(np.sum(M_highest_scores < M) / M)

    return np.mean(iter_privacy_scores), np.max(iter_privacy_scores)