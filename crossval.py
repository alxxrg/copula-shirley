# python main.py --n-fold=5 --dataset='adult' --model='cop-shirl' --categorical-encoder=None --cat-encoder-target=None --dp-epsilon=1.0 --dp-mechanism='Laplace' --vine-sample-ratio=0.6 --vine-family-set='all' --vine-trunc-lvl=None --vine-tree-crit='tau' --binary-classif-label=None --multi-classif-label=None --linearreg-label=None


import os
import sys
import argparse
from pathlib import Path
from time import perf_counter

from sklearn.model_selection import KFold

from data import TRAIN_DATASETS, DATASET_CONFIGS
from preprocess import *
from transform import *
from vine import *
from utility_tests import *
from privacy_test import *
from privbayes import *
from dpcopula import * 


parser = argparse.ArgumentParser(description='k-Fold Cross-Validation Utility Testing for Synthetic Data Generation')

parser.add_argument('--n-folds', type=int, default=5)
parser.add_argument('--dataset', default='adult', choices=list(TRAIN_DATASETS.keys()))
parser.add_argument('--model', default='cop-shirl', choices=['cop-shirl', 'privbayes', 'dpcopula', 'dp-histogram'])
parser.add_argument('--seed', type=int, default=76543)
parser.add_argument('--n-sample', type=int, default=None)

parser.add_argument('--categorical-encoder', default=None, choices=['ORD', 'WOE', 'GLMM', 'OHE'])
parser.add_argument('--cat-encoder-target', type=str, default=None)

parser.add_argument('--dp-epsilon', type=float, default=1.0)
parser.add_argument('--dp-mechanism', default='Laplace', choices=['Laplace', 'Gaussian', 'Geometric'])
parser.add_argument('--dp-global-sens', type=int, default=2)
parser.add_argument('--dp-gaussian-delta', type=float, default=0.001)

parser.add_argument('--vine-sample-ratio', type=float, default=0.5)
parser.add_argument('--vine-family-set', type=str, default='all')
parser.add_argument('--vine-par-method', type=str, default='mle')
parser.add_argument('--vine-nonpar-method', type=str, default='constant')
parser.add_argument('--vine-selcrit', type=str, default='aic')
parser.add_argument('--vine-trunc-lvl', type=int, default=None)
parser.add_argument('--vine-tree-crit', type=str, default='rho')

parser.add_argument('--privbayes-degree-max', type=int, default=3)

parser.add_argument('--MIA-n', type=int, default=500)
parser.add_argument('--MIA-metric', type=str, default='hamming')
parser.add_argument('--MIA-n-iter', type=int, default=50)

parser.add_argument('--output-dir', type=str, default='./out')

parser.add_argument('--n-cores', type=int, default=None)
parser.add_argument('--verbose', type=bool, default=True)

if __name__ == '__main__':
    
    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    n_folds = args.n_folds
    output_dir = args.output_dir
    n_cores = args.n_cores
    if n_cores is None:
        n_cores = os.cpu_count()
    verbose = args.verbose
    
    model = args.model
    
    dataset = args.dataset
    dataset_config = DATASET_CONFIGS[args.dataset]

    datetime_attributes = dataset_config['datetime']
    categorical_attributes = dataset_config['categorical']
    
    categorical_encoder = args.categorical_encoder
    if categorical_encoder is None: categorical_encoder = 'ORD'
    cat_encoder_target = args.cat_encoder_target
    
    dp_epsilon = args.dp_epsilon
    dp_mechanism = args.dp_mechanism
    dp_global_sens = args.dp_global_sens
    dp_gaussian_delta = args.dp_gaussian_delta
    
    vine_sample_ratio = args.vine_sample_ratio
    
    vine_family_set = args.vine_family_set
    vine_par_method = args.vine_par_method
    vine_nonpar_method = args.vine_nonpar_method
    vine_selcrit = args.vine_selcrit
    vine_trunc_lvl = args.vine_trunc_lvl
    vine_tree_crit = args.vine_tree_crit
    
    privbayes_degree_max = args.privbayes_degree_max
    
    n_sample = args.n_sample
    
    binary_classif_label = dataset_config['binary_classif_label']
    multi_classif_label = dataset_config['multi_classif_label']
    linearreg_label = dataset_config['linearreg_label']
    
    MIA_n = args.MIA_n
    MIA_metric = args.MIA_metric
    MIA_n_iter = args.MIA_n_iter
    
    #if verbose: sys.stdout.write('\r'+'Reading and Preprocessing Data...                         ')
    data = pd.read_csv(f'./Datasets/{dataset}.csv', skipinitialspace=True)
    if not dp_epsilon:
        clean_data, decoder = PreprocessData(data, cat_encoder=categorical_encoder, cat_attr=categorical_attributes, datetime_attr=datetime_attributes, enc_target=cat_encoder_target)
    
    if n_sample is None: n_sample = data.shape[0]

    column_names = [f'Fold {f}' for f in range(1, n_folds+1, 1)]
    row_names = ['Binary Class MCC', 'Multi Class MCC',
                 'LinReg RMSE', 'Max Corr Delta', 'Mean Corr Delta',
                 'KS Dist', 'KL Div',
                 'Privacy Mean', 'Privacy Worst',
                 'Execution time (sec)']
    results_df = pd.DataFrame(np.nan, columns=column_names, index=row_names)
    
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    f = 0
    
    for train_index, test_index in kf.split(data):
        f = f + 1
        if not dp_epsilon:
            kf_train, kf_test = clean_data.iloc[train_index], data.iloc[test_index]
        else:
            kf_test = data.iloc[test_index]
            kf_train, decoder = PreprocessData(data, train_index, test_index, cat_encoder=categorical_encoder, cat_attr=categorical_attributes, datetime_attr=datetime_attributes, enc_target=cat_encoder_target)

        # Remove columns that are single-valued
        constant_cols = (pd.Series([np.std(kf_train[col]) for col in kf_train.columns]) == 0)
        constant_cols = data.columns[constant_cols[constant_cols].index].to_list()
        constant_vals = [kf_train[col][1] for col in constant_cols]
        kf_train = kf_train.drop(constant_cols, axis=1)
        
        start_time = perf_counter()
        if model == 'cop-shirl':
            if verbose: sys.stdout.write('\r'+f'Fold {f} of {n_folds}: Training Vine Copula Model...                         ')
            if dp_epsilon: vine_train_samples, ecdf_train_samples = SampleForVine(kf_train, vine_sample_ratio, 2)
            else: vine_train_samples = ecdf_train_samples = kf_train

            dp_ecdfs = GetECDFs(ecdf_train_samples, epsilon=dp_epsilon, mechanism=dp_mechanism, GS=dp_global_sens, delta=dp_gaussian_delta)
            pseudo_obs = TransformToPseudoObservations(vine_train_samples, dp_ecdfs)
            rvine_struct = GetVineStructure(pseudo_obs,
                                            vine_family_set=vine_family_set,
                                            vine_par_method=vine_par_method,
                                            vine_nonpar_method=vine_nonpar_method,
                                            vine_selcrit=vine_selcrit,
                                            vine_trunc_lvl=vine_trunc_lvl,
                                            vine_tree_crit=vine_tree_crit,
                                            vine_cores=n_cores
                                            )
            if categorical_encoder == 'OHE':
                synth_samples = GetSamplesFromVineOHE(rvine_struct, n_sample=n_sample, col_names=kf_train.columns, decoder=decoder, dp_ecdfs=dp_ecdfs, constant_cols=constant_cols, constant_vals=constant_vals, vine_cores=n_cores)
            else:
                synth_samples = GetSamplesFromVine(rvine_struct, n_sample=n_sample, col_names=kf_train.columns, dp_ecdfs=dp_ecdfs, vine_cores=n_cores)
        if model == 'privbayes':
            if verbose: sys.stdout.write('\r'+f'Fold {f} of {n_folds}: Training PrivBayes Model...                         ')
            Path('./temp').mkdir(parents=True, exist_ok=True)
            kf_train.to_csv(f'./temp/temp_{dataset}.csv', index=False)
            synth_samples = PrivBayes(dataset, num_to_generate=n_sample, dp_eps=dp_epsilon, degree_max=privbayes_degree_max, seed=seed)
            os.remove(f'./temp/temp_{dataset}.csv')
        if model == 'dpcopula':
            if verbose: sys.stdout.write('\r'+f'Fold {f} of {n_folds}: Training DP-Copula Model...                         ')
            Path('./temp').mkdir(parents=True, exist_ok=True)
            kf_train.round(5).to_csv(f'./temp/temp_{dataset}.csv', index=False)
            synth_samples = DPCopula(dataset, dp_eps=dp_epsilon)
            os.remove(f'./temp/temp_{dataset}.csv')
        if model == 'dp-histogram':
            dp_ecdfs = GetECDFs(kf_train, epsilon=dp_epsilon, mechanism=dp_mechanism, GS=dp_global_sens, delta=dp_gaussian_delta)
            random_samples = pd.DataFrame(np.random.uniform(size=(n_sample, kf_train.shape[1])))
            random_samples.columns = kf_train.columns
            synth_samples = TransformToNatualScale(random_samples, dp_ecdfs)
        stop_time = perf_counter()

        # Put back single-valued columns for test framework
        for i in range(len(constant_cols)):
            kf_train[constant_cols[i]] = constant_vals[i]
            synth_samples[constant_cols[i]] = constant_vals[i]

        # Perform the test framework with Ordinal Encoding Data for consistency
        synth_samples, kf_train = decoder(synth_samples), decoder(kf_train)
        encoder = OrdinalEncoder(cols=categorical_attributes).fit(data)
        synth_samples, kf_test, kf_train = encoder.transform(synth_samples), encoder.transform(kf_test), encoder.transform(kf_train)
        
        results_df.loc['Execution time (sec)', f'Fold {f}'] = stop_time - start_time
        
        if verbose: sys.stdout.write('\r'+f'Fold {f} of {n_folds}: Evaluating Statistical Utility...                         ')
        results_df.loc['KL Div', f'Fold {f}'] = KLDivDF(synth_samples, kf_test)
        results_df.loc['KS Dist', f'Fold {f}'] = KSDistDF(synth_samples, kf_test)
        results_df.loc['Max Corr Delta', f'Fold {f}'], results_df.loc['Mean Corr Delta', f'Fold {f}'] = GlobalCorr(synth_samples, kf_test)
        
        if verbose: sys.stdout.write('\r'+f'Fold {f} of {n_folds}: Evaluating Machine Learning Utility...                         ')
        if binary_classif_label is not None: 
            results_df.loc['Binary Class MCC', f'Fold {f}'] = BinaryClassif(synth_samples, kf_test, label=binary_classif_label, n_cores=n_cores)
        if multi_classif_label is not None: 
            results_df.loc['Multi Class MCC', f'Fold {f}'] = MultiClassif(synth_samples, kf_test, label=multi_classif_label, n_cores=n_cores)
        if linearreg_label is not None: 
            results_df.loc['LinReg RMSE', f'Fold {f}'] = LinearRegr(synth_samples, kf_test, label=linearreg_label, n_cores=n_cores)
        
        if verbose: sys.stdout.write('\r'+f'Fold {f} of {n_folds}: Performing Membership Inference Attack...                         ')
        results_df.loc['Privacy Mean', f'Fold {f}'], results_df.loc['Privacy Worst', f'Fold {f}'] = MembershipInferenceAttack(kf_train, kf_test, synth_samples, n=MIA_n, metric=MIA_metric, n_iter=MIA_n_iter)
    
    # Write results to output folder
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f'{output_dir}/kfold_cv_{model}_{dataset}_eps{dp_epsilon}_{categorical_encoder}.csv')
    sys.stdout.write('\r'+f'Fold {f} of {n_folds}: Done!                                                  '+'\n')


