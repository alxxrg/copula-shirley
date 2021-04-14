# python main.py --dataset='adult' --model='cop-shirl' --categorical-encoder=None --cat-encoder-target=None --dp-epsilon=1.0 --dp-mechanism='Laplace' --vine-sample-ratio=0.6 --vine-family-set='all' --vine-trunc-lvl=None --vine-tree-crit='tau' 


import os
import argparse
from pathlib import Path
from time import perf_counter

from data import TRAIN_DATASETS, DATASET_CONFIGS
from preprocess import *
from transform import *
from vine import *
from privbayes import *
from dpcopula import * 

parser = argparse.ArgumentParser(description='Synthetic Data Generation')

parser.add_argument('--dataset', default='adult', choices=list(TRAIN_DATASETS.keys()))
parser.add_argument('--model', default='cop-shirl', choices=['cop-shirl', 'privbayes', 'dpcopula', 'dp-histogram'])
parser.add_argument('--seed', type=int, default=76543)
parser.add_argument('--n-sample', type=int, default=None)

parser.add_argument('--categorical-encoder', default=None, choices=('ORD', 'WOE', 'GLMM', 'OHE'))
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
parser.add_argument('--vine-tree-crit', type=str, default='tau')

parser.add_argument('--privbayes-degree-max', type=int, default=3)

parser.add_argument('--output-dir', type=str, default='./out')

parser.add_argument('--n-cores', type=int, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    output_dir = args.output_dir
    n_cores = args.n_cores
    if n_cores is None:
        n_cores = os.cpu_count()
    
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
    
    data = pd.read_csv(f'./Datasets/{dataset}.csv', skipinitialspace=True)
    if not dp_epsilon:
        clean_data, decoder = PreprocessData(data, cat_encoder=categorical_encoder, cat_attr=categorical_attributes, datetime_attr=datetime_attributes, enc_target=cat_encoder_target)
    else:
        train_index = data.sample(frac=0.8).index
        encoder_index = data.drop(train_index).index
        clean_data, decoder = PreprocessData(data, train_idx=train_index, test_idx=encoder_index, cat_encoder=categorical_encoder, cat_attr=categorical_attributes, datetime_attr=datetime_attributes, enc_target=cat_encoder_target)
    
    if n_sample is None: n_sample = data.shape[0]

    start_time = perf_counter()
    if model == 'cop-shirl':
        if dp_epsilon: vine_train_samples, ecdf_train_samples = SampleForVine(clean_data, vine_sample_ratio, 2)
        else: vine_train_samples = ecdf_train_samples = clean_data
        
        dp_ecdfs = GetECDFs(ecdf_train_samples, epsilon=dp_epsilon, mechanism=dp_mechanism, GS=dp_global_sens, delta=dp_gaussian_delta)
        pseudo_obs = TransformToPseudoObservations(vine_train_samples, dp_ecdfs)
        print('================ Constructing Vine Copula Model ================')
        rvine_struct = GetVineStructure(pseudo_obs,
                                        vine_family_set=vine_family_set,
                                        vine_par_method=vine_par_method,
                                        vine_nonpar_method=vine_nonpar_method,
                                        vine_selcrit=vine_selcrit,
                                        vine_trunc_lvl=vine_trunc_lvl,
                                        vine_tree_crit=vine_tree_crit,
                                        vine_cores=n_cores
                                        )
        print('================ Sampling from Vine Copula Model ================')
        if categorical_encoder == 'OHE':
            synth_samples = GetSamplesFromVineOHE(rvine_struct, n_sample=n_sample, col_names=clean_data.columns, decoder=decoder, dp_ecdfs=dp_ecdfs, constant_cols=constant_cols, constant_vals=constant_vals, vine_cores=n_cores)
        else:
            synth_samples = GetSamplesFromVine(rvine_struct, n_sample=n_sample, col_names=clean_data.columns, dp_ecdfs=dp_ecdfs, vine_cores=n_cores)
    if model == 'privbayes':
        clean_data.to_csv(f'./temp/temp_{dataset}.csv', index=False)
        synth_samples = PrivBayes(dataset, num_to_generate=n_sample, dp_eps=dp_epsilon, degree_max=privbayes_degree_max, seed=seed)
        synth_samples.drop('id', axis=1, inplace=True)
    if model == 'dpcopula':
        clean_data.round(5).to_csv(f'./temp/temp_{dataset}.csv', index=False)
        synth_samples = DPCopula(dataset, dp_eps=dp_epsilon)
    if model == 'dp-histogram':
        dp_ecdfs = GetECDFs(clean_data, epsilon=dp_epsilon, mechanism=dp_mechanism, GS=dp_global_sens, delta=dp_gaussian_delta)
        random_samples = pd.DataFrame(np.random.uniform(size=(n_sample, clean_data.shape[1])), columns=clean_data.columns)
        synth_samples = TransformToNatualScale(random_samples, dp_ecdfs)
        
    stop_time = perf_counter()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    decoder(synth_samples).to_csv(f'{output_dir}/{dataset}_{model}_eps{dp_epsilon}.csv', index=False)
    print(f"Done in {stop_time - start_time:0.4f} seconds")



    
