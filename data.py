# name and path to dataset
TRAIN_DATASETS = {
    'adult': './Datasets/adult.csv',
    'compas': './Datasets/compas.csv',
    'texas_hospital': './Datasets/texas_hospital.csv',
}

# attributes specification
DATASET_CONFIGS = {
    'adult': {
        'datetime': [],
        'categorical': ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
        'binary_classif_label': 'salary',
        'multi_classif_label': 'relationship',
        'linearreg_label': 'age',
    },
    'compas': {
        'datetime': [],
        'categorical': ['sex', 'race', 'score_text', 'charge_degree'],
        'binary_classif_label': 'is_violent_recid',
        'multi_classif_label': 'race',
        'linearreg_label': 'decile_score',
    },
    'texas_hospital': {
        'datetime': [],
        'categorical': ['TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', 'PAT_STATE', 'PAT_ZIP', 'PUBLIC_HEALTH_REGION', 'PAT_STATUS', 'RACE', 'ADMIT_WEEKDAY', 'POA_PROVIDER_INDICATOR', 'ADMITTING_DIAGNOSIS'],
        'binary_classif_label': 'ETHNICITY',
        'multi_classif_label': 'TYPE_OF_ADMISSION',
        'linearreg_label': 'TOTAL_CHARGES',
    },
}