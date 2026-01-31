from scipy.stats import randint,uniform

LIGHTGM_PARAMS={

    'max_depth': randint(10,15),
    'min_child_weight': uniform(1,10),
    'min_child_samples': randint(20,100),
    'num_leaves': randint(20,41,200),
    'learning_rate': uniform(0.01,0.3),
    'colsample_bytree': uniform(0.3,1.0)


    }


RANDOM_SEARCH_PARAMS = {
        'n_estimators' : 300,          
        'random_state' : 42,
        'n_jobs' : -1,
        'verbose' :2,
        'cv' : 2,
        'scoring' : 'f1',         
        'n_iter' : 2         
}
