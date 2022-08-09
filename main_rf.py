# Import packages
import numpy as np
import pandas as pd
import random
import os

from generate_data import load_wl_tabular_type
from optuna import Trial

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# Functions
def objectiveRF(trial: Trial, X_tr, y_tr, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [200]),
        'criterion':trial.suggest_categorical(
            'criterion', ['squared_error']),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1),
        # 'max_samples': trial.suggest_float('max_samples', 0.5, 1),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }

    model = MultiOutputRegressor(
        RandomForestRegressor(**params)
        )
    model_fitted = model.fit(X_tr, y_tr)
    
    r2 = r2_score(y_val, model_fitted.predict(X_val))
    rmse = mean_squared_error(y_val, model_fitted.predict(X_val))**0.5
    
    return rmse/r2


def get_rf_optuna(X_tr, y_tr, X_val, y_val, n_trials):
    study = optuna.create_study(
        study_name='rf_params_opt',
        direction='minimize',
        sampler=TPESampler(seed=42),
        )
    
    study.optimize(lambda trial: objectiveRF(trial, X_tr, y_tr, X_val, y_val),
                   n_trials=n_trials,
                   )
    
    best_mdl = RandomForestRegressor(**study.best_params).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_mdl, study


def to_datetime_format(df:pd.DataFrame, col_nm:str='ymdhm', as_index=False) -> pd.DataFrame:
    df[col_nm] = pd.to_datetime(df[col_nm], infer_datetime_format=True) #format = "%Y-%m-%d %H%M")
    df = df.sort_values(by=col_nm, ascending=True)
    if as_index:
        df.set_index(col_nm, inplace=True)
    
    return df 


def fix_random_seed(seed=42):
    import random
    import numpy as np
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Main script
if __name__=='__main__':
    # Import packages
    import pickle
    import optuna
    import time
    
    from multiprocessing import cpu_count
    from sklearn.ensemble import RandomForestRegressor    
    from sklearn.model_selection import train_test_split
    from optuna.samplers import TPESampler
    from tqdm import tqdm
    
    # Log start time
    total_run_time_start = time.time()
    
    # Fix random seed
    fix_random_seed()
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Configuration
    cpu_use = round(2*cpu_count()/3)
    optuna.logging.set_verbosity(0)


    def get_data_and_split(past_step: int, selc_year: list, selc_cols: list):
        if len(selc_year) == 0:
            selc_year = None
   
        X, y, df_smp_subm = load_wl_tabular_type(
            past_step=past_step,
            selc_year=selc_year,
            selc_cols=selc_cols,
            )
    
        # Change dtype oftime column in sample submission DataFrame as datetime type
        df_smp_subm = to_datetime_format(df_smp_subm, as_index=False)
    
        # Split train validation test    
        test_time_start = df_smp_subm.ymdhm[0]
        
        X_test = X[X.index >= test_time_start] # Split test X
        
        X = X[X.index < test_time_start]
        y = y[y.index < test_time_start]
        
        col_X = X.columns
        col_y = y.columns
        
        # Concat X, y for dropna
        concat_Xy = pd.concat([X, y], axis=1)
        concat_Xy = concat_Xy.dropna() # drop null
        
        # Divide X and y after dropna
        X = concat_Xy.loc[:, col_X]
        y = concat_Xy.loc[:, col_y]
        
        # Train validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, 
            test_size=0.2,
            random_state=seed,
            shuffle=True,
            )
        
        # Lower floating point precision
        X_tr = X_tr.astype('float32')
        X_val = X_val.astype('float32')
        X_test = X_test.astype('float32')
        y_tr = y_tr.astype('float32')
        y_val = y_val.astype('float32')

        return X_tr, X_val, y_tr, y_val, X_test
        
    # Hyperparameter search
    N_TRIALS = 30
   
    SELC_YEAR = [2012, 2013, 2016, 2017, 2018, 2020, 2022]
    SELC_COLS = [
        'fw_1018683',
        'fw_1019630'
        ]    
    
    past_step_grid = np.arange(24, 0, -3)
    
    df_step_search = pd.DataFrame(
        columns=['model', 'study', 'score'],
        index=past_step_grid,
        dtype='object',
        )
    df_step_search.index.name = 'backward'
    df_step_search.score = df_step_search.score.astype('float')
    
    for step_back in tqdm(past_step_grid):
        X_tr, X_val, y_tr, y_val, X_test = get_data_and_split(
            step_back,
            SELC_YEAR, 
            SELC_COLS,
            )
        
        best_mdl, study = get_rf_optuna(
            X_tr, y_tr, X_val, y_val, N_TRIALS
            )
        
        df_step_search.loc[step_back, :] = [best_mdl, study, study.best_value]     
        
    # plot - past time consideration gridsearch scoring result
    df_step_search.plot(
        column='score',
        marker='o',
        )
    
    # Print total time
    print(f'Total run time [{(time.time()-total_run_time_start)/60:.1f}]min')
    
    
    
    