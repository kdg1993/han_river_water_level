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
def objectiveLGBM_kfold(trial: Trial, X_tr, y_tr, X_val, y_val):
    params = {
        'boosting_type': trial.suggest_categorical(
            'boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'max_depth': trial.suggest_int('max_depth', 1, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
        'subsample': trial.suggest_float('subsample', 0.3, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }

    model = MultiOutputRegressor(
        LGBMRegressor(**params)
        )

    # Concat X, y redefine X_tr due to the memory
    X = pd.concat([X_tr, X_val], axis=0)
    y = pd.concat([y_tr, y_val], axis=0)

    del X_tr, X_val, y_tr, y_val # due to the memory

    # Calculate KFold mean score
    kfold_scores = []

    kf = KFold(n_splits=4) # Split 4 because data seems enough and get large val size
    kf.get_n_splits(X)
    for tr_idx, val_idx in kf.split(X):
        X_tr_temp, X_val_temp = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_temp, y_val_temp = y.iloc[tr_idx], y.iloc[val_idx]  

        model_fitted = model.fit(X_tr_temp, y_tr_temp)

        r2_temp = r2_score(y_val_temp, model_fitted.predict(X_val_temp))
        rmse_temp = mean_squared_error(y_val_temp, model_fitted.predict(X_val_temp))**0.5
    
        if r2_temp < 0:
            r2_temp = 1e-10

        score_temp = rmse_temp/r2_temp
        kfold_scores.append(score_temp)

    return np.mean(kfold_scores)


def get_lgbm_optuna_kfold(X_tr, y_tr, X_val, y_val, n_trials):
    study = optuna.create_study(
        study_name='LightGBM_params_opt_by_kfold',
        direction='minimize',
        sampler=TPESampler(seed=42),
        )
    
    study.optimize(lambda trial: objectiveLGBM_kfold(trial, X_tr, y_tr, X_val, y_val),
                   n_trials=n_trials,
                   )
    
    best_mdl = MultiOutputRegressor(
        LGBMRegressor(**study.best_params)
        ).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_mdl, study


def objectiveRF_kfold(trial: Trial, X_tr, y_tr, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100]),
        'max_depth': trial.suggest_int('max_depth', 1, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'max_features': trial.suggest_float('max_features', 0.3, 1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }

    model = MultiOutputRegressor(
        RandomForestRegressor(**params)
        )

    # Concat X, y redefine X_tr due to the memory
    X = pd.concat([X_tr, X_val], axis=0)
    y = pd.concat([y_tr, y_val], axis=0)

    del X_tr, X_val, y_tr, y_val # due to the memory

    # Calculate KFold mean score
    kfold_scores = []

    kf = KFold(n_splits=4) # Split 4 because data seems enough and get large val size
    kf.get_n_splits(X)
    for tr_idx, val_idx in kf.split(X):
        X_tr_temp, X_val_temp = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_temp, y_val_temp = y.iloc[tr_idx], y.iloc[val_idx]  

        model_fitted = model.fit(X_tr_temp, y_tr_temp)

        r2_temp = r2_score(y_val_temp, model_fitted.predict(X_val_temp))
        rmse_temp = mean_squared_error(y_val_temp, model_fitted.predict(X_val_temp))**0.5
    
        if r2_temp < 0:
            r2_temp = 1e-10

        score_temp = rmse_temp/r2_temp
        kfold_scores.append(score_temp)

    return np.mean(kfold_scores)


def get_rf_optuna_kfold(X_tr, y_tr, X_val, y_val, n_trials):
    study = optuna.create_study(
        study_name='RandomForestRegressor_params_opt_by_kfold',
        direction='minimize',
        sampler=TPESampler(seed=42),
        )
    
    study.optimize(lambda trial: objectiveRF_kfold(trial, X_tr, y_tr, X_val, y_val),
                   n_trials=n_trials,
                   )
    
    best_mdl = MultiOutputRegressor(
        RandomForestRegressor(**study.best_params)
        ).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_mdl, study


def objectiveETR_kfold(trial: Trial, X_tr, y_tr, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100]),
        'max_depth': trial.suggest_int('max_depth', 1, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }

    model = MultiOutputRegressor(
        ExtraTreesRegressor(**params)
        )

    # Concat X, y redefine X_tr due to the memory
    X = pd.concat([X_tr, X_val], axis=0)
    y = pd.concat([y_tr, y_val], axis=0)

    del X_tr, X_val, y_tr, y_val # due to the memory

    # Calculate KFold mean score
    kfold_scores = []

    kf = KFold(n_splits=4) # Split 4 because data seems enough and get large val size
    kf.get_n_splits(X)
    for tr_idx, val_idx in kf.split(X):
        X_tr_temp, X_val_temp = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_temp, y_val_temp = y.iloc[tr_idx], y.iloc[val_idx]  

        model_fitted = model.fit(X_tr_temp, y_tr_temp)

        r2_temp = r2_score(y_val_temp, model_fitted.predict(X_val_temp))
        rmse_temp = mean_squared_error(y_val_temp, model_fitted.predict(X_val_temp))**0.5
    
        if r2_temp < 0:
            r2_temp = 1e-10

        score_temp = rmse_temp/r2_temp
        kfold_scores.append(score_temp)

    return np.mean(kfold_scores)


def get_etr_optuna_kfold(X_tr, y_tr, X_val, y_val, n_trials):
    study = optuna.create_study(
        study_name='ExtraTreesRegressor_params_opt_by_kfold',
        direction='minimize',
        sampler=TPESampler(seed=42),
        )
    
    study.optimize(lambda trial: objectiveETR_kfold(trial, X_tr, y_tr, X_val, y_val),
                   n_trials=n_trials,
                   )
    
    best_mdl = MultiOutputRegressor(
        ExtraTreesRegressor(**study.best_params)
        ).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_mdl, study


def objectiveETR(trial: Trial, X_tr, y_tr, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [200]),
        'max_depth': trial.suggest_int('max_depth', 1, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }

    model = MultiOutputRegressor(
        ExtraTreesRegressor(**params)
        )
    model_fitted = model.fit(X_tr, y_tr)
    
    r2 = r2_score(y_val, model_fitted.predict(X_val))
    rmse = mean_squared_error(y_val, model_fitted.predict(X_val))**0.5
    
    if r2 < 0:
        r2 = 1e-10
    
    return rmse/r2


def get_etr_optuna(X_tr, y_tr, X_val, y_val, n_trials):
    study = optuna.create_study(
        study_name='ExtraTreesRegressor_params_opt',
        direction='minimize',
        sampler=TPESampler(seed=42),
        )
    
    study.optimize(lambda trial: objectiveETR(trial, X_tr, y_tr, X_val, y_val),
                   n_trials=n_trials,
                   )
    
    best_mdl = MultiOutputRegressor(
        ExtraTreesRegressor(**study.best_params)
        ).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_mdl, study


def objectiveXGBRF(trial: Trial, X_tr, y_tr, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100]),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        # 'subsample': trial.suggest_float('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1),
        # 'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        # 'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }

    model = MultiOutputRegressor(
        XGBRFRegressor(**params)
        )
    model_fitted = model.fit(X_tr, y_tr)
    
    r2 = r2_score(y_val, model_fitted.predict(X_val))
    rmse = mean_squared_error(y_val, model_fitted.predict(X_val))**0.5
    
    if r2 < 0:
        r2 = 1e-10
    
    return rmse/r2


def get_xgbrf_optuna(X_tr, y_tr, X_val, y_val, n_trials):
    study = optuna.create_study(
        study_name='xgboostRF_params_opt',
        direction='minimize',
        sampler=TPESampler(seed=42),
        )
    
    study.optimize(lambda trial: objectiveXGBRF(trial, X_tr, y_tr, X_val, y_val),
                   n_trials=n_trials,
                   )
    
    best_mdl = MultiOutputRegressor(
        XGBRFRegressor(**study.best_params)
        ).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_mdl, study


def objectiveLGBM(trial: Trial, X_tr, y_tr, X_val, y_val):
    params = {
        'boosting_type': trial.suggest_categorical(
            'boosting_type', ['gbdt', 'dart', 'goss']),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }

    model = MultiOutputRegressor(
        LGBMRegressor(**params)
        )
    model_fitted = model.fit(X_tr, y_tr)
    
    r2 = r2_score(y_val, model_fitted.predict(X_val))
    rmse = mean_squared_error(y_val, model_fitted.predict(X_val))**0.5
    
    if r2 < 0:
        r2 = 1e-10
    
    return rmse/r2


def get_lgbm_optuna(X_tr, y_tr, X_val, y_val, n_trials):
    study = optuna.create_study(
        study_name='lightgbm_params_opt',
        direction='minimize',
        sampler=TPESampler(seed=42),
        )
    
    study.optimize(lambda trial: objectiveLGBM(trial, X_tr, y_tr, X_val, y_val),
                   n_trials=n_trials,
                   )
    
    best_mdl = MultiOutputRegressor(
        LGBMRegressor(**study.best_params)
        ).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_mdl, study


def objectiveRF(trial: Trial, X_tr, y_tr, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [200]),
        'criterion':trial.suggest_categorical(
            'criterion', ['squared_error']),
        'max_depth': trial.suggest_int('max_depth', 3, 40),
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
    
    if r2 < 0:
        r2 = 1e-10
    
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
    
    best_mdl = MultiOutputRegressor(
        RandomForestRegressor(**study.best_params)
        ).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_mdl, study


def get_data_and_split(past_step: int, selc_year: list, selc_cols: list,
                       drop_na_in_X=True, random_seed=42,
                       na_fill_by_ffill=False, na_fill_limit: int=None, 
                       na_fill_by_int=False, na_filling_int=None):
    if na_fill_by_int is True:
        if na_filling_int is None:
            raise NotImplementedError('NA fill by int is True but no int is assigned')
        else:
            pass


    if len(selc_year) == 0:
        selc_year = None
   
    X, y, df_smp_subm = load_wl_tabular_type(
        past_step=past_step,
        selc_year=selc_year,
        selc_cols=selc_cols,
        )

    # Fill missing values as int if option is True
    if na_fill_by_int:
        X.fillna(na_filling_int, inplace=True)

    # Change dtype oftime column in sample submission DataFrame as datetime type
    df_smp_subm = to_datetime_format(df_smp_subm, as_index=False)

    # Split train validation test    
    test_time_start = df_smp_subm.ymdhm[0]
    
    X_test = X[X.index >= test_time_start] # Split test X
    
    X = X[X.index < test_time_start]
    y = y[y.index < test_time_start]
    
    # Concat X, y for dropna
    col_X = X.columns
    col_y = y.columns
    
    if drop_na_in_X:        
        concat_Xy = pd.concat([X, y], axis=1)
        concat_Xy = concat_Xy.dropna() # drop null
    else:
        concat_Xy = pd.concat([X, y], axis=1)
        # drop null only if null in y
        concat_Xy = concat_Xy.loc[y.dropna().index, :]
        
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
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor    
    from sklearn.model_selection import train_test_split, KFold
    from optuna.samplers import TPESampler
    from tqdm import tqdm
    from datetime import datetime
    from lightgbm import LGBMRegressor
    from xgboost import XGBRFRegressor
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
    optuna.logging.set_verbosity(2)
    
    model_nm = 'lgbm_kfold' # rf, lgbm, xgbrf, etr, xgbrf_na_fill, etr_kfold, rf_kfold, lgbm_kfold
    
    save_submission = False
    print(f'\nSave submission is [{save_submission}]\n')
        
    # Hyperparameter search
    N_TRIALS = 100
   
    SELC_YEAR = [2012, 2013, 2016, 2017, 2018, 2020, 2022]
    SELC_COLS = [
        # 'swl',
        # 'inf', 
        # 'sfw',
        # 'ecpc',
        # 'tototf',
        # 'tide_level',
        'fw_1018662',
        'fw_1018683',
        'fw_1019630',
        ]    
    DROP_NA_IN_X = False
    NA_FILL_BY_FFILL = True
    NA_FILL_LIMIT = None
    NA_FILL_BY_INT = True
    NA_FILLING_INT = -9999
    
    past_step_grid = [6, 12, 18]
    
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
            drop_na_in_X=DROP_NA_IN_X,
            random_seed=seed,
            na_fill_by_ffill=NA_FILL_BY_FFILL,
            na_fill_limit=NA_FILL_LIMIT,
            na_fill_by_int=NA_FILL_BY_INT,
            na_filling_int=NA_FILLING_INT,
            )
        
        if model_nm == 'rf':
            best_mdl, study = get_rf_optuna(X_tr, y_tr, X_val, y_val, N_TRIALS)
        elif model_nm == 'lgbm':
            best_mdl, study = get_lgbm_optuna(X_tr, y_tr, X_val, y_val, N_TRIALS)
        elif model_nm == 'xgbrf':
            best_mdl, study = get_xgbrf_optuna(X_tr, y_tr, X_val, y_val, N_TRIALS)
        elif model_nm == 'etr':
            best_mdl, study = get_etr_optuna(X_tr, y_tr, X_val, y_val, N_TRIALS)
        elif model_nm == 'etr_kfold':
            best_mdl, study = get_etr_optuna_kfold(X_tr, y_tr, X_val, y_val, N_TRIALS)
        elif model_nm == 'rf_kfold':
            best_mdl, study = get_rf_optuna_kfold(X_tr, y_tr, X_val, y_val, N_TRIALS)
        elif model_nm == 'lgbm_kfold':
            best_mdl, study = get_lgbm_optuna_kfold(X_tr, y_tr, X_val, y_val, N_TRIALS)
        else:
            raise NotImplementedError('Wrong model name to select')
            
        
        df_step_search.loc[step_back, :] = [best_mdl, study, study.best_value]     
        
    #%% plot - past time consideration gridsearch scoring result
    df_step_search.plot(
        column='score',
        marker='o',
        grid=True,
        )
    
    #%% Get best model and predict test time
    # get best model and window size
    mdl_best = df_step_search.sort_values(by='score').model.iat[0]
    best_past_step = df_step_search.sort_values(by='score').index[0]
    
    # Load best time window sized data    
    _, _, _, _, X_test = get_data_and_split(
        past_step=best_past_step,
        selc_year=SELC_YEAR,
        selc_cols=SELC_COLS,
        drop_na_in_X=DROP_NA_IN_X,
        random_seed=seed,
        na_fill_by_ffill=NA_FILL_BY_FFILL,
        na_fill_limit=NA_FILL_LIMIT,
        na_fill_by_int=NA_FILL_BY_INT,
        na_filling_int=NA_FILLING_INT,
        )

    # Change dtype oftime column in sample submission DataFrame as datetime type
    df_smp_subm = pd.read_csv('data/sample_submission.csv')    
    
    # Predict test time
    y_test = mdl_best.predict(X_test)
    
    # Merge y_test into submission shape
    df_smp_subm.iloc[:, 1:] = y_test
    
    #%% Save model and submission after making directory
    if not os.path.isdir('./submission'):
        os.mkdir('submission')
    
    if save_submission:
        # get current time
        current_time_str = \
            f"""_{datetime.now().month}_{datetime.now().day}_{datetime.now().hour}_{datetime.now().minute}"""
        
        # Make directory for one submission
        submit_dir_nm = 'time'+current_time_str
        try:
            save_path = os.path.join('submission', submit_dir_nm)
            os.mkdir(save_path)
        except:
            pass
        
        # Save model and prediction csv
        df_smp_subm.to_csv(save_path+'/submission.csv', index=None)
        
        with open(save_path+'/model', 'wb') as f:
            pickle.dump(mdl_best, f)
            
    
    # Print total time
    print(f'Total run time [{(time.time()-total_run_time_start)/60:.1f}]min')
    
    
    
    
    
    
    
    
    