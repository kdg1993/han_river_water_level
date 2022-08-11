#%% Import packages
import pandas as pd
import numpy as np

from glob import glob
from copy import deepcopy

# Functions
def to_datetime_format(df:pd.DataFrame, col_nm:str='ymdhm', as_index=False) -> pd.DataFrame:
    df[col_nm] = pd.to_datetime(df[col_nm], infer_datetime_format=True) #format = "%Y-%m-%d %H%M")
    df = df.sort_values(by=col_nm, ascending=True)
    if as_index:
        df.set_index(col_nm, inplace=True)
    
    return df 


def load_wl_tabular_type(
        past_step=3,
        selc_year: list=None, 
        selc_cols: list=None,
        ):   
    
    # Select every year if selc_year is not None
    if selc_year is None:
        # Change element of selc_year as int
        selc_year = list(np.arange(2012, 2023))
    else:
        selc_year = [int(x) for x in selc_year]
    
    # Load sample submission file
    df_smp_subm = pd.read_csv('C:/Everydata/competition_data/sample_submission.csv')
    tgt_cols = list(df_smp_subm)[1:]
    
    # Check elements of      selc_cols are string
    if selc_cols is not None:
        for col in selc_cols:
            if type(col) != str:
                raise TypeError('Element of selc_cols must be string')

    # Get water level data path
    wl_data_paths = sorted(glob("C:/Everydata/competition_data/water_data/*.csv"))

    # Load water level data of each year seperately and set index as datetime format
    df_wl_each = {
        yr: to_datetime_format(pd.read_csv(path), as_index=True) 
        for yr, path in zip(np.arange(2012, 2023), wl_data_paths)
        }

    # Select dataset by year if selc_year is not None
    if selc_year!=None:
        df_wl_each = {k: v for k, v in df_wl_each.items() if k in selc_year}
        
    # Select columns by selc_year if selc_cols is not None
    if selc_cols is None:
        selc_cols = list(df_wl_each[selc_year[0]]) - tgt_cols

    # Divide feature dataframe and target dataframe
    df_feat = {k: v[selc_cols] for k, v in df_wl_each.items()}
    df_tgt = {k: v[tgt_cols] for k, v in df_wl_each.items()}
    
    # Push the target one step
    for k, v in df_tgt.items():
        v.index -= pd.Timedelta(10, unit='min')
        df_tgt[k] = v
    
    # Concatenate t-i time information to the feature data
    for k, v in df_feat.items():
        past_dfs = [v]
        for step in np.arange(1, 1+past_step):
            df_cp = deepcopy(v)
            df_cp.columns = [
                col_nm+f'_t-{step}' for col_nm in df_cp.columns
                ] # add temporal movement information
            df_cp.index += pd.Timedelta(step*10, unit='min') # move time index
            past_dfs.append(df_cp)

        df_feat[k] = pd.concat(past_dfs, axis=1)
        
    # Eliminate data based on past time steps considered
    def eliminate_head_tail(k: int, v: pd.DataFrame):
        if k == 2022:
            return v.iloc[past_step:-(past_step)]
        else:
            return v.iloc[past_step:-(past_step+1)]
    
    df_feat = {k: eliminate_head_tail(k, v) for k, v in df_feat.items()}
    df_tgt = {k: v.iloc[past_step+1:] for k, v in df_tgt.items()}
    
    # Concat by time
    df_feat = pd.concat([v for k, v in df_feat.items()], axis=0)
    df_tgt = pd.concat([v for k, v in df_tgt.items()], axis=0)
    
    return df_feat, df_tgt, df_smp_subm

#%% Test
X, y, smp_sub = load_wl_tabular_type(
    past_step=4,
    selc_year=[2012, 2013, 2016, 2017, 2018, 2020, 2022],
    selc_cols=['fw_1018683', 'fw_1019630'],
    )

# X = X[2022]
# y = y[2022]

c = pd.concat([X, y], axis=1)

