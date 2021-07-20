import numpy as np
import pandas as pd


def load_data(cluster, opt):
    '''Data Loading & Processing'''
    if opt == "seq":
        df = pd.read_csv(
            f'../data/{cluster}/cluster_sequence.csv', parse_dates=['time'], index_col='time')
    elif opt == "tp":
        df = pd.read_csv(
            f'../data/{cluster}/cluster_throughput.csv', parse_dates=['time'], index_col='time')
    elif opt == "user":
        df = pd.read_pickle(f'../data/{cluster}/cluster_user.pkl')
    elif opt == "log":
        df = pd.read_csv(f'../data/{cluster}/cluster_log.csv',
                         parse_dates=['submit_time', 'start_time', 'end_time'])
    else:
        raise ValueError('Please check opt')

    return df


def set_interval(df, interval, agg):
    df_sampled = df.resample(interval).agg(agg)
    period = {"H": 24, "30min": 48, "10min": 144, "min": 1440}
    return df_sampled, period[interval]


def smape(prediction, test_df):
    """SMAPE symmetric mean absolute percentage error"""
    return (2.0 * np.mean(
        np.abs(prediction - test_df) /
        (np.abs(prediction) + np.abs(test_df))) * 100)
