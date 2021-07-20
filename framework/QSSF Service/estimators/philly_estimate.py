import pandas as pd
import datetime
import random
random.seed = 6


def trace_philly_process(dir, date_range):
    start = '2017-10-01 00:00:00'
    df = pd.read_csv(dir+'/cluster_log.csv', parse_dates=['submit_time'], usecols=['user', 'vc', 'jobname', 'gpu_num',
                                                                                   'state', 'submit_time', 'duration'])
    # Consider gpu jobs only
    df = df[df['gpu_num'] > 0]

    # VC filter
    vc_dict = pd.read_pickle(dir+'/vc_dict_homo.pkl')
    vc_list = vc_dict.keys()
    df = df[df['vc'].isin(vc_list)]

    df = df[df['submit_time'] >= pd.Timestamp(start)]
    df['submit_time'] = df['submit_time'].apply(
        lambda x: int(datetime.datetime.timestamp(pd.Timestamp(x))))

    df['state'] = df['state'].replace('Pass', 'COMPLETED')
    df['state'] = df['state'].replace('Failed', 'FAILED')
    df['state'] = df['state'].replace('Killed', 'CANCELLED')

    # Normalizing
    df['submit_time'] = df['submit_time'] - df.iloc[0]['submit_time']

    # Slicing simulation part
    begin = (pd.Timestamp(date_range[0])-pd.Timestamp(start)).total_seconds()
    end = (pd.Timestamp(date_range[1])-pd.Timestamp(start)).total_seconds()
    df = df[(df['submit_time'] >= begin) & (df['submit_time'] <= end)]

    df.sort_values(by='submit_time', inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def priority_generater(x):
    '''Use Helios estimate distribution to generate Philly priority
    10%   error <= 10%
    50%   error <= 50%
    70%   error <= 100%
    90%   error <= 1000%
    100%  error <= 10000%
    '''
    level = random.uniform(0, 1)

    if level <= 0.1:
        return x + x * random.uniform(-1, 1) * 0.1
    elif level <= 0.5:
        return x + x * random.uniform(0.1, 0.5) * random.choice((-1, 1))
    elif level <= 0.7:
        return x + x * random.uniform(0.5, 1) * random.choice((-1, 1))
    elif level <= 0.9:
        return x + x * random.uniform(1, 10) * random.choice((-1, 1))
    else:
        return x + x * random.uniform(10, 100) * random.choice((-1, 1))


df = trace_philly_process(
    '../data/Philly', ('2017-10-01 00:00:00', '2017-11-30 23:59:00'))
df['priority'] = df['duration'].apply(priority_generater)
df['priority'] = df['priority'].astype(int)
df.to_csv('Philly_lgb.csv', index=False)
