# from statsmodels.tsa.stattools import arma_order_select_ic
# from statsmodels.tsa.stattools import adfuller as ADF
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from sklearn.model_selection import GroupKFold, cross_val_predict

# import xgboost as xgb
# import torch.nn as nn
# import torch
# from torch.autograd import Variable

# from fbprophet import Prophet

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics, preprocessing
import warnings
import chinese_calendar
import utils

import datetime
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

matplotlib.font_manager._rebuild()

sns.set_style("ticks")
font = {
    "font.family": "Roboto",
    "font.size": 12,
}
sns.set_style(font)
paper_rc = {
    "lines.linewidth": 3,
    "lines.markersize": 10,
}
sns.set_context("paper", font_scale=1.6, rc=paper_rc)
current_palette = sns.color_palette()

warnings.filterwarnings("ignore")

cluster_list = ["Venus", "Earth", "Saturn", "Uranus"]
cluster = cluster_list[1]
df = pd.read_csv(
    f"./data/{cluster}/cluster_node_used_gpu_count.csv",
    parse_dates=["time"],
    index_col="time",
)

date_range = ("2020-04-01 00:00:00", "2020-09-28 23:50:00")
pred_range = ("2020-09-01 00:00:00", "2020-09-22 00:00:00")

df["Used Node"] = df.loc[:, "Use 1 GPU":"Use 8 GPU"].sum(axis=1)
df, period = utils.set_interval(df, "10min", "max")
train_df = df["Used Node"][df.index < pred_range[0]].copy()
test_df = df["Used Node"][
    (df.index < pred_range[1]) & (df.index >= pred_range[0])
].copy()


def hyperparameters(cluster):
    if cluster == 'Venus':
        history, future = 3, 18
        his_threshold = 0.5
        fut_threshold = 1
        de_threshold = 2

        buffer = 5
        keep = 12
        stable_pred = 0

    if cluster == 'Earth':
        history, future = 6, 18
        his_threshold = 1
        fut_threshold = 1
        de_threshold = 5

        buffer = 5
        keep = 12
        stable_pred = 1

    if cluster == 'Saturn':
        history, future = 6, 18
        his_threshold = 1
        fut_threshold = 1
        de_threshold = 1

        buffer = 5
        keep = 12
        stable_pred = 0

    if cluster == 'Uranus':
        history, future = 3, 18
        his_threshold = 1
        fut_threshold = 1
        de_threshold = 1

        buffer = 7
        keep = 12
        stable_pred = 1

    return history, future, his_threshold, fut_threshold, de_threshold, buffer, keep, stable_pred


def plot_node_predict(pred, test, cluster, save=False):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax = pred.plot(linestyle='--', alpha=0.8, label="Prediction", linewidth=2)
    ax = test.plot(linestyle='--', alpha=0.8,
                   label="Ground Truth", linewidth=2)

    ax.set_xlabel(f"Dates in September")
    ax.set_ylabel(f"Running Node Number")
    # ax.set_xticks(test_df.index[::432])
    # ax.set_xticklabels(arange(1, 25, 3))
    # ax.set_xlim(-1, 24)
    # ax.set_ylim(66, 88)
    ax.grid(axis="y", linestyle=":")
    ax.legend(
        loc="lower right",
        # bbox_to_anchor=(0.5, 1.24),
        # ncol=4,
        # fancybox=True,
        # shadow=True,
        # fontsize=14.5,
    )
    plt.show()


def eval_node_predict(pred, test, cluster, save=False):
    fig, ax = plt.subplots(figsize=(8, 3))
    error = (pred - test) / pred * 100
    error = error.values
    ax.hist(
        error,
        30,
        density=True,
        stacked=True,
        linewidth=2,
        histtype="step",
        range=(-100, 100),
        alpha=0.75,
    )

    ax.set_xlabel("Estimate error(%)")
    ax.set_ylabel(f"Probability")
    ax.set_ylim(0)
    plt.show()


def feature_engineering(train_df):
    data = train_df.reset_index()

    # Time Features
    time_features = ["year", "month", "day",
                     "hour", "minute", "dayofweek", "dayofyear"]
    for tf in time_features:
        data[tf] = getattr(data["time"].dt, tf).astype(np.int16)
    data["week"] = data["time"].dt.isocalendar(
    ).week.astype(np.int16)  # weekofyear

    # Rolling data
    window = period // 24  # shift 1 day
    periods = [period // 48, period // 24]  # 30min, 1hour, 3hours, 1day
    for p in periods:
        data["rolling_day_mean_" + str(p)] = data["Used Node"].transform(
            lambda x: x.shift(window).rolling(p).mean()
        )
        data["rolling_day_median_" + str(p)] = data["Used Node"].transform(
            lambda x: x.shift(window).rolling(p).median()
        )
        data["rolling_day_std_" + str(p)] = data["Used Node"].transform(
            lambda x: x.shift(window).rolling(p).std()
        )

    #     # Rolling data   last day
    #     window = period #shift 1 day
    #     periods = [period // 48, period // 24, period // 8, period] # 30min, 1hour, 3hours, 1day
    #     for p in periods:
    #         data['rolling_day_mean_' + str(p)] = data['Used Node'].transform(lambda x: x.shift(window).rolling(p).mean())
    #         data['rolling_day_median_' + str(p)] = data['Used Node'].transform(lambda x: x.shift(window).rolling(p).median())
    #         data['rolling_day_std_' + str(p)] = data['Used Node'].transform(lambda x: x.shift(window).rolling(p).std())

    #     data['1_day_ago'] = data['Used Node'].transform(lambda x: x.shift(period))
    #     data['3_day_ago'] = data['Used Node'].transform(lambda x: x.shift(period*3))
    #     data['7_day_ago'] = data['Used Node'].transform(lambda x: x.shift(period*7))

#         data['10min_ago'] = data['Used Node'].transform(lambda x: x.shift(1))
#     data['30min_ago'] = data['Used Node'].transform(lambda x: x.shift(period//48))
#     data["1h_ago"] = data["Used Node"].transform(lambda x: x.shift(period // 24))
    data["3h_ago"] = data["Used Node"].transform(
        lambda x: x.shift(period // 8))

    def soft_avg(data, t):
        avg = 0.5 * data.transform(lambda x: x.shift(t))
        avg += 0.25 * (
            data.transform(lambda x: x.shift(t + 1))
            + data.transform(lambda x: x.shift(t - 1))
        )
        return avg

#     data["1h_ago_soft"] = soft_avg(data["Used Node"], period // 24)
    data["3h_ago_soft"] = soft_avg(data["Used Node"], period // 8)

    #     # Rolling data   last week
    #     window = period * 7 #shift 1 day
    #     periods = [6, 18, period] # 1hour, 3hours, 1day
    #     for p in periods:
    #         data['rolling_week_mean_' + str(p)] = data['Used Node'].transform(lambda x: x.shift(window).rolling(p).mean())
    #         data['rolling_week_std_' + str(p)] = data['Used Node'].transform(lambda x: x.shift(window).rolling(p).std())

    # Holiday & Event
    data["holiday"] = (
        data["time"].apply(
            lambda x: chinese_calendar.is_holiday(x)).astype(int)
    )

    events = {"2020-03-06": "ECCV",
              "2020-06-06": "NeurIPS", "2020-09-10": "AAAI"}
    event_influence = 10
    event_days = []
    for d in events:
        day = datetime.datetime.strptime(d, "%Y-%m-%d")
        for i in range(1, event_influence + 1):
            date = day - timedelta(days=i)
            event_days.append(date.date())
    data["event"] = data["time"].apply(
        lambda x: 1 if x.date() in event_days else 0)

    return data


train_data = feature_engineering(train_df)
test_data = feature_engineering(test_df)

lgb_params = {
    #     'objective': 'poisson',
    #         'num_iterations': 10000,
    #         'boosting_type': 'gbdt',
    #         'n_jobs': -1,
    #         'seed': 66,
    #         'learning_rate': 0.1,
    "metric": "rmse",
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    #         'colsample_bytree': 0.85,
    #         'colsample_bynode': 0.85,
    #         'min_data_per_leaf': 25,
    #         'num_leaves': 200,
    #         'lambda_l1': 0.5,
    #         'lambda_l2': 0.5
}

cat_features = ["event", "holiday"]
# kf = GroupKFold(3)

lgb_train = lgb.Dataset(
    train_data[train_data["month"] < 8].iloc[:, 2:],
    train_data[train_data["month"] < 8]["Used Node"],
    categorical_feature=cat_features,
)
lgb_eval = lgb.Dataset(
    train_data[train_data["month"] == 8].iloc[:, 2:],
    train_data[train_data["month"] == 8]["Used Node"],
    categorical_feature=cat_features,
    reference=lgb_train,
)

model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=lgb_eval,
    early_stopping_rounds=200,
    num_boost_round=10000,
)

prediction = model.predict(test_data.iloc[:, 2:])
prediction = pd.Series(prediction, name="Used Node", index=test_df.index)

rmse_LGBM = metrics.mean_squared_error(test_df, prediction, squared=False)
smape_LGBM = utils.smape(prediction, test_df)
print(f"rmse_LGBM: {rmse_LGBM}, smape_LGBM: {smape_LGBM}")

plot_node_predict(prediction, test_df, cluster, save=False)
eval_node_predict(prediction, test_df, cluster, save=False)


def plot_sim_node(sim, save=False):
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(sim))

    ax.plot(x, sim["Used Node"].values, linestyle='--',
            alpha=0.9, label="Running", linewidth=2)
    ax.plot(x, sim["Prediction"].values, linestyle='--',
            alpha=0.9, label="Prediction", linewidth=2)
    ax.plot(x, sim["Active"].values, linestyle='-',
            alpha=0.9, label="Active", linewidth=2)
    ax.plot(x, sim["All Node Number"].values, linestyle='-',
            alpha=0.9, label="Total", linewidth=2)
    ax.set_xlabel(f"Dates in September")
    ax.set_ylabel(f"GPU Node Number")

    tick_interval = 144
    ax.set_xticks(x[::tick_interval])
    ax.set_xticklabels(sim.index.day[::tick_interval])
    ax.set_xlim(0, len(sim))
    ax.grid(axis="y", linestyle=":")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=4,
        fancybox=True,
        shadow=True,
        fontsize=14,
    )

    if save:
        fig.savefig(
            f"./helios_ces.pdf", bbox_inches="tight", dpi=600,
        )


def cdf_node_usage(sim, cluster, save=False):
    p1 = plt.figure(figsize=(8, 3))

    plt.hist(
        sim["Used rate"].values,
        10000,
        density=True,
        histtype="step",
        cumulative=True,
        linewidth=2,
        range=(50, 100),
        alpha=0.75,
    )
    plt.hist(
        sim["Optimized rate"].values,
        10000,
        density=True,
        histtype="step",
        cumulative=True,
        linewidth=2,
        range=(50, 100),
        alpha=0.75,
    )

    title = f"{cluster} CDF of Node Usage".replace("_", " ")
    plt.title(title, fontsize=15)
    plt.legend(
        ["Nodes Utilization", "Optimized Nodes Utilization"],
        fontsize=15,
        loc="upper left",
    )
    plt.xlabel("Node Usage(\%)", fontsize=15)
    plt.ylabel("CDF", fontsize=15)
    plt.ylim(0)


"""Simulation"""
sim = df[["All Node Number", "Used Node"]][
    (df.index < pred_range[1]) & (df.index >= pred_range[0])
]
sim["Prediction"] = prediction
sim[["Active", "Poweroff", "Change", "Lack"]] = 0
sim = sim.apply(np.floor).astype(int)

# select specific period for analyze
start = 1
end = 21
days = end - start + 1
sim = sim.loc[
    (sim.index.date >= datetime.date(2020, 9, start))
    & (sim.index.date <= datetime.date(2020, 9, end))
]

history, future, his_threshold, fut_threshold, de_threshold, buffer, keep, stable_pred = hyperparameters(
    cluster)

for i in range(len(sim)):
    # Initial first hour of simulation
    if i < 6:
        sim["Active"][i] = sim["Used Node"][i] + buffer
        continue

    '''Increase Node'''
    # start servers if current not enough
    sim["Active"][i] = sim["Active"][i - 1]
    if sim["Active"][i] < sim["Used Node"][i]:
        sim["Lack"][i] = sim["Used Node"][i] - sim["Active"][i]
        sim["Active"][i] = sim["Used Node"][i] + buffer
        sim["Change"][i] = sim["Active"][i] - sim["Active"][i - 1]
        stable_pred = i + keep  # keep nodes number stable

    # Aviod active server more than the total number
    sim["Active"][i] = (
        sim["All Node Number"][i]
        if sim["Active"][i] > sim["All Node Number"][i]
        else sim["Active"][i]
    )

    '''Decrease Node'''
    history_trend = np.mean(
        sim["Used Node"][i - history: i]) - sim["Used Node"][i]
    future_trend = sim["Used Node"][i] - \
        np.mean(sim["Prediction"][i: i + future])
    if history_trend >= his_threshold and future_trend >= fut_threshold:
        if i < stable_pred:
            continue
        else:
            stable_pred = i

        if np.mean(sim["Prediction"][i: i + future]) < np.mean(
            sim["Prediction"][i: i + 2 * future]
        ):
            continue

        decrease = sim["Active"][i] - \
            min(sim["Active"][i], sim["Used Node"][i] + buffer)
        if decrease >= de_threshold:
            sim["Active"][i] = min(
                sim["Active"][i], sim["Used Node"][i] + buffer)
        else:
            continue
        sim["Change"][i] = sim["Active"][i] - sim["Active"][i - 1]


sim["Poweroff"] = sim["All Node Number"] - sim["Active"]

dailychange_mean_times = len(sim[sim["Change"] != 0]) / days
dailyIchange_mean_times = len(sim[sim["Change"] > 0]) / days
Ichange_mean = sim[sim["Change"] > 0]["Change"].agg(np.mean)
change_mean = sim[sim["Change"] != 0]["Change"].apply(abs).agg(np.mean)
change_median = sim[sim["Change"] != 0]["Change"].apply(abs).agg(np.median)
poweroff_mean = sim["Poweroff"].mean()
print(cluster)
print(
    f"Daily Change Mean times : {dailychange_mean_times} \nDaily Increase Change Mean times : {dailyIchange_mean_times}")

print(f"\nIncrease Chang Mean : {Ichange_mean}  \nChang Mean : {change_mean} \nChang Median: {change_median}  \nPoweroff Mean: {poweroff_mean} "
      )

sim["Used rate"] = sim["Used Node"] / sim["All Node Number"] * 100
sim["Optimized rate"] = sim["Used Node"] / sim["Active"] * 100

ori_mean = sim["Used rate"].mean()
opt_mean = sim["Optimized rate"].mean()
print(f"\nOriginal Used rate : {ori_mean}  \nOptimized rate : {opt_mean}\n")

print(f"rmse_LGBM: {rmse_LGBM}, smape_LGBM: {smape_LGBM}")

plot_sim_node(sim, save=True)
