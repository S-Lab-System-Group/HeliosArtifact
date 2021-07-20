import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
import warnings
import utils

import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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

df = pd.read_csv(
    f"./data/philly_node_sequence.csv",
    parse_dates=["time"],
    index_col="time")

date_range = ("2017-10-03 00:00:00", "2017-12-14 23:59:00")
pred_range = ("2017-12-01 00:00:00", "2017-12-14 23:59:00")

df, period = utils.set_interval(df, "10min", "max")
train_df = df["active"][df.index < pred_range[0]].copy()
test_df = df["active"][
    (df.index < pred_range[1]) & (df.index >= pred_range[0])
].copy()


def hyperparameters():
    history, future = 2, 30
    his_threshold = 0.1
    fut_threshold = 0
    de_threshold = 1

    buffer = 20
    keep = 10
    stable_pred = 0

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
        data["rolling_day_mean_" + str(p)] = data["active"].transform(
            lambda x: x.shift(window).rolling(p).mean()
        )
        data["rolling_day_median_" + str(p)] = data["active"].transform(
            lambda x: x.shift(window).rolling(p).median()
        )
        data["rolling_day_std_" + str(p)] = data["active"].transform(
            lambda x: x.shift(window).rolling(p).std()
        )

    data["3h_ago"] = data["active"].transform(lambda x: x.shift(period // 8))

    def soft_avg(data, t):
        avg = 0.5 * data.transform(lambda x: x.shift(t))
        avg += 0.25 * (
            data.transform(lambda x: x.shift(t + 1))
            + data.transform(lambda x: x.shift(t - 1))
        )
        return avg

#     data["1h_ago_soft"] = soft_avg(data["active"], period // 24)
    data["3h_ago_soft"] = soft_avg(data["active"], period // 8)
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

lgb_train = lgb.Dataset(
    train_data[train_data["month"] < 11].iloc[:, 2:],
    train_data[train_data["month"] < 11]["active"],
)
lgb_eval = lgb.Dataset(
    train_data[train_data["month"] == 11].iloc[:, 2:],
    train_data[train_data["month"] == 11]["active"],
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
prediction = pd.Series(prediction, name="active", index=test_df.index)

rmse_LGBM = metrics.mean_squared_error(test_df, prediction, squared=False)
smape_LGBM = utils.smape(prediction, test_df)
print(f"rmse_LGBM: {rmse_LGBM}, smape_LGBM: {smape_LGBM}")


def plot_sim_node(sim, save=False):
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(sim))

    ax.plot(x, sim["active"].values, linestyle='--',
            alpha=0.9, label="Running", linewidth=2)
    ax.plot(x, sim["Prediction"].values, linestyle='--',
            alpha=0.9, label="Prediction", linewidth=2)
    ax.plot(x, sim["Active"].values, linestyle='-',
            alpha=0.9, label="Active", linewidth=2)
    ax.plot(x, sim["total"].values, linestyle='-',
            alpha=0.9, label="Total", linewidth=2)
    ax.set_xlabel(f"Dates in December")
    ax.set_ylabel(f"GPU Node Number")

    tick_interval = 144
    ax.set_xticks(x[::tick_interval])
    ax.set_xticklabels(sim.index.day[::tick_interval])
    ax.set_xlim(0, len(sim))
    ax.set_ylim(100)
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
            f"./philly_ces.pdf", bbox_inches="tight", dpi=600,
        )


"""Simulation"""
sim = df[["total", "active"]][
    (df.index < pred_range[1]) & (df.index >= pred_range[0])
]
sim["Prediction"] = prediction
sim[["Active", "Poweroff", "Change", "Lack"]] = 0
sim = sim.apply(np.floor).astype(int)

# select specific period for analyze
start = 1
end = 14
days = end - start + 1
sim = sim.loc[
    (sim.index.date >= datetime.date(2017, 12, start))
    & (sim.index.date <= datetime.date(2017, 12, end))
]

history, future, his_threshold, fut_threshold, de_threshold, buffer, keep, stable_pred = hyperparameters()

for i in range(len(sim)):
    # Initial first hour of simulation
    if i < 6:
        sim["Active"][i] = sim["active"][i] + buffer
        continue

    '''Increase Node'''
    # start servers if current not enough
    sim["Active"][i] = sim["Active"][i - 1]
    if sim["Active"][i] < sim["active"][i]:
        sim["Lack"][i] = sim["active"][i] - sim["Active"][i]
        sim["Active"][i] = sim["active"][i] + buffer
        sim["Change"][i] = sim["Active"][i] - sim["Active"][i - 1]
        stable_pred = i + keep  # keep nodes number stable

    # Aviod active server more than the total number
    sim["Active"][i] = (
        sim["total"][i]
        if sim["Active"][i] > sim["total"][i]
        else sim["Active"][i]
    )

    '''Decrease Node'''
    history_trend = np.mean(sim["active"][i - history: i]) - sim["active"][i]
    future_trend = sim["active"][i] - np.mean(sim["Prediction"][i: i + future])
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
            min(sim["Active"][i], sim["active"][i] + buffer)
        if decrease >= de_threshold:
            sim["Active"][i] = min(sim["Active"][i], sim["active"][i] + buffer)
        else:
            continue
        sim["Change"][i] = sim["Active"][i] - sim["Active"][i - 1]


sim["Poweroff"] = sim["total"] - sim["Active"]

dailychange_mean_times = len(sim[sim["Change"] != 0]) / days
dailyIchange_mean_times = len(sim[sim["Change"] > 0]) / days
Ichange_mean = sim[sim["Change"] > 0]["Change"].agg(np.mean)
change_mean = sim[sim["Change"] != 0]["Change"].apply(abs).agg(np.mean)
change_median = sim[sim["Change"] != 0]["Change"].apply(abs).agg(np.median)
poweroff_mean = sim["Poweroff"].mean()
print(
    f"Daily Change Mean times : {dailychange_mean_times} \nDaily Increase Change Mean times : {dailyIchange_mean_times}")

print(f"\nIncrease Chang Mean : {Ichange_mean}  \nChang Mean : {change_mean} \nChang Median: {change_median}  \nPoweroff Mean: {poweroff_mean} "
      )

sim["Used rate"] = sim["active"] / sim["total"] * 100
sim["Optimized rate"] = sim["active"] / sim["Active"] * 100

ori_mean = sim["Used rate"].mean()
opt_mean = sim["Optimized rate"].mean()
print(f"\nOriginal Used rate : {ori_mean}  \nOptimized rate : {opt_mean}\n")

print(f"rmse_LGBM: {rmse_LGBM}, smape_LGBM: {smape_LGBM}")

save = True
plot_sim_node(sim, save)
