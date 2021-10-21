from packaging import version

import os
import pandas as pd
import numpy as np
import seaborn as sns
import tensorboard as tb

from matplotlib import pyplot as plt
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.signal import savgol_filter

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

sns.set_style("ticks")

# helper function to smooth plots
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# https://stackoverflow.com/questions/46633544/smoothing-out-curve-in-python
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
def smooth2(y, window=21, order=2):
    return savgol_filter(y, window, order)


def plot_q_learning():
    sns.set(font_scale=1.3)
    sns.set_style("ticks")
    plt.figure(figsize=(10,5))
    run = "Q-Learning-Pong-v2s"
    x = EventAccumulator(path=os.getcwd() + "/ql/logs/" + run + "/")
    x.Reload()
    # print(x.Tags())
    df_r = pd.DataFrame(x.Scalars('Train/reward_episode'))
    # smooth
    df_r["s-value"] = smooth2(df_r["value"], 121)
    # create both plots, one unsmoothed with alpha, one smoothed
    p = sns.lineplot(data=df_r, x="step", y="value", alpha=0.3, legend=False)
    sns.lineplot(data=df_r, x="step", y="s-value")
    sns.despine(offset=1, trim=True)
    plt.tight_layout()
    #plt.savefig("dummy.pdf")
    plt.show()


# function to plot space exp
##### DONE #######
def plot_space_exp():
    sns.set(font_scale=1.3)
    sns.set_style("ticks")
    plt.figure(figsize=(10,5))
    #sns.set(font_scale=1.3)
    runs = ["DQ-Learning-Pong-v8-cnn", "DQ-Learning-Pong-v9-zw", "DQ-Learning-Pong-v11r"]
    df_list = []
    for i, run in enumerate(runs):
        x = EventAccumulator(path=os.getcwd() + "/dqn/logs/" + run + "/")
        x.Reload()
        # print(x.Tags())
        tdf = pd.DataFrame(x.Scalars('Train/reward_episode'))
        tag = "DuelDQN w. sorted SPACE output"
        # sert right experiment tag
        if i == 0:
            tag = "Baseline with raw images"
        elif i == 1:
            tag = "MLP w. sorted SPACE output"
        tdf["Experiment"] = tag
        tdf["s-value"] = smooth2(tdf["value"], 121)
        #tdf["s-value"] = smooth(tdf["value"], 60)
        df_list.append(tdf)
    df = pd.concat(df_list)
    print(df)
    p = sns.lineplot(data=df, x="step", y=df["value"], hue="Experiment", alpha=0.3, legend=False)
    sns.lineplot(data=df, x="step", y=df["s-value"], hue="Experiment", linewidth=2).set_title("Reward per episode")
    p.set(ylabel='Reward', xlabel='Episode')
    sns.despine(offset=1, trim=True)
    plt.tight_layout()
    #plt.savefig("dummy.pdf")
    plt.show()


# function to plot exp4
def plot_exp4():
    #runs = ["exp2-re-pong-v2", "exp2-re-pong-v2-2", "exp2-re-pong-v2-3"]
    runs = ["exp4-re-pong-ptr12", "exp4-re-pong-ptr12-2", "exp4-re-pong-ptr12-3", "exp4-re-pong-ptr12-4"]

    df_list = []

    for run in runs:
        x = EventAccumulator(path=os.getcwd() + "/xrl/relogs/" + run + "/")
        x.Reload()
        # print(x.Tags())
        tdf = pd.DataFrame(x.Scalars('Train/Avg reward'))
        df_list.append(tdf)

    df = pd.concat(df_list)
    print(df)

    sns.lineplot(data=df, x="step", y="reward").set_title("Average reward of runs")
    plt.show()

# call function to plot
plot_q_learning()