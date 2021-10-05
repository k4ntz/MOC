from packaging import version

import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

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

sns.lineplot(data=df, x="step", y="value").set_title("Average reward of runs")
plt.show()