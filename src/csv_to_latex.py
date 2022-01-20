import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import matplotlib.colors as mcolors
import re
import numpy as np

RESULT_TEX = os.path.join("D:", "results", "tmp", "result.tex")

sns.set_theme()
data_path = os.path.join("D:", "logs", "tmp")
result_path = os.path.join("D:", "results", "tmp")

metric_name_translator = {
    'adjusted_mutual_info_score': 'AMI',
    'few_shot_accuracy_cluster_nn': 'CCA',
    'ap_avg': "Avg AP",
    'aow0.0': "Motion Supervision",
    'aow10.0': "SPACE-Time",
    'baseline': "SPACE",
    'relevant_': "Relevant ",
    'space_invaders': "Space Invaders",
    'pong': "Pong",
    'mspacman': "Ms Pacman",
}


def translate(name):
    for k, v in metric_name_translator.items():
        name = re.sub(k, v, name)
    return name.replace("_", "\\_")


def group_by(col, key_extractor):
    result = defaultdict(list)
    for e in col:
        result[key_extractor(e)].append(e)
    return result


def sub_group_by(experiments, key_extractor):
    result = defaultdict(dict)
    for e in experiments:
        extracted = key_extractor(e)
        result[extracted][e.replace(extracted + "_", "")] = experiments[e]
    return result


def prepare_mean_std(experiments):
    columns = []
    first = True
    for k, directories in experiments.items():
        sub_dfs = [pd.read_csv(os.path.join(data_path, d, "metrics.csv"), sep=";") for d in directories]
        for sub_df in sub_dfs:
            add_contrived_columns(sub_df)
        if first:
            columns.append(sub_dfs[0]['global_step'])
            first = False
        for metric in sub_dfs[0]:
            concat_over_seeds = pd.concat([d[metric] for d in sub_dfs], axis=1)
            mean = concat_over_seeds.mean(axis=1)
            mean.name = f'{k}_{metric}_mean'
            std = concat_over_seeds.std(axis=1)
            std.name = f'{k}_{metric}_std'
            columns.extend([mean, std])
    df = pd.concat(columns, axis=1)
    return df


figure = """
\\begin{{figure}}[h]
  \\centering
  \\includegraphics[width=0.8\\textwidth]{{{1}}}
  \\caption{{{0}}}
  \\label{{fig:{2}}}
\\end{{figure}}
"""


def bar_plot(experiments, key, joined_df, title=None, caption="A plot of ...", group_key="space_invaders"):
    import matplotlib
    print(matplotlib.__version__)
    plt.clf()
    labels = experiments.keys()
    x = np.arange(len(labels))
    width = 0.8
    fig, ax = plt.subplots()
    bars = []
    styles = experiments[group_key]
    last_row = joined_df.iloc[-1]
    for i, expis in enumerate(styles):
        bars.append(ax.bar(x - width / 2 + width * (i + 0.5) / len(styles),
                           [last_row[f'{game}_{expis}_{key}_mean'] for game in experiments],
                           width / len(styles),
                           yerr=[last_row[f'{game}_{expis}_{key}_std'] for game in experiments],
                           capsize=3, label=expis))
    ax.set_ylabel(key)
    ax.set_title(f'{key} by game and variant of SPACE')
    ax.set_xticks(x, labels)
    ax.legend()
    for b in bars:
        ax.bar_label(b, padding=3)
    fig.tight_layout()
    save_and_tex(caption, key, "bar")


def save_and_tex(caption, key, kind="line"):
    img_path = os.path.join(result_path, "img", f"{kind}_{key}.png")
    if os.path.exists(img_path):
        os.remove(img_path)
    plt.savefig(img_path)
    with open(RESULT_TEX, "a") as tex:
        tex.write(figure.format(caption, f"img/{kind}_{key}", key))
        tex.write("\n")


def line_plot(experiments, key, joined_df, title=None, caption="A plot of ..."):
    plt.clf()
    for c, expis in zip(mcolors.TABLEAU_COLORS, experiments):
        plt.plot(joined_df['global_step'], joined_df[f'{expis}_{key}_mean'], color=c, label=f"{expis}")
        plt.fill_between(joined_df['global_step'], joined_df[f'{expis}_{key}_mean'] - joined_df[f'{expis}_{key}_std'],
                         joined_df[f'{expis}_{key}_mean'] + joined_df[f'{expis}_{key}_std'], alpha=0.5, color=c)
    plt.legend()
    plt.title(title if title else f"Development of {key} during training")
    save_and_tex(caption, key)


table_tex = """
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{{4}}}
\\hline
{0} \\\\
\\hline
{1} \\\\
\\hline
\\end{{tabular}}
\\caption{{{2}}}
\\label{{table:{3}}}
\\end{{table}}
"""


def table(experiment_groups, columns, joined_df, at=None, caption=None):
    if at is None:
        at = [int(joined_df.loc[len(joined_df) - 1]["global_step"].item())]
    if caption is None:
        caption = "A table showcasing " + ", ".join(columns[:-1]) + f" and {columns[-1]}."
        caption = caption.replace("_", "\\_")
    header = " & ".join(["Game", "Configuration"] + [translate(c) for c in columns]).replace("_", "\\_")
    content = []
    for game in experiment_groups:
        for expi in experiment_groups[game]:
            content.append(" & ".join(
                [game.replace("_", "\\_"), translate(expi.replace("_", "\\_"))] + [
                    f'{joined_df.loc[joined_df["global_step"] == idx][game + "_" + expi + "_" + c + "_mean"].item():.3f} '
                    f'\\pm {joined_df.loc[joined_df["global_step"] == idx][game + "_" + expi + "_" + c + "_std"].item():.3f}'
                    for c in columns for idx in at]))
    with open(RESULT_TEX, "a") as tex:
        tex.write(table_tex.format(header, " \\\\ \n".join(content), caption, ";".join(columns),
                                   "c".join(["|"] * (len(columns) + 3))))
        tex.write("\n")


table_metric_tex = """
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{{4}}}
\\hline
\\multicolumn{{{6}}}{{|c|}}{{\\textbf{{{5}}}}} \\\\
\\hline
{0} \\\\
\\hline
{1} \\\\
\\hline
\\end{{tabular}}
\\caption{{{2}}}
\\label{{table:{3}}}
\\end{{table}}
"""


def bf(name):
    return f'\\textbf{{{name}}}'


def table_by_metric(experiment_groups, columns, joined_df, at=None, caption=None, group_key="space_invaders"):
    if at is None:
        at = [int(joined_df.loc[len(joined_df) - 1]["global_step"].item())]
    for metric in columns:
        header = " & ".join([bf("Configuration")] + [bf(translate(c)) for c in experiment_groups])
        if caption is None:
            caption = f"A table showcasing {metric}."
            caption = translate(caption)
        content = []
        for expi in experiment_groups[group_key]:
            content.append(" & ".join(
                [bf(translate(expi))] + [
                    f'{joined_df.loc[joined_df["global_step"] == idx][game + "_" + expi + "_" + metric + "_mean"].item():.3f} '
                    f'$\\pm$ {joined_df.loc[joined_df["global_step"] == idx][game + "_" + expi + "_" + metric + "_std"].item():.3f}'
                    for game in experiment_groups for idx in at]))
        with open(RESULT_TEX, "a") as tex:
            tex.write(table_metric_tex.format(header, " \\\\ \n".join(content), caption, ";".join([metric]),
                                              "c".join(["|"] * (len(experiment_groups) + 2)),
                                              translate(metric), len(experiment_groups) + 1))
            tex.write("\n")


def add_contrived_columns(df):
    for style in ['relevant', 'all', 'moving']:
        aps = df.filter(regex=f'{style}_ap')
        df[f'{style}_ap_avg'] = aps.mean(axis=1)
    return df


def select_game(expi):
    for game in ["space_invaders", "pong", "mspacman", "carnival"]:
        if game in expi:
            return game


def main():
    if os.path.exists(RESULT_TEX):
        os.remove(RESULT_TEX)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path + "\\img", exist_ok=True)
    files = os.listdir(data_path)
    experiments = group_by(files,
                           lambda f: f.split("_seed")[0] + ("_" if f.split("_seed")[1][2:] else "") + f.split("_seed")[
                                                                                                          1][2:])
    joined_df = prepare_mean_std(experiments)
    experiment_groups = sub_group_by(experiments, select_game)
    mutual_info_columns = ["relevant_adjusted_rand_score", "relevant_accuracy",
                           "relevant_ap_avg", "relevant_few_shot_accuracy_with_4"]
    desired_experiment_order = ['pong', 'space_invaders', 'mspacman', 'carnival']
    experiment_groups = {k: experiment_groups[k] for k in desired_experiment_order if k in experiment_groups}
    bar_plot(experiment_groups, "relevant_few_shot_accuracy_with_4", joined_df)
    table_by_metric(experiment_groups, mutual_info_columns, joined_df)
    line_plot(experiments, "relevant_ap_avg", joined_df)
    line_plot(experiments, "relevant_accuracy", joined_df)
    print("Plotting completed")


if __name__ == '__main__':
    main()
