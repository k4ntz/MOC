import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import matplotlib.colors as mcolors

RESULT_TEX = os.path.join("D:", "results", "seed", "result.tex")

sns.set_theme()
data_path = os.path.join("D:", "logs", "flow")
result_path = os.path.join("D:", "results", "flow")

metric_name_translator = {
    'adjusted_mutual_info_score': 'AMI'
}


def translate(name):
    for k, v in metric_name_translator.items():
        name = name.replace(k, v)
    return name


def group_by(col, key_extractor):
    result = defaultdict(list)
    for e in col:
        result[key_extractor(e)].append(e)
    return result


def prepare_mean_std(experiments):
    df = pd.DataFrame()
    for k, directories in experiments.items():
        sub_dfs = [pd.read_csv(os.path.join(data_path, d, "metrics.csv"), sep=";") for d in directories]
        for metric in sub_dfs[0]:
            if df.empty:
                df['global_step'] = sub_dfs[0]['global_step']
            concat_over_seeds = pd.concat([d[metric] for d in sub_dfs], axis=1)
            df[f'{k}_{metric}_mean'] = concat_over_seeds.mean(axis=1)
            df[f'{k}_{metric}_std'] = concat_over_seeds.std(axis=1)
    return df


figure = """
\\begin{{figure}}[h]
  \\centering
  \\includegraphics[width=0.8\\textwidth]{{{1}}}
  \\caption{{{0}}}
  \\label{{fig:{2}}}
\\end{{figure}}
"""


def line_plot(experiments, key, joined_df, title=None, caption="A plot of ..."):
    for c, expis in zip(mcolors.TABLEAU_COLORS, experiments):
        plt.plot(joined_df['global_step'], joined_df[f'{expis}_{key}_mean'], color=c, label=f"{expis}")
        plt.fill_between(joined_df['global_step'], joined_df[f'{expis}_{key}_mean'] - joined_df[f'{expis}_{key}_std'],
                         joined_df[f'{expis}_{key}_mean'] + joined_df[f'{expis}_{key}_std'], alpha=0.5, color=c)
    plt.legend()
    plt.title(title if title else f"Development of {key} during training")
    img_path = os.path.join(result_path, "img", f"{key}.png")
    if os.path.exists(img_path):
        os.remove(img_path)
    plt.savefig(img_path)
    with open(RESULT_TEX, "a") as tex:
        tex.write(figure.format(caption, f"img/{key}", key))
        tex.write("\n")


table_tex = """
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{{4}}}
\\hline
Method or Weighting & {0} \\\\
\\hline
{1} \\\\
\\hline
\\end{{tabular}}
\\caption{{{2}}}
\\label{{table:{3}}}
\\end{{table}}
"""


def table(experiments, columns, joined_df, at=None, caption=None):
    if at is None:
        at = [int(joined_df.loc[len(joined_df) - 1]["global_step"].item())]
    if caption is None:
        caption = "A table showcasing " + " ,".join(columns[:-1]) + f" and {columns[-1]}."
        caption = caption.replace("_", "\\_")
    header = " & ".join([translate(c) for c in columns]).replace("_", "\\_")
    content = []
    for expi in experiments:
        content.append(" & ".join(
            [expi.replace("_", "\\_")] + [
                f'{joined_df.loc[joined_df["global_step"] == idx][expi + "_" + c + "_mean"].item():.4f}'
                for c in columns for idx in at]))
    with open(RESULT_TEX, "a") as tex:
        tex.write(table_tex.format(header, " \\\\ \n".join(content), caption, ";".join(columns),
                                   "c".join(["|"] * (len(columns) + 2))))
        tex.write("\n")


def main():
    if os.path.exists(RESULT_TEX):
        os.remove(RESULT_TEX)
    files = os.listdir(data_path)
    experiments = group_by(files, lambda f: f.split("_seed")[0])
    joined_df = prepare_mean_std(experiments)
    mutual_info_columns = ["all_adjusted_mutual_info_score", "moving_adjusted_mutual_info_score",
                           "relevant_adjusted_mutual_info_score"]
    table(experiments, mutual_info_columns, joined_df)
    line_plot(experiments, "all_perfect", joined_df)
    line_plot(experiments, "relevant_perfect", joined_df)


if __name__ == '__main__':
    main()
