import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import matplotlib.colors as mcolors
import re
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

label_list_pacman = ["no_label", "pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost", "eyes",
                     "white_ghost", "fruit", "save_fruit", "life1", "life2", "score", "corner_block"]

label_list_pong = ["no_label", "player", 'enemy', 'ball', 'enemy_score', 'player_score']

label_list_carnival = ["no_label", "owl", 'rabbit', 'shooter', 'refill', 'bonus', "duck",
                       "flying_duck", "score", "pipes", "eating_duck", "bullet"]

label_list_boxing = ["no_label", "black", 'black_score', 'clock', 'white', 'white_score', 'logo']

label_list_tennis = ["no_label", "player", 'enemy', 'ball', 'ball_shadow', 'net', 'logo',
                     'player_score', 'enemy_score']

# Maybe enemy bullets, but how should SPACE differentiate
label_list_space_invaders = ["no_label"] + [f"{side}_score" for side in ['left', 'right']] + [f"enemy_{idx}"
                                                                                              for idx in
                                                                                              range(6)] \
                            + ["space_ship", "player", "block", "bullet"]

label_list_riverraid = ["no_label", "player", 'fuel_gauge', 'fuel', 'lives', 'logo', 'score', 'shot', 'fuel_board',
                        'building', 'street', 'enemy']

label_list_air_raid = ["no_label", "player", 'score', 'building', 'shot', 'enemy']

RESULT_TEX = os.path.join("D:", "results", "final", "result.tex")

sns.set_theme()
data_path = os.path.join("D:", "final")
result_path = os.path.join("D:", "results", "final")

metric_name_translator = {
    'adjusted_mutual_info_score': 'AMI',
    'few_shot_accuracy_cluster_nn': 'CCA',
    'ap_avg': "Average AP",
    'accuarcy': "Accuracy",
    'aow0.0': "SPACE-Flow",
    'aow10.0': "SPACE-Time",
    'baseline': "SPACE",
    'relevant_': "Relevant ",
    'space_invaders': "Space Invaders",
    'pong': "Pong",
    'mspacman': "Ms Pacman",
    'tennis': "Tennis",
    'air_raid': "Air Raid",
    'boxing': "Boxing",
    'carnival': "Carnival",
    'riverraid': "Riverraid",
    'f_score': "F-Score",
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
        metrics = [column for column in sub_dfs[0]]
        if first:
            columns.append(sub_dfs[0]['global_step'])
            first = False

        for idx, sub_df in enumerate(sub_dfs):
            sub_df.rename(columns={column: f'{directories[idx]}_{column}' for column in sub_df}, inplace=True)
        for metric in metrics:
            concat_over_seeds = pd.concat([sub_df[f'{d}_{metric}'] for d, sub_df in zip(directories, sub_dfs)], axis=1)
            mean = concat_over_seeds.mean(axis=1)
            mean.name = f'{k}_{metric}_mean'
            std = concat_over_seeds.std(axis=1)
            std.name = f'{k}_{metric}_std'
            columns.extend([mean, std])
            columns.extend([concat_over_seeds[column] for column in concat_over_seeds])
    df = pd.concat(columns, axis=1)
    return df


figure = """
\\begin{{subfigure}}{{.45\\textwidth}}
  \\centering
  \\includegraphics[width=\\textwidth]{{{1}}}
  \\caption{{{0}}}
  \\label{{fig:{2}}}
\\end{{subfigure}}
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
    img_path = os.path.join(result_path, "img", f"{caption}_{kind}_{key}.pdf")
    if os.path.exists(img_path):
        os.remove(img_path)
    plt.savefig(img_path, bbox_inches="tight")
    plt.clf()
    with open(RESULT_TEX, "a") as tex:
        tex.write(figure.format(caption, f"img/{caption}_{kind}_{key}", key))
        tex.write("\n")


def line_plot(experiment_groups, key, joined_df, title=None, caption="A plot of ..."):
    plt.clf()
    for game in experiment_groups:
        plt.figure(figsize=(12, 7))
        plt.ylim((-0.1, 1.1))
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
        plt.xticks(fontsize=18)
        for c, expis in zip(mcolors.TABLEAU_COLORS, experiment_groups[game]):
            plt.plot(joined_df['global_step'], joined_df[f'{game}_{expis}_{key}_mean'], color=c,
                     label=f"{translate(expis)}")
            plt.fill_between(joined_df['global_step'],
                             joined_df[f'{game}_{expis}_{key}_mean'] - joined_df[f'{game}_{expis}_{key}_std'],
                             joined_df[f'{game}_{expis}_{key}_mean'] + joined_df[f'{game}_{expis}_{key}_std'],
                             alpha=0.5, color=c)
        plt.legend(loc="lower right", fontsize=18)
        save_and_tex(translate(game), key)


from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


def my_cmap(colors):
    nodes = [0.0, 1.0]
    return LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)), N=256)


def pr_plot(experiment_groups, joined_df):
    plt.clf()
    dth = 0.1
    for game in experiment_groups:
        plt.figure(figsize=(12, 7))

        plt.ylim((-0.05, 1.05))
        plt.xlim((-0.05, 1.05))
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        for c, expis in zip(mcolors.TABLEAU_COLORS, experiment_groups[game]):
            if expis == "aow10.0":
                continue
            if expis == "baseline":
                x = np.concatenate(
                    [joined_df[f'{game}_{expis}_seed{idx}_relevant_recall'].to_numpy() for idx in range(5)])
                y = np.concatenate(
                    [joined_df[f'{game}_{expis}_seed{idx}_relevant_precision'].to_numpy() for idx in range(5)])
            else:
                x = np.concatenate(
                    [joined_df[f'{game}_seed{idx}_{expis}_relevant_recall'].to_numpy() for idx in range(5)])
                y = np.concatenate(
                    [joined_df[f'{game}_seed{idx}_{expis}_relevant_precision'].to_numpy() for idx in range(5)])
            # x_err = joined_df[f'{game}_{expis}_relevant_recall_std'].to_numpy()
            # y_err = joined_df[f'{game}_{expis}_relevant_precision_std'].to_numpy()
            # dist = np.sqrt(x_**2 + y_**2)
            #
            high = np.array(mcolors.to_rgb(c))
            red = my_cmap((high * 0.5, high))
            colors = red(np.concatenate([np.linspace(0, 1, 26) for _ in range(5)]))
            # plt.scatter(x[4::5], y[4::5], c=colors2, zorder=3, edgecolors='black', alpha=0.8)
            # plt.scatter(x, y, c=colors, zorder=3, edgecolors='black', alpha=0.8)
            # for xe, ye, ex, ey, ec in zip(x[4::5], y[4::5], x_err[4::5], y_err[4::5], colors):
            #     plt.errorbar(xe, ye, xerr=ex, yerr=ey, fmt='none', ecolor=ec, capsize=4)
            # plt.quiver(x[:-1][dist > dth], y[:-1][dist > dth], x_[dist > dth],
            #            y_[dist > dth], np.linspace(0, 1, len(x) - 1)[dist > dth],
            #            angles='xy', width=0.003, scale_units='xy', scale=1,
            #            linewidth=1, edgecolor='#00000080', cmap=red, zorder=4)
            prs = np.stack((x, y), axis=1)
            bounds = np.array([pr for pr in prs if all(not all(pr2 > pr) for pr2 in prs)])
            bounds_ind = np.argsort(bounds[:, 0])
            bounds = bounds[bounds_ind]
            plt.scatter(x, y, alpha=0.5, c=colors, s=15, marker='x')
            bounds = np.insert(bounds, 0, [0.0, bounds[0, 1]], axis=0)
            bounds = np.append(bounds, [[bounds[-1, 0], 0.0]], axis=0)
            plt.plot(bounds[:, 0], bounds[:, 1], linestyle='dotted', c=c, zorder=5, label=translate(expis), )
        plt.legend(loc="lower left")
        save_and_tex(translate(game), "Precision-Recall Curve", kind="Quiver")


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


def table_entry(joined_df, idx, game, c, metric, variants):
    all_mean = [joined_df.loc[joined_df["global_step"] == idx][game + "_" + v + "_" + metric + "_mean"].item() for v in
                variants]

    mean = joined_df.loc[joined_df["global_step"] == idx][game + "_" + c + "_" + metric + "_mean"].item()
    std = joined_df.loc[joined_df["global_step"] == idx][game + "_" + c + "_" + metric + "_std"].item()
    if mean > 0.99 * np.max(all_mean):
        return f'\\textbf{{{mean :.3f} $\\pm$ {std :.3f}}}'
    return f'{mean :.3f} $\\pm$ {std :.3f}'


def table_by_metric(experiment_groups, columns, joined_df, at=None, caption=None, group_key="space_invaders"):
    if at is None:
        at = [int(joined_df.loc[len(joined_df) - 1]["global_step"].item())]
    for metric in columns:
        header = " & ".join([bf("Configuration")] + [bf(translate(c)) for c in experiment_groups[group_key]])
        if caption is None:
            caption = f"A table showcasing {metric}."
            caption = translate(caption)
        content = []
        for game in experiment_groups:
            content.append(" & ".join(
                [bf(translate(game))] + [
                    table_entry(joined_df, idx, game, c, metric, experiment_groups[group_key])
                    for c in experiment_groups[group_key] for idx in at]))
        with open(RESULT_TEX, "a") as tex:
            tex.write(table_metric_tex.format(header, " \\\\ \n".join(content), caption, ";".join([metric]),
                                              ("c".join(["|"] * (len(experiment_groups) + 2)).replace('|', '||',
                                                                                                      2).replace('||',
                                                                                                                 '|',
                                                                                                                 1)),
                                              translate(metric), 4))
            tex.write("\n")


def add_contrived_columns(df):
    for style in ['relevant', 'all', 'moving']:
        aps = df.filter(regex=f'{style}_ap')
        df[f'{style}_ap_avg'] = aps.mean(axis=1)
        df[f'{style}_f_score'] = 2 * df[f'{style}_precision'] * df[f'{style}_recall'] / (
                df[f'{style}_precision'] + df[f'{style}_recall'] + 1e-8)
    return df


qual_page = """\\newpage
\\thispagestyle{{empty}}
\\newgeometry{{left=1cm,bottom=1cm,right=1cm,top=1cm}}
    \\begin{{sidewaysfigure}}[htbp]
    \\centering
    \\centerline{{\\includegraphics[width=1\\textwidth]{{img_qual/0-separations_{0}.png}}}}
    \\caption{{{1}. The Columns from left to right: Input Image, Reconstruction, Foreground, 
    Bounding Boxes, Background, K x Background Components, K x Background Masks, K x Background Color Maps, Alpha}}
    \\end{{sidewaysfigure}}
\\restoregeometry
"""


def switch_aow(experiment_name):
    if "aow" in experiment_name:
        split = experiment_name.split("_")
        return "_".join(split[:-2] + ['z', split[-1]] + [split[-2]])
    return experiment_name


def switch_aow_back(experiment_name):
    experiment_name = experiment_name.replace("seed4", "")
    if "aow" in experiment_name:
        split = experiment_name.split("_")
        return "_".join(split[:-3] + [split[-1]] + [split[-2]])
    return experiment_name


def qualitative_appendix(files):
    pngs = [switch_aow(f) for f in files if "seed4" in f]
    pngs = sorted(pngs)
    for f in pngs:
        with open(RESULT_TEX, "a") as tex:
            tex.write(qual_page.format(switch_aow_back(f),
                                       translate(f).replace('\\_seed4', '').replace("z", " ").replace("\\_", " ")))
            tex.write("\n")


def select_game(expi):
    for game in ['air_raid', 'boxing', 'carnival', 'mspacman', 'pong', 'riverraid', 'space_invaders', 'tennis']:
        if game in expi:
            return game
    raise ValueError(f'{expi} does not contain any of the known games...')


object_table = """\\begin{{subtable}}[b]{{\\textwidth}}\\centering
\\begin{{tabular}}{{@{{}}lccrr@{{}}}}
\\toprule
\\textbf{{Object/Entity}} & \\textbf{{Method}} & \\textbf{{Relevant}} & \\textbf{{Precision}} & \\textbf{{Recall}} \\\\ 
\\midrule
{0} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{{1}}}
\\end{{subtable}}
"""


def generate_object_tables(desired_experiment_order):
    for game in desired_experiment_order:
        with open(RESULT_TEX, "a") as tex:
            tex.write(object_table.format(" \\\\ \n".join(
                [" & ".join([translate(label), "Color", "Yes/No", "1.000", "1.000"]) for label in label_list_for(game)
                 if label != "no_label"]), translate(game)))
            tex.write("\n")


def label_list_for(game):
    """
    Return Labels from line in csv file
    """
    if "mspacman" in game:
        return label_list_pacman
    elif "carnival" in game:
        return label_list_carnival
    elif "pong" in game:
        return label_list_pong
    elif "boxing" in game:
        return label_list_boxing
    elif "tennis" in game:
        return label_list_tennis
    elif "air_raid" in game:
        return label_list_air_raid
    elif "riverraid" in game:
        return label_list_riverraid
    elif "space_invaders" in game:
        return label_list_space_invaders
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def accuracy_plot(experiment_groups, joined_df):
    ax1 = plt.axes(projection='3d')

    xlen = 3 * 8
    _x = np.arange(xlen + 8)
    _x = np.array([sx for sx in _x if sx % 4 != 0])
    _y = np.arange(4)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = np.zeros_like(x, dtype=np.double)
    xc = 0
    for expis in experiment_groups:
        for c in experiment_groups["space_invaders"]:
            for yc, samples in enumerate([1, 4, 16, 64]):
                top[xc + yc * xlen] = joined_df.loc[joined_df["global_step"] == 5000][expis + "_" + c + "_" + f"relevant_few_shot_accuracy_with_{samples}_mean"].item()
            xc += 1

    bottom = np.zeros_like(top)
    width = depth = 1
    print(top)
    ax1.view_init(10, -82)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] * 8
    print(len(colors))
    print(len(top))
    ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors * 4, linewidth=0.3)
    ax1.set_title('Accuracy')
    ax1.set_zticks([0.0, 0.25, 0.5, 0.75, 1.0], size='small')
    ax1.set_yticks([0, 1, 2, 3], ["1", "4", "16", "64"], size='small')
    ax1.set_xticks([1, 5, 9, 13, 17, 21, 25, 29],
                   ["AirRaid", "Boxing", "Carnival", "MsPacman", "Pong", "Riverraid", "SpaceInvaders", "Tennis"],
                   size='small', rotation=60, ha="right")
    ax1.tick_params(pad=0)
    scale = np.diag([4.0, 1.0, 1.0, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax1), scale)

    # ax1.get_proj = short_proj
    ax1.set_box_aspect(aspect=(4, 1, 1))
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#1f77b4', lw=4),
                    Line2D([0], [0], color='#ff7f0e', lw=4),
                    Line2D([0], [0], color='#2ca02c', lw=4)]
    ax1.legend(custom_lines, ['SPACE', 'SPACE-Flow', 'SPACE-Time'])
    img_path = os.path.join(result_path, "img", "accuracy.svg")
    plt.savefig(img_path, bbox_inches="tight")



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
    # mutual_info_columns = ["relevant_accuracy", "relevant_ap_avg", "relevant_f_score",
    #             "all_accuracy", "all_ap_avg", "all_f_score"]
    # mutual_info_columns = ["relevant_few_shot_accuracy_with_1", "relevant_few_shot_accuracy_with_4",
    #                        "relevant_few_shot_accuracy_with_16", "relevant_few_shot_accuracy_with_64",
    #                        "relevant_few_shot_accuracy_cluster_nn"]
    #
    # mutual_info_columns += [column.replace("relevant", "all") for column in mutual_info_columns]
    mutual_info_columns = ["relevant_f_score", "relevant_few_shot_accuracy_with_4",
                           "relevant_adjusted_mutual_info_score"]
    desired_experiment_order = ['air_raid', 'boxing', 'carnival', 'mspacman', 'pong', 'riverraid', 'space_invaders', 'tennis']
    # desired_experiment_order = ['riverraid', 'space_invaders']
    experiment_groups = {k: experiment_groups[k] for k in desired_experiment_order if k in experiment_groups}
    # bar_plot(experiment_groups, "relevant_few_shot_accuracy_with_4", joined_df)
    # table_by_metric(experiment_groups, mutual_info_columns, joined_df)
    # line_plot(experiments, "relevant_ap_avg", joined_df)
    # line_plot(experiment_groups, "relevant_f_score", joined_df)
    # line_plot(experiment_groups, "relevant_few_shot_accuracy_with_1", joined_df)
    # line_plot(experiment_groups, "relevant_few_shot_accuracy_with_4", joined_df)
    # line_plot(experiment_groups, "relevant_few_shot_accuracy_with_16", joined_df)
    # line_plot(experiment_groups, "relevant_few_shot_accuracy_cluster_nn", joined_df)
    # line_plot(experiment_groups, "relevant_adjusted_mutual_info_score", joined_df)
    # pr_plot(experiment_groups, joined_df)
    # generate_object_tables(desired_experiment_order)
    # qualitative_appendix(files)
    accuracy_plot(experiment_groups, joined_df)
    print("Plotting completed")


if __name__ == '__main__':
    main()
