import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from argparse import Namespace
import os
from sklearn.neighbors import KNeighborsClassifier
from dataset import get_label_list
from collections import Counter
import numpy as np
import joblib

N_NEIGHBORS = 24
DISPLAY_CENTROIDS = True


def evaluate_z_what(arguments, z_what, labels, n, cfg, title=""):
    """
    :param arguments: dict of properties
    :param z_what: (#objects, encoding_dim)
    :param labels: (#objects)
    # :param z_where: (#objects, 4)
    :param cfg:
    :param title:
    :return:
        result: metrics
        path: to pca
        accuracy: few shot accuracy
    """
    c = Counter(labels.tolist() if labels is not None else [])
    if cfg.train.log:
        print("Distribution of matched labels:", c)
    relevant_labels = [int(part) for part in arguments['indices'].split(',')] if arguments['indices'] else list(c.keys())
    folder = f'{cfg.logdir}/{cfg.exp_name}' if cfg else f'{arguments["folder"]}'
    pca_path = f"{folder}/pca{arguments['indices'] if arguments['indices'] else ''}_{title}.png"
    if len(c) < 2:
        return Counter(), pca_path, Counter()
    label_list = get_label_list(cfg)
    relevant = torch.zeros(labels.shape, dtype=torch.bool)
    for rl in relevant_labels:
        relevant |= labels == rl
    z_what = z_what[relevant]
    labels = labels[relevant]
    train_portion = 0.9
    nb_sample = int(train_portion * len(labels))
    test_x = z_what[nb_sample:]
    test_y = labels[nb_sample:]
    train_x = z_what[:nb_sample]
    train_y = labels[:nb_sample]
    if len(torch.unique(train_y)) < 2:
        return Counter(), pca_path, Counter()
    few_shot_accuracy = {}
    z_what_by_game = {rl: train_x[train_y == rl] for rl in relevant_labels}
    labels_by_game = {rl: train_y[train_y == rl] for rl in relevant_labels}

    os.makedirs(f'{cfg.logdir}/{cfg.exp_name}', exist_ok=True)
    os.makedirs(f'classifiers', exist_ok=True)
    for training_objects_per_class in [1, 4, 16, 64]:
        current_train_sample = torch.cat([z_what_by_game[rl][:training_objects_per_class] for rl in relevant_labels])
        current_train_labels = torch.cat([labels_by_game[rl][:training_objects_per_class] for rl in relevant_labels])
        clf = RidgeClassifier()
        clf.fit(current_train_sample, current_train_labels)
        acc = clf.score(test_x, test_y)
        filename = f'{cfg.logdir}/{cfg.exp_name}/z_what-classifier_with_{training_objects_per_class}.joblib.pkl'
        joblib.dump(clf, filename)
        few_shot_accuracy[f'few_shot_accuracy_with_{training_objects_per_class}'] = acc
    model_name = cfg.resume_ckpt.split("/")[-1].replace(".pth", "")
    save_path = f"classifiers/{model_name}_z_what_classifier.joblib.pkl"
    joblib.dump(clf, save_path)
    print(f"Saved classifiers in {save_path}")

    clf = KMeans(n_clusters=len(relevant_labels))
    y = clf.fit_predict(z_what)
    results = {
        'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(labels, y),
        'adjusted_rand_score': metrics.adjusted_rand_score(labels, y),
    }

    centroids = clf.cluster_centers_
    X = train_x.numpy()
    nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, len(X))).fit(X)
    _, z_w_idx = nn.kneighbors(centroids)
    centroid_label = []
    for cent, nei in zip(centroids, z_w_idx):
        count = {rl: 0 for rl in relevant_labels}
        added = False
        for i in range(N_NEIGHBORS):
            nei_label = labels[nei[i]].item()
            count[nei_label] += 1
            if count[nei_label] > 6.0 / (i + 1) if nei_label in centroid_label else 3.0 / (i + 1):
                centroid_label.append(nei_label)
                added = True
                break
        if not added:
            leftover_labels = [i for i in relevant_labels if i not in centroid_label]
            centroid_label.append(leftover_labels[0])
    nn_class = KNeighborsClassifier(n_neighbors=1)
    nn_class.fit(centroids, centroid_label)
    few_shot_accuracy[f'few_shot_accuracy_cluster_nn'] = nn_class.score(test_x, test_y)

    train_all = torch.cat((z_what, labels.unsqueeze(1)), 1)
    colors = ['black', 'r', 'g', 'b', 'c', 'm', 'y', 'pink', 'purple', 'orange',
              'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
    # SORT THE INDICES
    sorted = []
    for i in relevant_labels:
        mask = train_all.T[-1] == i
        indices = torch.nonzero(mask)
        sorted.append(indices)
    pca = PCA(n_components=arguments['dim'])
    z_what_emb = pca.fit_transform(z_what.numpy())
    if arguments['edgecolors']:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(20, 12)
        ax.set_facecolor((0.3, 0.3, 0.3))
        plt.suptitle("Labeled PCA of z_whats", y=0.96, fontsize=28)
        plt.title("Inner Color is GT, Outer is greedy Centroid-based label", fontsize=18, pad=20)
        n = min(n, 10000)
        for i, idx in enumerate(sorted):
            if torch.numel(idx) == 0:
                continue
            y_idx = y[idx] if torch.numel(idx) > 1 else [[y[idx]]]
            colr = colors[relevant_labels[i]]
            edge_colors = [colors[centroid_label[assign[0]]] for assign in y_idx]
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:1],
                       z_what_emb[:, 1][idx].squeeze()[:1],
                       c=colr, label=label_list[relevant_labels[i]].replace("_", " "),
                       alpha=0.7)
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:n],
                       z_what_emb[:, 1][idx].squeeze()[:n],
                       c=colr,
                       alpha=0.7, edgecolors=edge_colors, s=100, linewidths=2)
        centroid_emb = pca.transform(centroids)

        if DISPLAY_CENTROIDS:
            for c_emb, cl in zip(centroid_emb, centroid_label):
                if "pacman" in label_list:
                    colr = [np.array(base_objects_colors[label_list[cl]]) / 255]
                else:
                    colr = colors[cl]
                ax.scatter([c_emb[0]],
                               [c_emb[1]],
                               c=colr,
                               edgecolors='black', s=100, linewidths=2)
        plt.legend(prop={'size': 6})
        directory = f"{folder}"
        if not os.path.exists(directory):
            print(f"Writing PCA to {directory}")
            os.makedirs(directory)
        plt.savefig(pca_path)
    else:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(30, 12)
        axs[0].set_facecolor((0.3, 0.3, 0.3))
        axs[1].set_facecolor((0.3, 0.3, 0.3))
        axs[0].set_title("Ground Truth Labels")
        axs[1].set_title("Labels Following Clustering")
        for i, idx in enumerate(sorted):
            # dimension issue only if there is exactly one object of one kind
            if torch.numel(idx) == 0:
                continue
            y_idx = y[idx] if torch.numel(idx) > 1 else [[y[idx]]]
            colr = colors[relevant_labels[i]]
            edge_colors = [colors[centroid_label[assign[0]]] for assign in y_idx]
            axs[0].scatter(z_what_emb[:, 0][idx].squeeze(),
                           z_what_emb[:, 1][idx].squeeze(),
                           c=colr,
                           label=label_list[relevant_labels[i]],
                           alpha=0.7)
            axs[1].scatter(z_what_emb[:, 0][idx].squeeze(),
                           z_what_emb[:, 1][idx].squeeze(),
                           c=edge_colors,
                           label=label_list[relevant_labels[i]],
                           alpha=0.7)
        centroid_emb = pca.transform(centroids)

        for c_emb, cl in zip(centroid_emb, centroid_label):
            colr = colors[cl]
            axs[0].scatter([c_emb[0]],
                           [c_emb[1]],
                           c=colr,
                           edgecolors='black', s=100, linewidths=2)
            axs[1].scatter([c_emb[0]],
                           [c_emb[1]],
                           c=colr,
                           edgecolors='black', s=100, linewidths=2)

        axs[0].legend(prop={'size': 6})
        axs[1].legend(prop={'size': 6})
        if not os.path.exists(f"{folder}"):
            os.makedirs(f"{folder}")
        plt.tight_layout()
        plt.savefig(pca_path)
        plt.close(fig)
    return results, pca_path, few_shot_accuracy


# all_train_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/train_labels.csv")
# all_validation_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/validation_labels.csv")

# label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
#               "white_ghost", "fruit", "save_fruit", "life", "life2", "score0"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the z_what encoding')
    parser.add_argument('-folder', type=str, default="mspacman_atari_example",
                        help='the output folder')
    parser.add_argument('-indices', type=str, default=None,
                        help='The relevant objects by their index, e.g. \"0,1\" for Pacman and Sue')
    parser.add_argument('-edgecolors', type=bool, default=True,
                        help='True iff the ground truth labels and the predicted labels '
                             '(Mixture of some greedy policy and NN) should be drawn in the same image')
    parser.add_argument('-dim', type=int, choices=[2, 3], default=2,
                        help='Number of dimension for PCA/TSNE visualization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nb_used_sample = 500

    # z_what_train = torch.randn((400, 32))
    # train_labels = torch.randint(high=8, size=(400,))
    z_what_train = torch.cat(torch.load(f"labeled/{args.folder}/z_what_validation.pt"))
    train_labels = torch.cat(torch.load(f"labeled/{args.folder}/labels_validation.pt"))

    evaluate_z_what(vars(args), z_what_train, train_labels, nb_used_sample, cfg=None)
