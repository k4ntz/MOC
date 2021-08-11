import torch
import torch.nn.functional as F
from simple_net import Network
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn import metrics
from utils import get_labels
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

N_NEIGHBORS = 24

warnings.filterwarnings("ignore", category=UserWarning)
import argparse
from utils import base_objects_colors
import numpy as np


def evaluate_z_what(arguments, z_what, labels, n, cfg):
    relevant_labels = [int(part) for part in arguments['indices'].split(',')] if arguments['indices'] else range(
        labels.max() + 1)
    folder = f'hyper/{cfg.exp_name}{cfg.seed}' if cfg else f'{arguments["folder"]}1'
    relevant = torch.zeros(labels.shape, dtype=torch.bool)
    for rl in relevant_labels:
        relevant |= labels == rl
    z_what = z_what[relevant]
    labels = labels[relevant]
    train_portion = 0.8
    nb_sample = int(train_portion * len(labels))
    test_x = z_what[nb_sample:]
    test_y = labels[nb_sample:]
    few_shot_accuracy = {}
    z_what_by_game = {rl: z_what[labels == rl] for rl in relevant_labels}
    labels_by_game = {rl: labels[labels == rl] for rl in relevant_labels}
    for train_portion in [1, 4, 16, 64]:
        current_train_sample = torch.cat([z_what_by_game[rl][:train_portion] for rl in relevant_labels])
        current__train_labels = torch.cat([labels_by_game[rl][:train_portion] for rl in relevant_labels])
        clf = LogisticRegression()
        clf.fit(current_train_sample, current__train_labels)
        acc = clf.score(test_x, test_y)
        few_shot_accuracy[f'few_shot_accuracy_with_{train_portion}'] = acc

    clf = KMeans(n_clusters=len(relevant_labels))
    y = clf.fit_predict(z_what)

    results = {
        'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(labels, y),
        'adjusted_rand_score': metrics.adjusted_rand_score(labels, y),
        # 'homogeneity_score': metrics.homogeneity_score(labels, y),
        # 'completeness_score': metrics.completeness_score(labels, y),
        # 'v_measure_score': metrics.v_measure_score(labels, y),
        # 'fowlkes_mallows_score': metrics.fowlkes_mallows_score(labels, y)
    }
    centroids = clf.cluster_centers_
    X = z_what.numpy()
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS).fit(X)
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

    train_all = torch.cat((z_what, labels.unsqueeze(1)), 1)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'purple', 'orange',
              'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
    # SORT THE INDICES
    sorted = []
    for i in relevant_labels:
        mask = train_all.T[-1] == i
        indices = torch.nonzero(mask)
        sorted.append(indices)
    pca = PCA(n_components=arguments['dim'])
    z_what_emb = pca.fit_transform(z_what.numpy())

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(30, 12)
    axs[0].set_facecolor((0.3, 0.3, 0.3))
    axs[1].set_facecolor((0.3, 0.3, 0.3))
    axs[0].set_title("Ground Truth Labels")
    axs[1].set_title("Labels Following Clustering")
    n = min(n, 10000)
    for i, idx in enumerate(sorted):
        if "pacman" in label_list:
            colr = [np.array(base_objects_colors[label_list[relevant_labels[i]]]) / 255]
            edge_colors = [np.array(base_objects_colors[label_list[centroid_label[assign[0]]]]) / 255 for assign in
                           y[idx]]
        else:
            colr = colors[relevant_labels[i]]
            edge_colors = [colors[centroid_label[assign[0]]] for assign in y[idx]]
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
        if "pacman" in label_list:
            colr = [np.array(base_objects_colors[label_list[cl]]) / 255]
        else:
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
    plt.savefig(
        f"../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}.png")
    plt.close(fig)
    return results, f"../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}.png", \
           few_shot_accuracy


def classify_z_what(arguments, z_what, labels, n, cfg):
    print(arguments, z_what.shape, labels.shape)
    relevant_labels = [int(part) for part in arguments['indices'].split(',')] if arguments['indices'] else range(
        labels.max() + 1)

    folder = f'hyper/{cfg.exp_name}{cfg.seed}' if cfg else f'{arguments["folder"]}1'
    if arguments['type'] == "show":
        print("showing...")
        train_all = torch.cat((z_what, labels.unsqueeze(1)), 1)
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'purple', 'orange',
                  'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
        # SORT THE INDICES
        sorted = []
        for i in relevant_labels:
            mask = train_all.T[-1] == i
            indices = torch.nonzero(mask)
            sorted.append(indices)
        if arguments['method'] == "pca":
            pca = PCA(n_components=arguments['dim'])
            z_what_emb = pca.fit_transform(z_what.numpy())
        else:
            try:
                from MulticoreTSNE import MulticoreTSNE as TSNE

                if arguments['dim'] != 2:
                    print("Un cuml, only dim = 2 is supported")
            except ImportError:
                print("Install cuml if GPU available to improve speed (requires conda)")
                from sklearn.manifold import TSNE
            tsne = TSNE(n_jobs=4, n_components=arguments['dim'], verbose=True)
            z_what_emb = tsne.fit_transform(z_what.numpy())

        if arguments['dim'] == 2:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(20, 12)
            ax.set_facecolor((0.3, 0.3, 0.3))
            n = min(n, 10000)
            for i, idx in enumerate(sorted):
                if "pacman" in label_list:
                    colr = [np.array(base_objects_colors[label_list[relevant_labels[i]]]) / 255]
                else:
                    colr = colors[relevant_labels[i]]
                ax.scatter(z_what_emb[:, 0][idx].squeeze()[:n],
                           z_what_emb[:, 1][idx].squeeze()[:n],
                           c=colr, label=label_list[relevant_labels[i]].replace("_", " "),
                           alpha=0.7)
        else:  # 3
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            n = 1000  # too heavy otherwise
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor("red")
            for i, idx in enumerate(sorted):
                ax.scatter(z_what_emb[:, 0][idx].squeeze()[:n],
                           z_what_emb[:, 1][idx].squeeze()[:n],
                           z_what_emb[:, 2][idx].squeeze()[:n],
                           c=colors[i], label=label_list[i].replace("_", " "),
                           alpha=0.7)

        plt.legend(prop={'size': 6})
        plt.savefig(
            f"../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}.png")

    else:
        # relevant = torch.zeros(labels.shape, dtype=torch.bool)
        # for rl in relevant_labels:
        #     relevant |= labels == rl
        # z_what = z_what[relevant]
        # labels = labels[relevant]
        # train_portion = 0.8
        # nb_sample = int(train_portion * len(labels))
        # test_x = z_what[nb_sample:]
        # test_y = labels[nb_sample:]
        # if arguments['method'] == "linear":
        #     few_shot_accuracy = {}
        #     z_what_by_game = {rl: z_what[labels == rl] for rl in relevant_labels}
        #     labels_by_game = {rl: labels[labels == rl] for rl in relevant_labels}
        #     for train_portion in [1, 4, 16, 64]:
        #         current_train_sample = torch.cat([z_what_by_game[rl][:train_portion] for rl in relevant_labels])
        #         current__train_labels = torch.cat([labels_by_game[rl][:train_portion] for rl in relevant_labels])
        #         clf = LogisticRegression()
        #         clf.fit(current_train_sample, current__train_labels)
        #         acc = clf.score(test_x, test_y)
        #         few_shot_accuracy[train_portion] = acc
        #     return few_shot_accuracy
        if arguments['method'] == "nn":
            net = Network().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
            # optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
            train_portion = 0.95
            nb_sample = int(train_portion * len(labels))
            train_x = z_what[:nb_sample]
            test_x = z_what[nb_sample:]
            train_y = labels[:nb_sample]
            test_y = labels[nb_sample:]

            for i in range(10000):
                output_train = net(train_x)
                optimizer.zero_grad()
                loss = F.cross_entropy(output_train, train_y)
                loss.backward()
                pred = output_train.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels[:nb_sample].view_as(pred)).sum().item()
                optimizer.step()
                if i % 40 == 0:
                    output_test = net(test_x)
                    test_loss = F.cross_entropy(output_test, test_y)
                    pred_te = output_test.argmax(dim=1, keepdim=True)
                    correct_te = pred_te.eq(test_y.view_as(pred_te)).sum().item()
                    print("-" * 13)
                    print("Train loss: ", round(loss.item(), 5))
                    print("Test loss: ", round(test_loss.item(), 5))
                    print("Train Acc : ", round(correct / len(pred) * 100, 2))
                    print("Test Acc : ", round(correct_te / len(pred_te) * 100, 2))
                # import ipdb; ipdb.set_trace()
        # elif arguments['method'] == "kmeans":
        # clf = KMeans(n_clusters=len(relevant_labels))
        # y = clf.fit_predict(z_what)
        # results = {'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(labels, y),
        #            'adjusted_rand_score': metrics.adjusted_rand_score(labels, y),
        #            'homogeneity_score': metrics.homogeneity_score(labels, y),
        #            'completeness_score': metrics.completeness_score(labels, y),
        #            'v_measure_score': metrics.v_measure_score(labels, y),
        #            'fowlkes_mallows_score': metrics.fowlkes_mallows_score(labels, y)}
        # centroids = clf.cluster_centers_
        # X = z_what.numpy()
        # nn = NearestNeighbors(n_neighbors=N_NEIGHBORS).fit(X)
        # _, z_w_idx = nn.kneighbors(centroids)
        # centroid_label = []
        # for cent, nei in zip(centroids, z_w_idx):
        #     count = {rl: 0 for rl in relevant_labels}
        #     for i in range(N_NEIGHBORS):
        #         nei_label = relevant_labels[labels[nei[i]]]
        #         count[nei_label] += 1
        #         if count[nei_label] > 3.0 / (i + 1) - i / (N_NEIGHBORS - 1) and nei_label not in centroid_label:
        #             centroid_label.append(nei_label)
        #             break
        # train_all = torch.cat((z_what, labels.unsqueeze(1)), 1)
        # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'purple', 'orange',
        #           'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
        # # SORT THE INDICES
        # sorted = []
        # for i in relevant_labels:
        #     mask = train_all.T[-1] == i
        #     indices = torch.nonzero(mask)
        #     sorted.append(indices)
        # pca = PCA(n_components=arguments['dim'])
        # z_what_emb = pca.fit_transform(z_what.numpy())
        #
        # fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(20, 12)
        # ax.set_facecolor((0.3, 0.3, 0.3))
        # n = min(n, 10000)
        # for i, idx in enumerate(sorted):
        #     if "pacman" in label_list:
        #         colr = [np.array(base_objects_colors[label_list[relevant_labels[i]]]) / 255]
        #     else:
        #         colr = colors[relevant_labels[i]]
        #     ax.scatter(z_what_emb[:, 0][idx].squeeze()[:n],
        #                z_what_emb[:, 1][idx].squeeze()[:n],
        #                c=colr, label=label_list[relevant_labels[i]].replace("_", " "),
        #                alpha=0.7)
        # centroid_emb = pca.transform(centroids)
        #
        # for c_emb, cl in zip(centroid_emb, centroid_label):
        #     if "pacman" in label_list:
        #         colr = [np.array(base_objects_colors[label_list[cl]]) / 255]
        #     else:
        #         colr = colors[cl] * 0.8
        #     ax.scatter([c_emb[0]],
        #                [c_emb[1]],
        #                c=colr,
        #                edgecolors='black', s=100)
        #
        # plt.legend(prop={'size': 6})
        # plt.savefig(
        #     f"../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}.png")
        # return results, f"../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}.png"
        else:
            print("Invalid method supplied!")


all_train_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/train_labels.csv")
all_validation_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/validation_labels.csv")

label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
              "white_ghost", "fruit", "save_fruit", "life", "life2", "score0"]


def get_labels_validation(idx, boxes_batch):
    labels = []
    for i, boxes in zip(idx, boxes_batch):
        labels.append(get_labels(all_validation_labels.iloc[[i]], boxes))
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Look at the z_what and classify them')
    parser.add_argument('type', metavar='N', type=str,
                        choices=["show", "classify"],
                        help='an integer for the accumulator')
    parser.add_argument('-dim', type=int, choices=[2, 3], default=2,
                        help='Number of dimension for PCA/TSNE visualization')
    parser.add_argument('-folder', type=str, default="mspacman_atari+z_what_1e-2",
                        help='config-file for pca')
    parser.add_argument('-method', choices=["pca", "tsne", "linear", "nn"],
                        default="pca", type=str,
                        help='Method : "pca", "tsne" for show \n \
                        "linear", "nn" for classify')
    parser.add_argument('-indices', type=str, default=None,
                        help='The relevant objects by their index')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.type == "nn" else 'cpu')

    nb_used_sample = 500

    z_what_train = torch.cat(torch.load(f"labeled_tr/{args.folder}/z_what_validation.pt"))
    train_labels = torch.cat(torch.load(f"labeled_tr/{args.folder}/labels_validation.pt"))

    classify_z_what(args, z_what_train, train_labels, nb_used_sample, cfg=None)
