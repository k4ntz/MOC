import torch
import torch.nn.functional as F
from simple_net import Network
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import argparse
from utils import base_objects_colors
import numpy as np

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

nb_used_sample = 100

z_what_train = torch.cat(torch.load(f"labeled_tr/{args.folder}/z_what_validation.pt"))
train_labels = torch.cat(torch.load(f"labeled_tr/{args.folder}/labels_validation.pt"))

label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
              "white_ghost", "fruit", "save_fruit", "life", "life2", "score0"]
relevant_labels = [int(part) for part in args.indices.split(',')] if args.indices else range(train_labels.max() + 1)
print(z_what_train.shape)
print(train_labels)
if args.type == "show":
    print("showing...")
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    train_all = torch.cat((z_what_train, train_labels.unsqueeze(1)), 1)
    print(train_all.shape)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'purple', 'orange',
              'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
    # SORT THE INDICES
    sorted = []
    for i in relevant_labels:
        mask = train_all.T[-1] == i
        indices = torch.nonzero(mask)
        sorted.append(indices)
    if args.method == "pca":
        pca = PCA(n_components=args.dim)
        z_what_emb = pca.fit_transform(z_what_train.numpy())
    else:
        try:
            from MulticoreTSNE import MulticoreTSNE as TSNE

            if args.dim != 2:
                print("Un cuml, only dim = 2 is supported")
        except ImportError:
            print("Install cuml if GPU available to improve speed (requires conda)")
            from sklearn.manifold import TSNE
        tsne = TSNE(n_jobs=4, n_components=args.dim, verbose=True)
        z_what_emb = tsne.fit_transform(z_what_train.numpy())

    if args.dim == 2:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(20, 12)
        ax.set_facecolor((0.3, 0.3, 0.3))
        nb_used_sample = min(nb_used_sample, 10000)
        for i, idx in enumerate(sorted):
            if "pacman" in label_list:
                colr = [np.array(base_objects_colors[label_list[relevant_labels[i]]]) / 255]
            else:
                colr = colors[relevant_labels[i]]
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:nb_used_sample],
                       z_what_emb[:, 1][idx].squeeze()[:nb_used_sample],
                       c=colr, label=label_list[relevant_labels[i]].replace("_", " "),
                       alpha=0.7)
    else:  # 3
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        nb_used_sample = 1000  # too heavy otherwise
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("red")
        for i, idx in enumerate(sorted):
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:nb_used_sample],
                       z_what_emb[:, 1][idx].squeeze()[:nb_used_sample],
                       z_what_emb[:, 2][idx].squeeze()[:nb_used_sample],
                       c=colors[i], label=label_list[i].replace("_", " "),
                       alpha=0.7)

    plt.legend(prop={'size': 6})
    plt.savefig(f"../output/logs/{args.folder}1/pca{args.indices if args.indices else ''}.png")

else:
    if args.method == "linear":
        for clf in [LogisticRegression(), LinearRegression(), SGDClassifier()]:
            for nb_sample in range(500, 4501, 1000):
                clf.fit(z_what_train[:nb_sample], train_labels[:nb_sample])
                print("-" * 15)
                print(f"classifier: {clf.__class__.__name__} nb samples: {nb_sample}")
                acc = clf.score(z_what_train[nb_sample:], train_labels[nb_sample:])
                print("Acc :", round(acc, 2))

    elif args.method == "nn":
        net = Network().to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
        train_portion = 0.95
        relevant = torch.zeros(train_labels.shape, dtype=torch.bool)
        for rl in relevant_labels:
            relevant |= train_labels == rl
        print(z_what_train.shape)
        z_what_train = z_what_train[relevant]
        print(z_what_train.shape)
        train_labels = train_labels[relevant]
        nb_sample = int(train_portion * len(train_labels))
        train_x = z_what_train[:nb_sample]
        test_x = z_what_train[nb_sample:]
        train_y = train_labels[:nb_sample]
        test_y = train_labels[nb_sample:]

        for i in range(10000):
            output_train = net(train_x)
            optimizer.zero_grad()
            loss = F.cross_entropy(output_train, train_y)
            loss.backward()
            pred = output_train.argmax(dim=1, keepdim=True)
            correct = pred.eq(train_labels[:nb_sample].view_as(pred)).sum().item()
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
