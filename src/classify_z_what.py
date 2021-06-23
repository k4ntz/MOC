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
parser.add_argument('-method', choices=["pca", "tsne", "linear", "nn"],
                    default="pca", type=str,
                    help='Method : "pca", "tsne" for show \n \
                    "linear", "nn" for classify')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() and args.type=="nn" else 'cpu')

nb_used_sample = 100

z_what_train = torch.stack(torch.load("labeled_tr/z_what_validation.pt")).to(device)[:nb_used_sample]
train_labels = torch.stack(torch.load("labeled_tr/labels_validation.pt")).to(device)[:nb_used_sample]

print(train_labels)
print(z_what_train.shape)
# z_what_test = torch.load("labeled/z_what_test.pt").to(device)[:nb_used_sample]
# test_labels = torch.load("labeled/labels_test.pt").to(device)[:nb_used_sample]

label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
              "white_ghost", "fruit", "save_fruit", "life", "life2", "score0"]



if args.type == "show":
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    train_all = torch.cat((z_what_train, train_labels.unsqueeze(1)), 1)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'purple', 'orange',
              'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
    # SORT THE INDICES
    sorted = []
    for i in range(train_labels.max() + 1):
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
        # ax.set_facecolor('xkcd:salmon')
        ax.set_facecolor((0.3, 0.3, 0.3))
        nb_used_sample = min(nb_used_sample, 10000)
        for i, idx in enumerate(sorted):
            if "pacman" in label_list:
                colr = [np.array(base_objects_colors[label_list[i]])/255]
            else:
                colr = colors[i]
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:nb_used_sample],
                       z_what_emb[:, 1][idx].squeeze()[:nb_used_sample],
                       c=colr, label=label_list[i].replace("_", " "),
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

    plt.legend(prop={'size': 19})
    plt.show()

else:
    if args.method == "linear":
        for nb_sample in [300, 500, 800, 1000, 3000]:
            clf = LogisticRegression()
            # clf = LinearRegression()
            # clf = SGDClassifier()
            clf.fit(z_what_train[:nb_sample], train_labels[:nb_sample])
            print("-" * 15)
            print(f"nb samples: {nb_sample}")
            acc = clf.score(z_what_test, test_labels)
            print("Acc :", round(acc, 2))

    elif args.method == "nn":
        net = Network().to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

        for i in range(3000):
            output_train = net(z_what_train)
            output_test = net(z_what_test)
            optimizer.zero_grad()
            # import ipdb; ipdb.set_trace()
            loss = F.cross_entropy(output_train, train_labels)
            loss.backward()
            test_loss = F.cross_entropy(output_test, test_labels)
            pred = output_train.argmax(dim=1, keepdim=True)
            pred_te = output_test.argmax(dim=1, keepdim=True)
            correct = pred.eq(train_labels.view_as(pred)).sum().item()
            correct_te = pred_te.eq(test_labels.view_as(pred_te)).sum().item()
            optimizer.step()
            # scheduler.step()
            if i % 400 == 0:
                print("-"*13)
                print("Train loss: ", round(loss.item(), 5))
                print("Test loss: ", round(test_loss.item(), 5))
                print("Train Acc : ", round(correct/len(pred)*100, 2))
                print("Test Acc : ", round(correct_te/len(pred_te)*100, 2))
            # import ipdb; ipdb.set_trace()
