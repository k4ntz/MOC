#from src.engine.train import train
import sys
import pickle
from matplotlib import offsetbox
from numpy.core.numeric import isclose
from numpy.lib.ufunclike import isposinf
from torch.serialization import save
sys.path.append("../")
from model.space.space import Space
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import itertools
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import CheckButtons
import cv2
import math
import matplotlib.patches as mpatches
import hoverUtils as hu

N_NEIGHBORS = 24
DISPLAY_CENTROIDS = False

warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import numpy as np

base_objects_colors = {"sue": (180, 122, 48), "inky": (84, 184, 153),
                       "pinky": (198, 89, 179), "blinky": (200, 72, 72),
                       "pacman": (210, 164, 74), "fruit": (184, 50, 50),
                       "save_fruit": (144, 10, 10),
                       "white_ghost": (214, 214, 214),
                       "blue_ghost": (66, 114, 194), "score0": (200, 50, 200),
                       "life": (70, 210, 70), "life2": (20, 150, 20)
                       }


    
#this instance is needed to set plot settings for hovering
hoverPlot = hu.HoverUtils()

def evaluate_z_what(arguments, z_what, labels, image_refs_tuple, nb_samples, cfg):
    #can basically ignore
    relevant_labels = [int(part) for part in arguments['indices'].split(',')] if arguments['indices'] else range(
        labels.max() + 1)
    folder = f'hyper/{cfg.exp_name}{cfg.seed}' if cfg else f'{arguments["folder"]}1'
    relevant = torch.zeros(labels.shape, dtype=torch.bool)
    for rl in relevant_labels:
        relevant |= labels == rl
    z_what = z_what[relevant]
    labels = labels[relevant]
    relevantLst = relevant.tolist()
    
    #needed to load images later when hovering, dont want to load all images
    image_refs_path = image_refs_tuple[0]
    #actual names of images to be loaded
    image_refs = image_refs_tuple[1]
    image_refs = list(itertools.compress(image_refs, relevantLst))
    #until here
    train_portion = 0.8
    nb_sample = int(train_portion * len(labels))
    test_x = z_what[nb_sample:]
    test_y = labels[nb_sample:]
    #image_refs = image_refs[nb_sample:]
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
    clf.fit(z_what)
    y = clf.predict(z_what)
    dists_to_center = clf.transform(z_what)
    #y = clf.fit_predict(z_what)



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
    numOfElements = 0


    for i in relevant_labels:
        gt_mask = train_all.T[-1] == i
        indices = torch.nonzero(gt_mask)
        sorted.append(indices)
        sorted = sorted

        #hoverIndices = np.random.choice(indices.shape[0], n, replace=False)f
        #toHoverOn.append(indices[hoverIndices].squeeze().numpy())

        numOfElements +=indices.size()[0]
    
    pca = PCA(n_components=arguments['dim'])
    z_what_emb = pca.fit_transform(z_what.numpy())
    xPoints = z_what_emb[:, 0]
    yPoints = z_what_emb[:, 1]

    #safe for every index/point the respective color index (N,)
    categories = np.zeros(numOfElements)


    wantsHover = arguments['hover']

    if arguments['edgecolors']:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(20, 12)
        ax.set_facecolor((0.3, 0.3, 0.3))
        plt.suptitle("Labeled PCA of z_whats", y=0.96, fontsize=28)
        plt.title("Inner Color is GT, Outer is greedy Centroid-based label", fontsize=18, pad=20)
        nb_samples = min(nb_samples, 10000)
        #for all labels the corresponding color (N_labels, 3)
        colorMap = []
        #for all points the corresponding edge color (N, 3)
        edgeColors = np.zeros((numOfElements, 3))
        labelList = []
        #firstIndicies = []
        for i, idx in enumerate(sorted):
            #firstIndicies.append(idx[0].numpy())

            #add respective colors number to indices
            categories[idx] = i
            if "pacman" in label_list:
                colorMap.append([np.array(base_objects_colors[label_list[relevant_labels[i]]]) / 255])
                edgeColors[idx] = np.array([[np.array(base_objects_colors[label_list[centroid_label[assign[0]]]]) / 255 for assign in
                                  y[idx]]]).reshape((-1, 1, 3))
                labelList.append(label_list[relevant_labels[i]].replace("_", " "))
            else:
                colorMap.append(colors[relevant_labels[i]])
                edgeColors[idx] = np.array([colors[centroid_label[assign[0]]] for assign in y[idx]]).reshape((-1,1,3))


        colorMap = np.array(colorMap).squeeze()
        categories = np.array(categories).astype(int)
        edgeColors = np.array(edgeColors)
        #firstIndicies = np.array(firstIndicies)
    

        #new hover points technique 
        toHoverOn = hu.hoverPicker(dists_to_center, edgeColors, colorMap[categories])


        """ ax.scatter(xPoints.squeeze()[firstIndicies],
                    yPoints.squeeze()[firstIndicies],
                    c=colorMap[categories[firstIndicies]].squeeze(), label=labelList,
                    alpha=0.7) """
        plot = ax.scatter(xPoints.squeeze(),
                    yPoints.squeeze(),
                    c=colorMap[categories],
                    alpha=0.7, edgecolors=edgeColors, s=100, linewidths=2)
        recs = []
        for i in range(0, len(colorMap)):
            recs.append(mpatches.Rectangle((0,0), 1,1, fc=colorMap[i]))
        
        plt.legend(recs,labelList, loc=2)
        hoverPlot.setPlot(plot)

        
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

        #hovering stuff
        #indices of points that we want to highlight
        #https://matplotlib.org/3.1.0/gallery/widgets/check_buttons.html
        #show only highlighted dots
        #toHoverOn = np.array(toHoverOn).flatten()


        directory = f"../../output/logs/{folder}"
        if not os.path.exists(directory):
            print(f"Writing PCA to {directory}")
            os.makedirs(directory)
        plt.savefig(
            f"../../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}.png")
        
        if wantsHover:
            buttonAxes = plt.axes([0, 0.4, 0.1, 0.15])
            buttonAxes.set_facecolor("grey")
            highlightButton = CheckButtons(buttonAxes, ["Hover"])

            hoverPlot.setData(xPoints, yPoints)
            highlightButton.on_clicked(lambda _: hoverPlot.highlight(_, highlightButton, ax, toHoverOn, colorMap, categories, edgeColors))

            #https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point
            #array of images 
            images_dict, pointIndexList = hu.loadHighlighted(toHoverOn, image_refs, image_refs_path, xPoints, yPoints)
            im = OffsetImage(next(iter(images_dict.values())), zoom = 5)
            xybox = (5., 5.)
            annotBox = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
                    boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
            ax.add_artist(annotBox)
            annotBox.set_visible(False)



            fig.canvas.mpl_connect("motion_notify_event", lambda event: hoverPlot.hoverImage(event, fig, annotBox, xybox, pointIndexList, im, images_dict, image_refs))
            plt.show()


    else:
        #for all labels the corresponding color (N_labels, 3)
        colorMap = []
        #for all points the corresponding edge color (N, 3)
        edgeColors = np.zeros((numOfElements, 3))
        labelList = []

        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(30, 12)


        axs[0].set_facecolor((0.3, 0.3, 0.3))
        axs[1].set_facecolor((0.3, 0.3, 0.3))
        axs[0].set_title("Ground Truth Labels")
        axs[1].set_title("Labels Following Clustering")
        for i, idx in enumerate(sorted):
            #add respective colors number to indices
            categories[idx] = i
            if "pacman" in label_list:
                colorMap.append([np.array(base_objects_colors[label_list[relevant_labels[i]]]) / 255])
                edgeColors[idx] = np.array([[np.array(base_objects_colors[label_list[centroid_label[assign[0]]]]) / 255 for assign in
                                  y[idx]]]).reshape((-1, 1, 3))
                labelList.append(label_list[relevant_labels[i]].replace("_", " "))
            else:
                #colr = colors[relevant_labels[i]]
                #edge_colors = [colors[centroid_label[assign[0]]] for assign in y[idx]]
                colorMap.append(colors[relevant_labels[i]])
                edgeColors[idx] = np.array([colors[centroid_label[assign[0]]] for assign in y[idx]]).reshape((-1,1,3))


        colorMap = np.array(colorMap).squeeze()
        categories = np.array(categories).astype(int)
        edgeColors = np.array(edgeColors)

        recs = []
        for i in range(0, len(colorMap)):
            recs.append(mpatches.Rectangle((0,0), 1,1, fc=colorMap[i]))
        

        axs[0].scatter(xPoints.squeeze(), yPoints.squeeze(), c=colorMap[categories], alpha=0.7)
        plot = axs[1].scatter(xPoints.squeeze(), yPoints.squeeze(), c=edgeColors, alpha=0.7)
        hoverPlot.setPlot(plot)



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
            axs[0].legend(recs,labelList, loc=2)
            axs[1].legend(recs,labelList, loc=2)
        

        if not os.path.exists(f"../../output/logs/{folder}"):
            os.makedirs(f"../../output/logs/{folder}")
        plt.tight_layout()
        plt.savefig(
            f"../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}.png")
        #plt.close(fig)

        #hovering stuff 
        if wantsHover:
            toHoverOn = hu.hoverPicker(dists_to_center, edgeColors, colorMap[categories])
            highlightButton = CheckButtons(axs[1], ["POI"])

            hoverPlot.setData(xPoints, yPoints)
            highlightButton.on_clicked(lambda _: hoverPlot.highlight(_, highlightButton, axs[1], toHoverOn, colorMap, categories, edgeColors))

            #https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point
            #array of images 
            images_dict, pointIndexList = hu.loadHighlighted(toHoverOn, image_refs, image_refs_path, xPoints, yPoints)
            im = OffsetImage(next(iter(images_dict.values())), zoom = 5)
            xybox = (50., 50.)
            annotBox = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
                    boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
            axs[1].add_artist(annotBox)
            annotBox.set_visible(False)
            fig.canvas.mpl_connect("motion_notify_event", lambda event: hoverPlot.hoverImage(event, fig, annotBox, xybox, pointIndexList, im, images_dict))
            plt.show()

    return results, f"../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}.png", \
           few_shot_accuracy


all_train_labels = pd.read_csv(f"/home/patrick/Desktop/SPACE/aiml_atari_data/rgb/MsPacman-v0/train_labels.csv")
all_validation_labels = pd.read_csv(f"/home/patrick/Desktop/SPACE/aiml_atari_data/rgb/MsPacman-v0/validation_labels.csv")

label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
              "white_ghost", "fruit", "save_fruit", "life", "life2", "score0"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the z_what encoding')
    

    parser.add_argument('-folder', type=str, default="mspacman_atari_example",
                        help='the output folder')
    parser.add_argument('-indices', type=str, default=None,
                        help='The relevant objects by their index, e.g. \"0,1\" for Pacman and Sue')
    parser.add_argument('-edgecolors', type=bool, default=False,
                        help='True iff the ground truth labels and the predicted labels '
                             '(Mixture of some greedy policy and NN) should be drawn in the same image')
    parser.add_argument('-dim', type=int, choices=[2, 3], default=2,
                        help='Number of dimension for PCA/TSNE visualization')

    parser.add_argument("-hover", type=bool, default=True, help='If you want to further inspect'
                        'the PCA plot and how z_what is correlated to the input image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nb_used_sample = 500
    #import ipdb; ipdb.set_trace()
    z_what_train = torch.load(f"../labeled/{args.folder}/z_what_validation.pt")
    train_labels = torch.load(f"../labeled/{args.folder}/labels_validation.pt")
    image_refs = torch.load(f"../labeled/{args.folder}/all_images_refs_validation.pt")
    org_path = "../../aiml_atari_data/space_like/MsPacman-v0/train"

    image_refs = [ref for singleLst in image_refs for ref in singleLst]
    image_refs_tuple = (org_path, image_refs)

    z_what_train = torch.cat(z_what_train)
    train_labels = torch.cat(train_labels)



    evaluate_z_what(vars(args), z_what_train, train_labels, image_refs_tuple, nb_used_sample, cfg=None)
