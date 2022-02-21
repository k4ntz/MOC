import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score
from .utils import flatten
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import torch
import time

# License: BSD 3 clause
# Options for incorporating labels:
# 1. Add Variables soft-bound to (0, 1) for each object + label
# 2. Let simple classifier accuracy influence R and/or H
# 3. RNN over former label distributions
# 4. Gaussian NB seems to provide useful probabilities
# 5. Currently, "not an object" is first non-position entry in x, which is different
#    from classifying something as no_label
# 6. Only accuracy of detected is determined, no recall

def basic_classifier(cfg, observations, labels, few_shot):
    c = Counter(labels.tolist())
    if cfg.train.log:
        print("Distribution of matched labels:", c)
    train_portion = 0.9
    nb_sample = int(train_portion * len(labels))
    relevant_labels = list(c.keys())
    test_x = observations[nb_sample:]
    test_y = labels[nb_sample:]
    train_x = observations[:nb_sample]
    train_y = labels[:nb_sample]
    z_what_by_game = {rl: train_x[train_y == rl] for rl in relevant_labels}
    labels_by_game = {rl: train_y[train_y == rl] for rl in relevant_labels}
    current_train_sample = np.concatenate([z_what_by_game[rl][:few_shot] for rl in relevant_labels])
    current_train_labels = np.concatenate([labels_by_game[rl][:few_shot] for rl in relevant_labels])
    scaler = StandardScaler()
    X = scaler.fit_transform(current_train_sample)
    test_x = scaler.transform(test_x)
    clf = RandomForestClassifier(max_depth=5)
    clf.fit(X, current_train_labels)
    return clf, clf.score(test_x, test_y), relevant_labels, scaler


def classify_encodings(cfg, observations, labels, few_shot=4):
    """
    :param cfg: config
    :param observations: (B, T, N*, D+4) where N is those entries that have z_pres as
    :param labels: (B, T, N*)
    :param few_shot: how many encodings by class to train the simple classifier on
    :return:
        b_accuracy: a scalar. accuracy of simple classifier
        log: a dictionary for visualization
    """

    flat_obs = flatten(observations).detach().cpu()
    if flat_obs.nelement() / 36 < len(observations) * 4:
        return 0
    b_classifier, b_score, relevant_labels, scaler = basic_classifier(cfg,
                                                                      flat_obs.numpy(),
                                                                      flatten(labels).detach().cpu().numpy(),
                                                                      few_shot=few_shot)
    np.set_printoptions(suppress=True)
    filter_size = len(relevant_labels) + 3
    test_labels = []
    test_predictions = []
    test_predictions_non_kalman = []
    assignment = None
    target_classes = [-1] + [c for c in b_classifier.classes_]
    for data_point, label_point in zip(observations, labels):
        filters = [default_kalman(filter_size) for _ in range(max([len(objects) for objects in data_point]) + 1)]
        c_obj = None

        for objects in data_point:
            try:
                c_obj = scaler.transform(torch.nan_to_num(objects).detach().cpu().numpy())
                probabilities = b_classifier.predict_proba(c_obj)
                assignment = step(filters, c_obj, probabilities, filter_size, label_point)
            except Exception as e:
                print(e)
                pass
        if data_point[-1].nelement == 0:
            continue
        try:
            by_filters = predict_by_filters(data_point[-1], assignment, filters, target_classes)
            predict = b_classifier.predict(c_obj)
        except Exception as e:
            print(e)
            continue
        test_labels.extend(label_point[-1])
        test_predictions.extend(by_filters)
        test_predictions_non_kalman.extend(predict)

    test_labels = [t_l.item() for t_l in test_labels]
    # print(f"Score: {accuracy_score(test_labels, test_predictions)}")
    # print(f"Score w/o Kalman: {accuracy_score(test_labels, test_predictions_non_kalman)}")
    # kalman_cm = confusion_matrix(test_labels, test_predictions)
    # not_kalman_cm = confusion_matrix(test_labels, test_predictions_non_kalman)
    # print(f"Per Class: {kalman_cm.diagonal() / kalman_cm.sum(axis=1)}")
    # print(f"Per Class w/o Kalman: {not_kalman_cm.diagonal() / not_kalman_cm.sum(axis=1)}")
    return accuracy_score(test_labels, test_predictions)


def predict_by_filters(objects, assignment, filters, target_classes):
    predictions = []
    for _, assign in zip(objects, assignment):
        squeeze = filters[assign].x[2:].squeeze()
        predictions.append(target_classes[np.argmax(squeeze)])
    return predictions


def default_kalman(filter_size):
    kalman_filter = KalmanFilter(dim_x=filter_size, dim_z=filter_size)  # +2 position variables +1 for not an object
    kalman_filter.x = np.random.randn(filter_size)
    kalman_filter.F = np.eye(filter_size)
    kalman_filter.P = np.eye(filter_size) * 5.0
    kalman_filter.Q = np.eye(filter_size) * 0.5
    kalman_filter.H = np.eye(filter_size)
    return kalman_filter


def hungarian_matching(x, obs):
    cost = distance_matrix(x, obs)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind, cost[row_ind, col_ind]


def distance_matrix(x, obs):
    distances = squareform(pdist(np.append(x, obs, axis=0)))
    return distances[len(x):, :len(x)]


def step(filters, objects, probabilities, filter_size, test_labels):
    x = np.array([f.x[:2] for f in filters])  # filter positions
    objects = objects[:, 2:4]  # object positions
    objects = np.concatenate([objects, np.ones((len(filters) - len(objects), 2)) * -10], axis=0)
    assignment, costs = hungarian_matching(x, objects)
    # print("First: ", costs, objects[:2])
    fill_probabilities = np.append(probabilities,
                                   np.ones((len(filters), probabilities.shape[1])) / probabilities.shape[1], axis=0)
    filter_idx = 4
    # print(f'{filter_idx=}')
    for obj, matching_idx, matching_cost, probs in zip(objects, assignment, costs, fill_probabilities):
        f = filters[matching_idx]
        if obj[0] > -5 and obj[1] > -5:
            not_an_object_chance = 0.001
        else:
            not_an_object_chance = 0.999
        # if filter_idx == matching_idx:
            # print(not_an_object_chance, not_an_object_chance * f.x[:2] + (1 - not_an_object_chance) * obj)
        f.R = (1 + not_an_object_chance * 5) * np.eye(filter_size)
        obs = np.concatenate((not_an_object_chance * f.x[:2] + (1 - not_an_object_chance) * obj,
                              np.array([not_an_object_chance]),
                              probs * (1 - not_an_object_chance)))[..., np.newaxis]
        f.predict()
        f.update(obs)
    as_list = [a for a in assignment]
    # print(filters[filter_idx].x, test_labels, as_list.index(filter_idx), costs[as_list.index(filter_idx)], fill_probabilities[as_list.index(filter_idx)])
    return assignment


if __name__ == "__main__":
    observations = [np.random.randn(n * 2, 1)]
    for _ in range(5):
        observations.append(observations[-1] + np.random.randn(n * 2, 1) * 0.001)
