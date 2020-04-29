from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder
import numpy as np


def pretty_print_confusion_matrix(target, predicted, label_map):
    """
    Returns a nicely formatted confusion matrix ready for printing
    :return:
    """

    cnf_mat = confusion_matrix(target, predicted)

    table = [[] for idx in range(len(label_map))]

    labels = [None for idx in range(len(label_map))]

    for key, label in label_map.items():
        idx = int(key)
        table[idx] = [label] + cnf_mat[idx, :].tolist()
        labels[idx] = label

    return cnf_mat, tabulate(table, headers=labels)


def check_is_in(needles, haystack):
    if needles.shape[0] != haystack.shape[0]:
        raise ValueError("Needles and Haystack shape mismatch")

    result = np.zeros((needles.shape[0], ), dtype=bool)

    for idx in range(haystack.shape[0]):
        result[idx] = np.isin(needles[idx], haystack[idx, :]).reshape(1, )[0]

    return result


def top_n_score(y_target, y_pred, class_labels, top_n=10, weighted=True):
    """
    This method returns the top_n_score.
    The top_n_score returns 1 if the target_label is in the first n predictions in y_pred
    :param y_target: This is an array of target labels with shape (n_samples,)
    :param y_pred: This is an array of predicted probabilities with shape (n_samples, n_classes)
    :param class_labels: This is the list of all possible classes
    :param top_n: This determines how many predictions to consider
    :param weighted: Return the raw score or weighted by the number of samples
    :return:
    """

    if top_n >= len(class_labels):
        top_n -= 1

    labelbin = LabelEncoder()
    labelbin.fit(class_labels)

    encoded_labels = labelbin.transform(y_target)

    sorted_prob = np.argsort(-y_pred, axis=1)

    top_n_predictions = sorted_prob[:, :top_n]
    encoded_probability = np.take_along_axis(y_pred, encoded_labels[:, None], axis=1)
    encoded_probability = encoded_probability.reshape(encoded_probability.shape[0], )

    bool_top_n = check_is_in(encoded_labels, top_n_predictions)
    combined = np.logical_and(bool_top_n, (encoded_probability > 0))

    score = sum(combined) if not weighted else sum(combined)/combined.shape[0]
    return score
