from sklearn.metrics import confusion_matrix
from tabulate import tabulate


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
