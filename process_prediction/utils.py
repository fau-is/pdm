import argparse
import sys
import numpy as np
import csv
import sklearn
import arrow
import os
from matplotlib import pyplot
from functools import reduce
import seaborn as sns
import pandas
import tensorflow.keras.backend as K
import tensorflow as tf

output = {
    "accuracy_values": [],
    "accuracy_value": 0.0,
    "precision_values_micro": [],
    "precision_value_micro": 0.0,
    "precision_values_macro": [],
    "precision_value_macro": 0.0,
    "precision_values_weighted": [],
    "precision_value_weighted": 0.0,
    "recall_values_micro": [],
    "recall_value_micro": 0.0,
    "recall_values_macro": [],
    "recall_value_macro": 0.0,
    "recall_values_weighted": [],
    "recall_value_weighted": 0.0,
    "f1_values_micro": [],
    "f1_value_micro": 0.0,
    "f1_values_macro": [],
    "f1_value_macro": 0.0,
    "f1_values_weighted": [],
    "f1_value_weighted": 0.0,
    "auc_roc_values": [],
    "auc_roc_value": 0.0,
    "training_time_seconds": []
}


def set_seeds(args):
    """
    Sets random seeds for reproducible results.
    Returns
    -------
    None
    """
    np.random.seed(args.seed_val)
    tf.random.set_seed(args.seed_val)


def load_output():
    return output


def avg(numbers):
    if len(numbers) == 0:
        return sum(numbers)
    return sum(numbers) / len(numbers)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    open('./%s/results/output_%s.csv' % (args.task, args.data_set[:-4]), "w").close()
    open('./%s/results/output_class%s.csv' % (args.task, args.data_set[:-4]), "w").close()


def get_output(args, _output, predicted_distributions, ground_truths):
    prefix = 0
    prefix_all_enabled = 1

    predicted_label = list()
    ground_truth_label = list()

    result_dir_fold = './' + args.task + args.result_dir[1:] + args.data_set.split(".csv")[0] + "_0.csv"

    with open(result_dir_fold, 'r') as result_file_fold:
        result_reader = csv.reader(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(result_reader)

        for row in result_reader:
            if not row:
                continue
            else:
                if int(row[1]) == prefix or prefix_all_enabled == 1:
                    ground_truth_label.append(row[2])
                    predicted_label.append(row[3])

    _output["accuracy_values"].append(sklearn.metrics.accuracy_score(ground_truth_label, predicted_label))
    _output["precision_values_micro"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='micro'))
    _output["precision_values_macro"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='macro'))
    _output["precision_values_weighted"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='weighted'))
    _output["recall_values_micro"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='micro'))
    _output["recall_values_macro"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='macro'))
    _output["recall_values_weighted"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='weighted'))
    _output["f1_values_micro"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='micro'))
    _output["f1_values_macro"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='macro'))
    _output["f1_values_weighted"].append(
        sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted'))
    _output["auc_roc_values"].append(binary_class_roc_auc_score(predicted_distributions, ground_truths))

    save_confusion_matrix(ground_truth_label, predicted_label, args)
    save_classification_report(ground_truth_label, predicted_label, args)

    return _output


def binary_class_roc_auc_score(prob_dist, ground_truth_one_hot, average='macro'):

    ground_truth_one_hot = [e[1] for e in ground_truth_one_hot]
    prob_dist = [e[1] for e in prob_dist]

    try:
        value = sklearn.metrics.roc_auc_score(ground_truth_one_hot, prob_dist, average=average)
    except:
        value = 0

    return value


def print_output(_output, index_fold):

    llprint("\nAccuracy: %f\n" % _output["accuracy_values"][index_fold])
    llprint("Precision (micro): %f\n" % _output["precision_values_micro"][index_fold])
    llprint("Precision (macro): %f\n" % _output["precision_values_macro"][index_fold])
    llprint("Precision (weighted): %f\n" % _output["precision_values_weighted"][index_fold])
    llprint("Recall (micro): %f\n" % _output["recall_values_micro"][index_fold])
    llprint("Recall (macro): %f\n" % _output["recall_values_macro"][index_fold])
    llprint("Recall (weighted): %f\n" % _output["recall_values_weighted"][index_fold])
    llprint("F1-Score (micro): %f\n" % _output["f1_values_micro"][index_fold])
    llprint("F1-Score (macro): %f\n" % _output["f1_values_macro"][index_fold])
    llprint("F1-Score (weighted): %f\n" % _output["f1_values_weighted"][index_fold])
    llprint("Auc-Roc (macro): %f\n" % _output["auc_roc_values"][index_fold])
    llprint("Training time: %f seconds\n\n" % _output["training_time_seconds"][index_fold])



def get_mode(index_fold, args):

    if index_fold == -1:
        return "split-%s" % args.split_rate_train
    elif index_fold != args.num_folds:
        return "fold%s" % index_fold
    else:
        return "avg"


def get_output_value(_mode, _index_fold, _output, measure, args):

    if _mode != "split-%s" % args.split_rate_train and _mode != "avg":
        return _output[measure][_index_fold]
    else:
        return avg(_output[measure])


def write_output(args, _output, index_fold):

    with open('./%s%soutput_%s.csv' % (args.task, args.result_dir[1:], args.data_set[:-4]), mode='a',
              newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')

        if os.stat('./%s%soutput_%s.csv' % (args.task, args.result_dir[1:], args.data_set[:-4])).st_size == 0:
            writer.writerow(
                ["experiment", "mode", "validation", "accuracy",
                 "precision_micro", "precision_macro", "precision_weighted",
                 "recall_micro", "recall_macro", "recall_weighted",
                 "f1-score_micro", "f1-score_macro", "f1-score_weighted",
                 "auc_roc_macro",
                 "training-time",
                 "time-stamp"])
        writer.writerow([
            "%s" % (args.data_set[:-4]),
            get_mode(index_fold, args),
            "split-validation",
            get_output_value(get_mode(index_fold, args), index_fold, _output, "accuracy_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_values_micro", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_values_macro", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_values_weighted", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_values_micro", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_values_macro", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_values_weighted", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_values_micro", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_values_macro", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_values_weighted", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "auc_roc_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "training_time_seconds", args),
            arrow.now()
        ])


def save_classification_report(ground_truth_label, predicted_label, args):
    report = sklearn.metrics.classification_report(ground_truth_label, predicted_label, output_dict=True)

    file = open('./%s%soutput_class_%s.csv' % (args.task, args.result_dir[1:], args.data_set[:-4]), mode='a',
                newline='')
    file.write("\n" + str(report))
    file.close()


def save_confusion_matrix(label_ground_truth, label_prediction, args):

    label_ground_truth = np.array(label_ground_truth)
    label_prediction = np.array(label_prediction)
    classes = sklearn.utils.multiclass.unique_labels(label_ground_truth, label_prediction)

    cms = []
    cm = sklearn.metrics.confusion_matrix(label_ground_truth, label_prediction)
    cm_df = pandas.DataFrame(cm, index=classes, columns=classes)
    cms.append(cm_df)

    def prettify(n):
        return str(n)

    cm = reduce(lambda x, y: x.add(y, fill_value=0), cms)
    annot = cm.applymap(prettify)
    cm = (cm.T / cm.sum(axis=1)).T
    fig, g = pyplot.subplots(figsize=(7, 4.5))
    g = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, rasterized=True, linewidths=0.1)
    _ = g.set(ylabel='Actual', xlabel='Prediction')

    for _, spine in g.spines.items():
        spine.set_visible(True)

    pyplot.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig("./outcome" + str(args.result_dir + 'cm_' + args.data_set[:-4] + '.pdf'))


def f1_score(y_true, y_pred):
    """
    Computes the f1 score - performance indicator for the prediction accuracy.
    The F1 score is the harmonic mean of the precision and recall.
    The evaluation metric to be optimized during hyper-parameter optimization.

    Parameters
    ----------
    y_true : Tensor, dtype=float32
        True labels.
    y_pred : Tensor, dtype=float32
        Predicted labels.

    Returns
    -------
    Tensor : dtype=float32
        The F1-Score.

    """

    def recall(y_true, y_pred):
        """
        Computes the recall (only a batch-wise average of recall), a metric for multi-label classification of
        how many relevant items are selected.

        Parameters
        ----------
        y_true : Tensor, dtype=float32
            True labels.
        y_pred : Tensor, dtype=float32
            Predicted labels.

        Returns
        -------
        Tensor : dtype=float32
            The recall.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Computes the precision (only a batch-wise average of precision), a metric for multi-label classification of
        how many selected items are relevant.

        Parameters
        ----------
        y_true : Tensor, dtype=float32
            True labels.
        y_pred : Tensor, dtype=float32
            Predicted labels.

        Returns
        -------
        Tensor : dtype=float32
            The precision.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)


    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))  # epsilon to avoid division by 0
