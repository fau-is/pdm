import argparse
import pickle
import sys
import numpy
import csv
import sklearn
import arrow
import os
import mlflow
import pandas
from functools import reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import seaborn as sns
import uuid
import copy

output = {
    "accuracy_values": [],
    "accuracy_value": 0.0,
    "precision_micro_values": [],
    "precision_micro_value": 0.0,
    "precision_macro_values": [],
    "precision_macro_value": 0.0,
    "precision_weighted_values": [],
    "precision_weighted_value": 0.0,
    "recall_micro_values": [],
    "recall_micro_value": 0.0,
    "recall_macro_values": [],
    "recall_macro_value": 0.0,
    "recall_weighted_values": [],
    "recall_weighted_value": 0.0,
    "f1_micro_values": [],
    "f1_micro_value": 0.0,
    "f1_macro_values": [],
    "f1_marco_value": 0.0,
    "f1_weighted_values": [],
    "f1_weighted_value": 0.0,
    "auc_roc_values": [],
    "auc_roc_value": 0.0,
    "auc_prc_values": [],
    "auc_prc_value": 0.0,
    "training_time_seconds": []
}


remote_server_uri = "databricks"  # databricks" set to your server URI
experiment = "/Users/sven.weinzierl@fau.de/dss-paper-bpi2013i"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(experiment)



def load(path):
    return pickle.load(open(path, 'rb'))


def load_output():
    return copy.deepcopy(output)


def avg(numbers):
    if len(numbers) == 0:
        return sum(numbers)

    return sum(numbers) / len(numbers)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def load(path):
    return pickle.load(open(path, 'rb'))


def onehot(index, size):
    vec = numpy.zeros(int(size), dtype=numpy.float32)
    vec[int(index)] = 1.0
    return vec


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    if os.path.exists('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task)):
        open('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task), "w").close()


def check_directory_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_output(args, preprocessor, _output):
    prefix = 0
    prefix_all_enabled = 1

    predicted_label = list()
    ground_truth_label = list()

    if not args.cross_validation:
        result_dir_fold = \
            args.result_dir + \
            args.data_set.split(".csv")[0] + \
            "__" + args.task + \
            "_0.csv"
    else:
        result_dir_fold = \
            args.result_dir + \
            args.data_set.split(".csv")[0] + \
            "__" + args.task + \
            "_%d" % preprocessor.data_structure['support']['iteration_cross_validation'] + ".csv"

    with open(result_dir_fold, 'r') as result_file_fold:
        result_reader = csv.reader(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(result_reader)

        for row in result_reader:
            if not row:
                continue
            else:
                if row[1] == prefix or prefix_all_enabled == 1:
                    ground_truth_label.append(row[2])
                    predicted_label.append(row[3])

    _output["accuracy_values"].append(sklearn.metrics.accuracy_score(ground_truth_label, predicted_label))
    _output["precision_micro_values"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='micro'))
    _output["precision_macro_values"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='macro'))
    _output["precision_weighted_values"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='weighted'))
    _output["recall_micro_values"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='micro'))
    _output["recall_macro_values"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='macro'))
    _output["recall_weighted_values"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='weighted'))
    _output["f1_micro_values"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='micro'))
    _output["f1_macro_values"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='macro'))
    _output["f1_weighted_values"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted'))
    save_confusion_matrix(ground_truth_label, predicted_label, args)

    try:
        _output["auc_roc_values"].append(multi_class_roc_auc_score(ground_truth_label, predicted_label))
    except:
        print("Warning: Roc auc score can not be calculated ...")

    try:
        # we use the average precision at different threshold values as auc of the pr-curve
        # and not the auc-pr-curve with the trapezoidal rule / linear interpolation, because it could be too optimistic
        _output["auc_prc_values"].append(multi_class_prc_auc_score(ground_truth_label, predicted_label))
    except:
        print("Warning: Auc prc score can not be calculated ...")

    return _output


def split_train_test(data_set):
    return sklearn.model_selection.train_test_split(data_set, data_set, test_size=0.1, random_state=0)


def save_confusion_matrix(ground_truth_label, predicted_label, args):

    classes = sklearn.utils.multiclass.unique_labels(ground_truth_label, predicted_label)
    cms = []
    cm = sklearn.metrics.confusion_matrix(ground_truth_label, predicted_label)
    cm_df = pandas.DataFrame(cm, index=classes, columns=classes)
    cms.append(cm_df)

    def prettify(n):
        if n > 1000000:
            return str(numpy.round(n / 1000000, 1)) + 'M'
        elif n > 1000:
            return str(numpy.round(n / 1000, 1)) + 'K'
        else:
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
    fig.savefig(str(args.result_dir + 'cm.pdf'))
    pyplot.close()


def multi_class_prc_auc_score(ground_truth_label, predicted_label, average='weighted'):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(ground_truth_label)

    ground_truth_label = label_binarizer.transform(ground_truth_label)
    predicted_label = label_binarizer.transform(predicted_label)

    return sklearn.metrics.average_precision_score(ground_truth_label, predicted_label, average=average)


def multi_class_roc_auc_score(ground_truth_label, predicted_label, average='weighted'):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(ground_truth_label)

    ground_truth_label = label_binarizer.transform(ground_truth_label)
    predicted_label = label_binarizer.transform(predicted_label)

    return sklearn.metrics.roc_auc_score(ground_truth_label, predicted_label, average=average)


def print_output(args, _output, index_fold):
    if args.cross_validation and index_fold < args.num_folds:
        llprint("\nAccuracy of fold %i: %f\n" % (index_fold, _output["accuracy_values"][index_fold]))
        llprint("Precision (micro) of fold %i: %f\n" % (index_fold, _output["precision_micro_values"][index_fold]))
        llprint("Precision (macro) of fold %i: %f\n" % (index_fold, _output["precision_macro_values"][index_fold]))
        llprint("Precision (weighted) of fold %i: %f\n" % (index_fold, _output["precision_weighted_values"][index_fold]))
        llprint("Recall (micro) of fold %i: %f\n" % (index_fold, _output["recall_micro_values"][index_fold]))
        llprint("Recall (macro) of fold %i: %f\n" % (index_fold, _output["recall_macro_values"][index_fold]))
        llprint("Recall (weighted) of fold %i: %f\n" % (index_fold, _output["recall_weighted_values"][index_fold]))
        llprint("F1-score (micro) of fold %i: %f\n" % (index_fold, _output["f1_micro_values"][index_fold]))
        llprint("F1-score (macro) of fold %i: %f\n" % (index_fold, _output["f1_macro_values"][index_fold]))
        llprint("F1-score (weighted) of fold %i: %f\n" % (index_fold, _output["f1_weighted_values"][index_fold]))
        llprint("Auc-roc of fold %i: %f\n" % (index_fold, _output["auc_roc_values"][index_fold]))
        llprint("Auc-prc of fold %i: %f\n" % (index_fold, _output["auc_prc_values"][index_fold]))
        llprint("Training time of fold %i: %f seconds\n\n" % (index_fold, _output["training_time_seconds"][index_fold]))

    else:
        llprint("\nAccuracy avg: %f\n" % (avg(_output["accuracy_values"])))
        llprint("Precision (micro) avg: %f\n" % (avg(_output["precision_micro_values"])))
        llprint("Precision (macro) avg: %f\n" % (avg(_output["precision_macro_values"])))
        llprint("Precision (weighted) avg: %f\n" % (avg(_output["precision_weighted_values"])))
        llprint("Recall (micro) avg: %f\n" % (avg(_output["recall_micro_values"])))
        llprint("Recall (macro) avg: %f\n" % (avg(_output["recall_macro_values"])))
        llprint("Recall (weighted) avg: %f\n" % (avg(_output["recall_weighted_values"])))
        llprint("F1-score (micro) avg: %f\n" % (avg(_output["f1_micro_values"])))
        llprint("F1-score (macro) avg: %f\n" % (avg(_output["f1_macro_values"])))
        llprint("F1-score (weighted) avg: %f\n" % (avg(_output["f1_weighted_values"])))
        llprint("Auc-roc avg: %f\n" % (avg(_output["auc_roc_values"])))
        llprint("Auc-prc avg: %f\n" % (avg(_output["auc_prc_values"])))
        llprint("Training time avg: %f seconds\n\n" % (avg(_output["training_time_seconds"])))


def get_mode(index_fold, args):
    """ Gets the mode - split, fold or avg. """

    if index_fold == -1:
        return "split-%s" % args.split_rate_test
    elif index_fold != args.num_folds:
        return "fold%s" % index_fold
    else:
        return "avg"


def get_output_value(_mode, _index_fold, _output, measure, args):
    """ If fold < max number of folds in cross validation than use a specific value, else avg works. In addition,
    this holds for split. """

    if _mode != "split-%s" % args.split_rate_test and _mode != "avg":
        mlflow.log_metric(measure, _output[measure][_index_fold])
        return _output[measure][_index_fold]
    else:
        mlflow.log_metric(measure, avg(_output[measure]))
        return avg(_output[measure])



def write_output(args, _output, index_fold):
    """ Writes the output. """

    with open('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task), mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')

        # if file is empty
        if os.stat('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task)).st_size == 0:
            writer.writerow(
                ["experiment[ds-cat_enc-dl_arch]", "mode", "validation", "accuracy",
                 "precision (micro)", "precision (macro)", "precision (weighted)",
                 "recall (micro)", "recall (macro)", "recall (weighted)",
                 "f1-score (micro)", "f1-score (macro)", "f1-score (weighted)",
                 "auc-roc", "auc-prc", "training-time",
                 "time-stamp"])
        writer.writerow([
            "%s-%s-%s" % (args.data_set[:-4], args.encoding_cat, args.dnn_architecture),  # experiment
            get_mode(index_fold, args),  # mode
            "cross-validation" if args.cross_validation else "split-validation",  # validation
            get_output_value(get_mode(index_fold, args), index_fold, _output, "accuracy_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_micro_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_macro_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_weighted_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_micro_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_macro_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_weighted_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_micro_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_macro_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_weighted_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "auc_roc_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "auc_prc_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "training_time_seconds", args),
            arrow.now()
        ])

def get_experiments_configuration(args):
    exp_df = pandas.read_csv(args.experiments_config_file)
    exp_dicts = exp_df.to_dict('index')

    return exp_dicts


def set_single_experiment_config(args, exp_dict_tuple):
    experiment_config = exp_dict_tuple[1]

    args.data_set = experiment_config['data_set']
    args.dnn_architecture = experiment_config['dnn_architecture']
    args.batch_size_train = experiment_config['batch_size_train']
    args.encoding_cat = experiment_config['encoding_cat']

    start_parent_run_mlflow(args)

    return args


def start_parent_run_mlflow(args):
    mlflow.start_run(run_name="experiment_%d" % uuid.uuid1())

    mlflow.log_param("dataset", args.data_set[:-4])
    mlflow.log_param("categorical_encoding", args.encoding_cat)
    mlflow.log_param("deep-learning_architecture", args.dnn_architecture)
    mlflow.log_param("validation", "cross-validation" if args.cross_validation else "split-validation")


def end_run_mlflow():
    mlflow.end_run()


def start_nested_run_mlflow(args, index_fold):
    if index_fold < args.num_folds:
        if index_fold == -1:
            mlflow.start_run(nested=True, run_name="run")
        else:
            mlflow.start_run(nested=True, run_name="run_%d" % index_fold)
    else:
        mlflow.start_run(nested=True, run_name="runs_results")
        mlflow.log_artifact('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task))

    mlflow.log_param("mode", get_mode(index_fold, args))


def log_at_end_of_nested_run_mlflow(args, index_fold):

    if index_fold < args.num_folds:
        mlflow.log_artifact('./results/cm.pdf')
        mlflow.log_artifact('./results/model.pdf')