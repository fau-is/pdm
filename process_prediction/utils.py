import argparse
import pickle
import sys
import numpy
import csv
import sklearn
import arrow
import os

output = {
    "accuracy_values": [],
    "accuracy_value": 0.0,
    "precision_values": [],
    "precision_value": 0.0,
    "recall_values": [],
    "recall_value": 0.0,
    "f1_values": [],
    "f1_value": 0.0,
    "training_time_seconds": []
}


def load_output():
    return output


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
    open('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task), "w").close()


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
                if int(row[1]) == prefix or prefix_all_enabled == 1:
                    ground_truth_label.append(row[2])
                    predicted_label.append(row[3])

    _output["accuracy_values"].append(sklearn.metrics.accuracy_score(ground_truth_label, predicted_label))
    _output["precision_values"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='weighted'))
    _output["recall_values"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='weighted'))
    _output["f1_values"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted'))

    # ToDo: calc confusion matrix
    # util.llprint(confusion_matrix(ground_truth_label, predicted_label))

    return _output


def print_output(args, _output, index_fold):

    if args.cross_validation and index_fold < args.num_folds:
        llprint("\nAccuracy of fold %i: %f\n" % (index_fold, _output["accuracy_values"][index_fold]))
        llprint("Precision of fold %i: %f\n" % (index_fold, _output["precision_values"][index_fold]))
        llprint("Recall of fold %i: %f\n" % (index_fold, _output["recall_values"][index_fold]))
        llprint("F1-Score of fold %i: %f\n" % (index_fold, _output["f1_values"][index_fold]))
        llprint("Training time of fold %i: %f seconds\n\n" % (index_fold, _output["training_time_seconds"][index_fold]))

    else:
        llprint("\nAccuracy avg: %f\n" % (avg(_output["accuracy_values"])))
        llprint("Precision avg: %f\n" % (avg(_output["precision_values"])))
        llprint("Recall avg: %f\n" % (avg(_output["recall_values"])))
        llprint("F1-Score avg: %f\n" % (avg(_output["f1_values"])))
        llprint("Training time avg: %f seconds" % (avg(_output["training_time_seconds"])))


def get_mode(index_fold, args):
    """ Gets the mode. """

    if index_fold == -1:
        return "split-%s" % args.split_rate_test
    elif index_fold != args.num_folds:
        return "fold%s" % index_fold
    else:
        return "avg"


def get_output_value(_mode, _index_fold, _output, measure, args):
    """ If fold < max number of folds in cross validation than use a specific value, else avg works. In addition, this holds for split. """

    if _mode != "split-%s" % args.split_rate_test and _mode != "avg":
        return _output[measure][_index_fold]
    else:
        return avg(_output[measure])


def write_output(args, _output, index_fold):
    """ Writes the output. """

    with open('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task), mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')

        # if file is empty
        if os.stat('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task)).st_size == 0:
            writer.writerow(
                ["experiment", "mode", "validation", "accuracy", "precision", "recall", "f1-score", "training-time",
                 "time-stamp"])
        writer.writerow([
            # todo: add to experiment name the encoding identifier
            "%s-%s" % (args.data_set[:-4], args.dnn_architecture),  # experiment
            get_mode(index_fold, args),  # mode
            "cross-validation" if args.cross_validation else "split-validation",  # validation
            get_output_value(get_mode(index_fold, args), index_fold, _output, "accuracy_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "training_time_seconds", args),
            arrow.now()
        ])
