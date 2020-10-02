from __future__ import division
import csv
import process_prediction.utils as utils
from tensorflow.keras.models import load_model


def test(args, event_log, preprocessor, test_indices_per_fold):
    # TODO description

    # init
    model = load_model('%s%smodel_%s.h5' % (
    args.task, args.model_dir[1:], preprocessor.iteration_cross_validation))

    cases_of_fold = preprocessor.get_cases_of_fold(event_log, test_indices_per_fold)

    # start prediction
    with open(get_result_dir_fold(args, preprocessor), 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(["Case_id", "Prefix_length", "Groud_truth", "Predicted"])

        for idx_case, case in enumerate(cases_of_fold, 1):
            utils.llprint("Case %i of %i \n" % (idx_case, len(cases_of_fold)))

            # for each prefix with a length >= 2 TODO prefix size? comment and implementation differ..
            # for prefix_size in range(1, len(process_instance) + 1):

            for prefix_size in range(1, len(case) + 1):
                subseq = preprocessor.get_subsequence_of_case(case, prefix_size)

                ground_truth = subseq[-1][args.outcome_key]
                features = preprocessor.get_features_tensor(args, 'test', event_log, [subseq])

                predicted_label = predict_label(model, features, preprocessor)

                result_writer.writerow([
                    case._list[0].get(args.case_id_key),
                    prefix_size,
                    str(ground_truth).encode("utf-8"),
                    str(predicted_label).encode("utf-8")
                ])

def predict_label(model, features, preprocessor):
    """
    Predicts and returns a label.

    Parameters
    ----------
    model : keras.engine.training.Model
        The model to predict classes.
    ndarray : shape[S, T, E], S is number of samples, T is number of time steps, E is number of embedding values.
        The features of a single sample of the test set.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    tuple :
        Represents a predicted encoded class.

    """
    y = model.predict(features)
    y = y[0][:]
    predicted_label = preprocessor.get_class_label(y)

    return predicted_label

def get_result_dir_fold(args, preprocessor):
    """
    Returns result directory of a fold.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    str :
        Directory of the result file for the current fold.

    """

    result_dir_generic = './' + args.task + args.result_dir[1:] + args.data_set.split('.csv')[0]
    result_dir_fold = result_dir_generic + "_%d%s" % (preprocessor.iteration_cross_validation, ".csv")

    return result_dir_fold