from __future__ import division
from keras.models import load_model
import csv
try:
    from itertools import izip as zip
except ImportError:
    pass
import process_prediction.utils as utils


def predict_prefix(preprocessor, process_instance, labels, model):

    # process_instance = ['NEW', 'RELEASE', 'DELETE']
    ground_truth = labels[-1]
    test_data = preprocessor.get_data_tensor_for_single_prediction(process_instance)

    y = model.predict(test_data)
    y = y[0][:]

    y = y.tolist()
    # print(y)
    prediction = str(y.index(max(y)))

    test_data_reshaped = test_data.reshape(-1, test_data.shape[2])

    """
    if prediction == '1':
        prediction = '0'
    else:
        prediction = '1'
    """

    return prediction, ground_truth, process_instance, model, test_data_reshaped


def test(args, preprocessor):
    # init
    preprocessor.get_instances_of_fold('test')
    model = load_model('%s%smodel_%s.h5' % (args.task, args.model_dir[1:], preprocessor.data_structure['support']['iteration_cross_validation']))

    # output
    data_set_name = args.data_set.split('.csv')[0]
    result_dir_generic = './' + args.task + args.result_dir[1:] + data_set_name
    result_dir_fold = result_dir_generic + "_%d%s" % (
        preprocessor.data_structure['support']['iteration_cross_validation'], ".csv")

    # start prediction
    with open(result_dir_fold, 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            ["Case_id", "Prefix_length", "Groud_truth", "Predicted"])

        # for each process instance
        index = 0
        for process_instance, event_id, process_instance_label in zip(
                    preprocessor.data_structure['data']['test']['process_instances'],
                    preprocessor.data_structure['data']['test']['event_ids'],
                    preprocessor.data_structure['data']['test']['labels']):

            utils.llprint("Process instance %i of %i \n" % (index, len(preprocessor.data_structure['data']['test']['process_instances'])))
            index = index + 1


            ground_truth = process_instance_label
            test_data = preprocessor.get_data_tensor_for_single_prediction(process_instance)

            y = model.predict(test_data)
            y = y[0][:]

            prediction = preprocessor.get_class_val(y)

            result_writer.writerow([
                event_id,
                -1,
                str(ground_truth).encode("utf-8"),
                str(prediction).encode("utf-8")
            ])
