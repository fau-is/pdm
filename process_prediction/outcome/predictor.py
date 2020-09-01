from __future__ import division
import csv
import process_prediction.utils as utils
from tensorflow.keras.models import load_model


def predict_prefix(args, preprocessor, process_instance, labels, prefix_size, model):

    # todo: check this statement; remove end event
    # remove label of outcome2
    labels = labels[0:len(labels) - 1]
    process_instance = process_instance[0:len(process_instance) - 1]

    cropped_process_instance, cropped_process_instance_label = preprocessor.get_cropped_instance(
        prefix_size,
        process_instance,
        labels
    )

    ground_truth = cropped_process_instance_label
    test_data = preprocessor.get_data_tensor_for_single_prediction(cropped_process_instance)

    y = model.predict(test_data)
    y = y[0][:]

    y = y.tolist()
    prediction = str(y.index(max(y)))
    test_data_reshaped = test_data.reshape(-1, test_data.shape[2])

    """
    if prediction == '1':
        prediction = '0'
    else:
        prediction = '1'
    """

    return prediction, ground_truth, cropped_process_instance, model, test_data_reshaped


def test(args, preprocessor):
    # init
    preprocessor.get_instances_of_fold('test')
    model = load_model('%s%smodel_%s.h5' % (
    args.task, args.model_dir[1:], preprocessor.data_structure['support']['iteration_cross_validation']))

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
        for process_instance, event_id, process_instance_labels in zip(
                preprocessor.data_structure['data']['test']['process_instances'],
                preprocessor.data_structure['data']['test']['event_ids'],
                preprocessor.data_structure['data']['test']['labels']):

            utils.llprint("Process instance %i of %i \n" % (
            index, len(preprocessor.data_structure['data']['test']['process_instances'])))
            index = index + 1

            # for each prefix with a length >= 2
            for prefix_size in range(2, len(process_instance)):
                cropped_process_instance, cropped_process_instance_label = preprocessor.get_cropped_instance(
                    prefix_size,
                    process_instance,
                    process_instance_labels
                )

                ground_truth = cropped_process_instance_label
                test_data = preprocessor.get_data_tensor_for_single_prediction(cropped_process_instance)

                y = model.predict(test_data)
                y = y[0][:]
                # print(y)

                prediction = preprocessor.get_class_val(y)

                result_writer.writerow([
                    event_id,
                    prefix_size,
                    str(ground_truth).encode("utf-8"),
                    str(prediction).encode("utf-8")
                ])
