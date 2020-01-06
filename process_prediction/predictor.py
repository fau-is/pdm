from __future__ import division
from keras.models import load_model
import csv

try:
    from itertools import izip as zip
except ImportError:
    pass
import process_prediction.utils as utils
import numpy as numpy


def predict(args, preprocessor):
    # assumption: load the model of the first fold
    model = load_model('%smodel_%s.h5' % (args.checkpoint_dir, 0))

    # select a random prefix
    process_instances = preprocessor.data_structure['data']['process_instances']
    labels = preprocessor.data_structure['data']['labels']

    # prefix of process instance must have a size of >= 2
    while True:
        index = numpy.random.randint(0, len(process_instances))
        prefix_size = numpy.random.randint(0, len(process_instances[index]))
        if prefix_size >= 2:
            break

    prefix = process_instances[index][0:prefix_size]
    label = labels[index][prefix_size]

    # output
    print("Process instance %i with prefix size %i: %s" % (index, prefix_size, prefix))
    print("Label: %s" % label)

    cropped_process_instance, cropped_process_instance_label = preprocessor.get_cropped_instance(
        prefix_size,
        process_instances[index],
        labels[index]
    )

    ground_truth = cropped_process_instance_label
    test_data = preprocessor.get_data_tensor_for_single_prediction(cropped_process_instance)


    y = model.predict(test_data)
    y = y[0][:]

    prediction = preprocessor.get_class_val(y)
    test_data_reshaped = test_data.reshape(-1, test_data.shape[2])

    return prediction, ground_truth, prefix, model, test_data_reshaped


def test(args, preprocessor):
    # init
    preprocessor.get_instances_of_fold('test')
    model = load_model('%smodel_%s.h5' % (args.checkpoint_dir, preprocessor.data_structure['support']['iteration_cross_validation']))

    # output
    data_set_name = args.data_set.split('.csv')[0]
    result_dir_generic = args.result_dir + data_set_name + "__" + args.task
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

            utils.llprint("Process instance %i of %i \n" % (index, len(preprocessor.data_structure['data']['test']['process_instances'])))
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

                prediction = preprocessor.get_class_val(y)

                result_writer.writerow([
                    event_id,
                    prefix_size,
                    str(ground_truth).encode("utf-8"),
                    str(prediction).encode("utf-8")
                ])
