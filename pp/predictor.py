from __future__ import division
from keras.models import load_model
from keras_contrib.layers import InstanceNormalization
import csv
import mlflow

try:
    from itertools import izip as zip
except ImportError:
    pass
import pp.utils as utils
from pp.algorithms.utils.AttentionWithContext import AttentionWithContext


def test(args, preprocessor):
    preprocessor.set_instances_of_fold('test')

    # https://github.com/keras-team/keras-contrib
    custom_objects = {'InstanceNormalization': InstanceNormalization, 'AttentionWithContext': AttentionWithContext}
    model = load_model(
        '%smodel_%s.h5' % (args.checkpoint_dir, preprocessor.data_structure['support']['iteration_cross_validation']),
        custom_objects)

    prediction_size = 1

    # set options for result output
    data_set_name = args.data_set.split('.csv')[0]
    result_dir_generic = args.result_dir + data_set_name + "__" + args.task
    result_dir_fold = result_dir_generic + "_%d%s" % (
        preprocessor.data_structure['support']['iteration_cross_validation'], ".csv")

    # start prediction
    with open(result_dir_fold, 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            ["Event", "Prefix length", "Ground truth", "Predicted"])

        # for each prefix_size
        for prefix_size in range(2, preprocessor.data_structure['meta']['max_length_process_instance']):
            utils.llprint("Prefix size: %d\n" % prefix_size)

            index = 0
            for process_instance, time_deltas, outcome_value, id_process_instance in zip(
                    preprocessor.data_structure['data']['test']['process_instances'],
                    preprocessor.data_structure['data']['test']['time_deltas'],
                    preprocessor.data_structure['data']['test']['outcome_values'],
                    preprocessor.data_structure['data']['test']['ids_process_instances']):

                cropped_process_instance, cropped_time_deltas, cropped_context_attributes, cropped_outcome_values = preprocessor.get_cropped_instance(
                    prefix_size,
                    index,
                    process_instance,
                    time_deltas,
                    outcome_value)
                index = index + 1

                ground_truth = outcome_value[prefix_size:prefix_size + prediction_size]
                prediction = []

                # predict only next activity's outcome (i = 1)
                for i in range(prediction_size):

                    if len(ground_truth) <= i:
                        continue

                    test_data = preprocessor.get_data_tensor_for_single_prediction(
                        args,
                        cropped_process_instance,
                        cropped_time_deltas,
                        cropped_context_attributes,
                        cropped_outcome_values,
                        preprocessor.data_structure
                    )

                    y = model.predict(test_data)
                    y_char = y[0][:]

                    predicted_outcome = preprocessor.get_outcome_type(y_char)

                    cropped_outcome_values.append(predicted_outcome)
                    prediction.append(predicted_outcome)

                output = []
                if len(ground_truth) > 0:

                    output.append(id_process_instance)
                    output.append(prefix_size)
                    output.append(str([int(i) for i in list(ground_truth[0])]).encode("utf-8"))
                    output.append(str([int(i) for i in list(prediction[0])]).encode("utf-8"))

                    result_writer.writerow(output)

    mlflow.log_artifact(result_dir_fold)
