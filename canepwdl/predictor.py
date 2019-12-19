from __future__ import division
from keras.models import load_model
from keras_contrib.layers import InstanceNormalization
import csv
import mlflow

try:
    from itertools import izip as zip
except ImportError:
    pass
import canepwdl.utils as utils
from canepwdl.algorithms.utils.AttentionWithContext import AttentionWithContext


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
            for process_instance, time_deltas, event_id in zip(
                    preprocessor.data_structure['data']['test']['process_instances'],
                    preprocessor.data_structure['data']['test']['time_deltas'],
                    preprocessor.data_structure['data']['test']['event_ids']):

                cropped_process_instance, cropped_time_deltas, cropped_context_attributes = preprocessor.get_cropped_instance(
                    prefix_size,
                    index,
                    process_instance, time_deltas)
                index = index + 1

                # make no prediction for this case, since this case has ended already
                encoded_end_mark = preprocessor.get_encoded_end_mark()
                if encoded_end_mark in cropped_process_instance:
                    continue

                ground_truth = process_instance[prefix_size:prefix_size + prediction_size]
                prediction = []

                # predict only next activity (i = 1)
                for i in range(prediction_size):

                    if len(ground_truth) <= i:
                        continue

                    test_data = preprocessor.get_data_tensor_for_single_prediction(
                        args,
                        cropped_process_instance,
                        cropped_time_deltas,
                        cropped_context_attributes,
                        preprocessor.data_structure
                    )

                    y = model.predict(test_data)
                    y_char = y[0][:]

                    predicted_event = preprocessor.get_event_type(y_char)

                    cropped_process_instance.append(predicted_event)
                    prediction.append(predicted_event)

                    if predicted_event == encoded_end_mark:
                        print('! predicted, end of process instance ... \n')
                        break

                output = []
                if len(ground_truth) > 0:
                    # TODO: how to encode results in order to use following evaluation metrics
                    # print(preprocessor.data_structure['support']['map_event_label_to_event_id'][ground_truth[0]])
                    # print(preprocessor.data_structure['support']['map_event_label_to_event_id'][prediction[0]])

                    output.append(event_id)
                    output.append(prefix_size)
                    output.append(str([int(i) for i in list(ground_truth[0])]).encode("utf-8"))
                    output.append(str([int(i) for i in list(prediction[0])]).encode("utf-8"))

                    result_writer.writerow(output)

    mlflow.log_artifact(result_dir_fold)
