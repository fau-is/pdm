import process_prediction.config as config
import process_prediction.utils as utils
from process_prediction.explanation.LSTM.LSTM_bidi import *
from process_prediction.explanation.util.heatmap import html_heatmap
import process_prediction.explanation.util.browser as browser
from keras.models import load_model


def apply_lrp(args, prefix_heatmaps, predicted_class, model, input_embedded, prefix_words):
    target_class = predicted_class

    if isinstance(target_class, str):
        target_class = int(target_class)

    # compute lrp relevances
    # LRP hyperparameters:
    eps = 0.001  # small positive number
    bias_factor = 0.0  # recommended value
    net = LSTM_bidi(args, model, input_embedded)  # load trained LSTM model
    Rx, Rx_rev, R_rest = net.lrp(prefix_words, target_class, eps, bias_factor)  # perform LRP
    R_words = np.sum(Rx + Rx_rev, axis=1)  # compute word-level LRP relevances
    scores = net.s.copy()  # classification prediction scores

    """
    print("prediction scores:", scores)
    print("\nLRP target class:", target_out_class)
    print("\nLRP relevances:")
    for idx, w in enumerate(prefix_words):
        print("\t\t\t" + "{:8.10f}".format(R_words[idx]) + "\t" + w)
    print("\nLRP heatmap:")
    """

    prefix_heatmaps = prefix_heatmaps + html_heatmap(prefix_words, R_words) + "<br>"

    # How to sanity check global relevance conservation:
    bias_factor = 1.0  # value to use for sanity check
    Rx, Rx_rev, R_rest = net.lrp(prefix_words, target_class, eps, bias_factor)  # prefix -> w_indices
    R_tot = Rx.sum() + Rx_rev.sum() + R_rest.sum()  # sum of all "input" relevances

    """
    print(R_tot)
    print("Sanity check passed? ", np.allclose(R_tot, net.s[target_out_class]))
    """

    browser.display_html(prefix_heatmaps)

    return prefix_heatmaps


def create_output(args, process_instance, label, out_preprocessor, out_model, act_preprocessor, act_model,
                  out2_preprocessor, out2_model):

    prefix_heatmaps_out = ""
    # predict no 1
    for prefix_index in range(2, len(process_instance)):
        predicted_out_class, target_out_class, prefix_words, out_model, out_input_embedded = out_test.predict_prefix(
            args, out_preprocessor, process_instance, label, prefix_index, out_model)
        apply_lrp(args, prefix_heatmaps_out, predicted_out_class, out_model, out_input_embedded, prefix_words)
        predicted_act_class, target_act_class, _, _, _, prob_event_types = act_test.predict_prefix(args,
                                                                                                   act_preprocessor,
                                                                                                   process_instance,
                                                                                                   prefix_index,
                                                                                                   act_model)
        print("Prefix: %s; Outcome prediction: %s; Outcome target: %s" % (
            prefix_index, predicted_out_class, target_out_class))
        print("Prefix: %s; Next act prediction: %s; Next act target: %s" % (
            prefix_index, predicted_act_class, target_act_class))
        print(prob_event_types)
    # predict not 2
    predicted_out2_class, target_out_class, prefix_words, out2_model, out2_input_embedded = out2_test.predict_prefix(
        out2_preprocessor, process_instance, label, out2_model)
    apply_lrp(args, "", predicted_out2_class, out2_model, out2_input_embedded, prefix_words)
    print("All; Outcome prediction: %s; Outcome target: %s" % (predicted_out_class, target_out_class))


if __name__ == '__main__':
    args = config.load()
    output = utils.load_output()
    utils.clear_measurement_file(args)

    # explanation mode for outcome prediction
    if args.explain:
        from process_prediction.outcome.preprocessor import Preprocessor as Out_Preprocessor
        from process_prediction.outcome2.preprocessor import Preprocessor as Out2_Preprocessor
        from process_prediction.nextevent.preprocessor import Preprocessor as Act_Preprocessor
        import process_prediction.outcome.predictor as out_test
        import process_prediction.outcome2.predictor as out2_test
        import process_prediction.nextevent.predictor as act_test

        out_preprocessor = Out_Preprocessor(args)
        out2_preprocessor = Out2_Preprocessor(args)
        act_preprocessor = Act_Preprocessor(args)

        """
        Overview of cases
        Case 1: a process instance is conform -> indicator 1 = 0 for each time step; indicator 2 = 0 at last time step; 
        Case 2: a process instance is not conform -> indicator 1 = 1 at time step k; indicator 2 = 0 at last time step;
        Case 3: a process instance is not conform -> indicator 1 = 0 for each time step; indicator 2 = 1 at last time step;
        Case 4 a process instance is not conform -> indicator 1 = 1 at time step k; indicator 2 = 1 at last time step;
        """

        # Load models of first fold
        act_model = load_model('%s%smodel_%s.h5' % ("nextevent", args.model_dir[1:], 0))
        out_model = load_model('%s%smodel_%s.h5' % ("outcome", args.model_dir[1:], 0))
        out2_model = load_model('%s%smodel_%s.h5' % ("outcome2", args.model_dir[1:], 1))

        process_instances, labels = out_preprocessor.get_process_instance()  # get process instance

        # Case 1 ########################################################################################
        print("Case 1: process is conform (1=False; 2=False)")
        select_index = 0
        select = False

        for index in range(0, len(process_instances)):
            if len(process_instances[index]) >= 3:
                if sum([int(val) for val in labels[index]]) == 0:
                    is_correct = True
                    # predict not 1
                    for prefix_index in range(2, len(process_instances[index])):
                        predicted_out_class, _, _, _, _ = out_test.predict_prefix(args, out_preprocessor,
                                                                                  process_instances[index], labels[index],
                                                                                  prefix_index, out_model)
                        if predicted_out_class != labels[index][prefix_index]:
                            is_correct = False
                    # predict not 2
                    predicted_out2_class, _, _, _, _ = out2_test.predict_prefix(out2_preprocessor, process_instances[index],
                                                                                labels[index], out2_model)
                    if predicted_out2_class != labels[index][-1]:
                        is_correct = False
                    if is_correct:
                        select_index = index
                        select = True
                        break

        if select:
            create_output(args, process_instances[select_index], labels[select_index],
                          out_preprocessor, out_model,
                          act_preprocessor, act_model,
                          out2_preprocessor, out2_model)
        else:
            print("Case 1: process instance not found")


        # Case 2 ########################################################################################
        print("Case 2: process is not conform (1=True; 2=False)")
        select_index = 0
        select = False

        for index in range(0, len(process_instances)):
            if len(process_instances[index]) >= 3:
                if '1' in labels[index] and '2' not in labels[index]:
                    is_correct = True
                    # predict not 1
                    for prefix_index in range(2, len(process_instances[index])):
                        predicted_out_class, _, _, _, _ = out_test.predict_prefix(args, out_preprocessor,
                                                                                  process_instances[index], labels[index],
                                                                                  prefix_index, out_model)
                        if predicted_out_class != labels[index][prefix_index]:
                            is_correct = False
                    # predict not 2
                    predicted_out2_class, _, _, _, _ = out2_test.predict_prefix(out2_preprocessor, process_instances[index],
                                                                                labels[index], out2_model)
                    if predicted_out2_class != labels[index][-1]:
                        is_correct = False
                    if is_correct:
                        select_index = index
                        break

        if select:
            create_output(args, process_instances[select_index], labels[select_index],
                          out_preprocessor, out_model,
                          act_preprocessor, act_model,
                          out2_preprocessor, out2_model)
        else:
            print("Case 2: process instance not found")


        # Case 3 ########################################################################################
        print("Case 3: process is not conform (1=False; 2=True)")
        select_index = 0
        select = False

        for index in range(0, len(process_instances)):
            if len(process_instances[index]) >= 3:
                if '1' not in labels[index] and '2' in labels[index]:
                    is_correct = True
                    # predict not 1
                    for prefix_index in range(2, len(process_instances[index])):
                        predicted_out_class, _, _, _, _ = out_test.predict_prefix(args, out_preprocessor,
                                                                                  process_instances[index],
                                                                                  labels[index],
                                                                                  prefix_index, out_model)
                        if predicted_out_class != labels[index][prefix_index]:
                            is_correct = False
                    # predict not 2
                    predicted_out2_class, _, _, _, _ = out2_test.predict_prefix(out2_preprocessor,
                                                                                process_instances[index],
                                                                                labels[index], out2_model)
                    if predicted_out2_class != labels[index][-1]:
                        is_correct = False
                    if is_correct:
                        select_index = index
                        break

        if select:
            create_output(args, process_instances[select_index], labels[select_index],
                          out_preprocessor, out_model,
                          act_preprocessor, act_model,
                          out2_preprocessor, out2_model)
        else:
            print("Case 3: process instance not found")

        # Case 4 ########################################################################################
        print("Case 4: process is not conform (1=True; 2=True)")
        select_index = 0
        select = False

        for index in range(0, len(process_instances)):
            if len(process_instances[index]) >= 3:
                if '1' in labels[index] and '2' in labels[index]:
                    is_correct = True
                    # predict not 1
                    for prefix_index in range(2, len(process_instances[index])):
                        predicted_out_class, _, _, _, _ = out_test.predict_prefix(args, out_preprocessor,
                                                                                  process_instances[index],
                                                                                  labels[index],
                                                                                  prefix_index, out_model)
                        if predicted_out_class != labels[index][prefix_index]:
                            is_correct = False
                    # predict not 2
                    predicted_out2_class, _, _, _, _ = out2_test.predict_prefix(out2_preprocessor,
                                                                                process_instances[index],
                                                                                labels[index], out2_model)
                    if predicted_out2_class != labels[index][-1]:
                        is_correct = False
                    if is_correct:
                        select_index = index
                        break

        if select:
            create_output(args, process_instances[select_index], labels[select_index],
                          out_preprocessor, out_model,
                          act_preprocessor, act_model,
                          out2_preprocessor, out2_model)
        else:
            print("Case 4: process instance not found")




    # eval outcome prediction
    elif not args.explain and args.task == "outcome":

        from process_prediction.outcome.preprocessor import Preprocessor
        import process_prediction.outcome.predictor as out_test
        import process_prediction.outcome.trainer as train

        preprocessor = Preprocessor(args)

        if args.cross_validation:

            for iteration_cross_validation in range(0, args.num_folds):
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                out_test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1)

        # split validation
        else:
            output["training_time_seconds"].append(train.train(args, preprocessor))
            out_test.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1)


    # eval outcome2 prediction
    elif not args.explain and args.task == "outcome2":

        from process_prediction.outcome2.preprocessor import Preprocessor
        import process_prediction.outcome2.predictor as out_test
        import process_prediction.outcome2.trainer as train

        preprocessor = Preprocessor(args)

        if args.cross_validation:

            for iteration_cross_validation in range(0, args.num_folds):
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                out_test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1)

        # split validation
        else:
            output["training_time_seconds"].append(train.train(args, preprocessor))
            out_test.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1)

    # eval next activity prediction
    elif not args.explain and args.task == "nextevent":

        from process_prediction.nextevent.preprocessor import Preprocessor

        preprocessor = Preprocessor(args)
        import process_prediction.nextevent.predictor as out_test
        import process_prediction.nextevent.trainer as train

        if args.cross_validation:
            for iteration_cross_validation in range(0, args.num_folds):
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                out_test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1)

        # split validation
        else:
            output["training_time_seconds"].append(train.train(args, preprocessor))
            out_test.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1)

    else:
        print("No mode selected ...")
