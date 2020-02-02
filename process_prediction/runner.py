import process_prediction.config as config
import process_prediction.utils as utils
from process_prediction.explanation.LSTM.LSTM_bidi import *
from process_prediction.explanation.util.heatmap import html_heatmap
import process_prediction.explanation.util.browser as browser


if __name__ == '__main__':
    args = config.load()
    output = utils.load_output()
    utils.clear_measurement_file(args)

    # explanation mode for outcome prediction
    if args.explain:
        from process_prediction.outcome.preprocessor import Preprocessor as Out_Preprocessor
        from process_prediction.nextevent.preprocessor import Preprocessor as Act_Preprocessor
        import process_prediction.outcome.predictor as out_test
        import process_prediction.nextevent.predictor as act_test

        out_preprocessor = Out_Preprocessor(args)
        act_preprocessor = Act_Preprocessor(args)

        # todo: select a process instance with each rule!
        # 1. case: only 0s; 2. case at least a label with value 0
        # assumption: value 0 is conformant
        val_out_classes = [int(val) for val in out_preprocessor.data_structure['meta']['val_classes']]
        val_out_classes.sort()

        for out_class_index, out_class_val in enumerate(val_out_classes):

            # print(out_class_index)
            # print(out_class_val)

            # select a sequence
            process_instance, out_labels = out_preprocessor.get_process_instance(out_class_val)
            prefix_heatmaps = ""

            print("Conformance class: %s" % out_class_val)

            for prefix_index in range(2, len(process_instance)):


                # next activity prediction
                predicted_act_class, target_act_class, _, _, _ = act_test.predict_prefix(args, act_preprocessor,
                                                                                         process_instance, prefix_index)

                # outcome prediction
                predicted_out_class, target_out_class, prefix_words, out_model, out_input_embedded = out_test.predict_prefix(args, out_preprocessor, process_instance, out_labels, prefix_index)


                print("Prefix: %s; Outcome prediction: %s; Outcome target: %s" % (prefix_index, predicted_out_class, target_out_class))
                print("Prefix: %s; Next act prediction: %s; Next act target: %s" % (prefix_index, predicted_act_class, target_act_class))

                target_out_class = predicted_out_class

                if isinstance(target_out_class, str):
                    target_out_class = int(target_out_class)

                # compute lrp relevances
                # LRP hyperparameters:
                eps = 0.001  # small positive number
                bias_factor = 0.0  # recommended value
                net = LSTM_bidi(args, out_model, out_input_embedded)  # load trained LSTM model
                Rx, Rx_rev, R_rest = net.lrp(prefix_words, target_out_class, eps, bias_factor)  # perform LRP
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
                Rx, Rx_rev, R_rest = net.lrp(prefix_words, target_out_class, eps, bias_factor)  # prefix -> w_indices
                R_tot = Rx.sum() + Rx_rev.sum() + R_rest.sum()  # sum of all "input" relevances

                """
                print(R_tot)
                print("Sanity check passed? ", np.allclose(R_tot, net.s[target_out_class]))
                """
            browser.display_html(prefix_heatmaps)



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