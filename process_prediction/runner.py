import process_prediction.config as config
# todo: differ between the prediction tasks
import process_prediction.outcome.predictor as test
import process_prediction.outcome.trainer as train
from process_prediction.outcome.preprocessor import Preprocessor
import process_prediction.utils as utils
from process_prediction.explanation.LSTM.LSTM_bidi import *
from process_prediction.explanation.util.heatmap import html_heatmap
import process_prediction.explanation.util.browser as browser

if __name__ == '__main__':

    args = config.load()
    preprocessor = Preprocessor(args)
    output = utils.load_output()
    utils.clear_measurement_file(args)

    # explanation mode for outcome prediction
    if args.explain:

        predicted_class, target_class, words, model, input_embedded = test.predict(args, preprocessor)
        print("Prediction: %s" % predicted_class)
        target_class = predicted_class

        if isinstance(target_class, str):
            target_class = int(target_class)

        # compute lrp relevances
        # LRP hyperparameters:
        eps = 0.001  # small positive number
        bias_factor = 0.0  # recommended value

        net = LSTM_bidi(args, model, input_embedded)  # load trained LSTM model

        # w_indices = [net.voc.index(p) for p in prefix]  # convert input sentence to word IDs
        Rx, Rx_rev, R_rest = net.lrp(words, target_class, eps, bias_factor)  # perform LRP
        R_words = np.sum(Rx + Rx_rev, axis=1)  # compute word-level LRP relevances

        scores = net.s.copy()  # classification prediction scores

        print("prediction scores:", scores)
        print("\nLRP target class:", target_class)
        print("\nLRP relevances:")
        for idx, w in enumerate(words):
            print("\t\t\t" + "{:8.10f}".format(R_words[idx]) + "\t" + w)
        print("\nLRP heatmap:")

        browser.display_html(html_heatmap(words, R_words))

        # How to sanity check global relevance conservation:
        bias_factor = 1.0  # value to use for sanity check
        Rx, Rx_rev, R_rest = net.lrp(words, target_class, eps, bias_factor)  # prefix -> w_indices
        R_tot = Rx.sum() + Rx_rev.sum() + R_rest.sum()  # sum of all "input" relevances

        print(R_tot)
        print("Sanity check passed? ", np.allclose(R_tot, net.s[target_class]))

    # eval outcome prediction
    elif not args.explain and args.task == "outcome":

        if args.cross_validation:

            for iteration_cross_validation in range(0, args.num_folds):
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1)

        # split validation
        else:
            output["training_time_seconds"].append(train.train(args, preprocessor))
            test.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1)

    # eval next activity prediction
    else:
        print("Is coming ...")
