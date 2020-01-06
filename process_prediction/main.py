import canepwdl.config as config
import canepwdl.predictor as test
import canepwdl.trainer as train
from canepwdl.preprocessor import Preprocessor
import canepwdl.utils as utils

from IPython.display import display, HTML
from canepwdl.tools.lrp.LSTM.LSTM_bidi import *
from canepwdl.tools.lrp.util.heatmap import html_heatmap
import codecs

if __name__ == '__main__':

    args = config.load()
    preprocessor = Preprocessor(args)
    output = utils.load_output()
    utils.clear_measurement_file(args)

    # explanation mode
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
        display(HTML(html_heatmap(words, R_words)))

        # How to sanity check global relevance conservation:
        bias_factor = 1.0  # value to use for sanity check
        Rx, Rx_rev, R_rest = net.lrp(words, target_class, eps, bias_factor)  # prefix -> w_indices
        R_tot = Rx.sum() + Rx_rev.sum() + R_rest.sum()  # sum of all "input" relevances

        print(R_tot)
        print("Sanity check passed? ", np.allclose(R_tot, net.s[target_class]))

        print(0)



    # evaluation mode
    else:

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
