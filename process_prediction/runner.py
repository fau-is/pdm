import process_prediction.config as config
import process_prediction.utils as utils
from process_prediction.outcome.preprocessor import Preprocessor
import process_prediction.outcome.trainer as train
import process_prediction.outcome.predictor as predictor

if __name__ == '__main__':
    args = config.load()
    output = utils.load_output()
    utils.clear_measurement_file(args)

    preprocessor = Preprocessor(args)

    if args.cross_validation:

        for iteration_cross_validation in range(0, args.num_folds):
            preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

            output["training_time_seconds"].append(train.train(args, preprocessor))
            predictor.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, iteration_cross_validation)
            utils.write_output(args, output, iteration_cross_validation)

        utils.print_output(args, output, iteration_cross_validation + 1)
        utils.write_output(args, output, iteration_cross_validation + 1)

    else:
        output["training_time_seconds"].append(train.train(args, preprocessor))
        predictor.test(args, preprocessor)

        output = utils.get_output(args, preprocessor, output)
        utils.print_output(args, output, -1)
        utils.write_output(args, output, -1)

