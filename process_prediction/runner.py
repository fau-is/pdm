import process_prediction.config as config
import process_prediction.utils as utils
from process_prediction.outcome.preprocessor import Preprocessor
import process_prediction.outcome.trainer as trainer
import process_prediction.outcome.predictor as predictor

if __name__ == '__main__':
    args = config.load()
    output = utils.load_output()
    # utils.clear_measurement_file(args)

    preprocessor = Preprocessor()
    event_log = preprocessor.get_event_log(args)

    if args.cross_validation:

        train_indices_per_fold, test_indices_per_fold = preprocessor.get_indices_k_fold_validation(args, event_log)

        for iteration_cross_validation in range(0, args.num_folds):
            preprocessor.iteration_cross_validation = iteration_cross_validation

            training_time_seconds, best_model_id = trainer.train(args, event_log, preprocessor, train_indices_per_fold)
            predictor.test(args, event_log, preprocessor, test_indices_per_fold, best_model_id)

            output["training_time_seconds"].append(training_time_seconds)
            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, iteration_cross_validation)
            utils.write_output(args, output, iteration_cross_validation)

        utils.print_output(args, output, iteration_cross_validation + 1)
        utils.write_output(args, output, iteration_cross_validation + 1)

    else:
        train_indices, test_indices = preprocessor.get_indices_split_validation(args, event_log)

        training_time_seconds, best_model_id = trainer.train(args, event_log, preprocessor, train_indices)
        predictor.test(args, event_log, preprocessor, test_indices, best_model_id)

        output["training_time_seconds"].append(training_time_seconds)
        output = utils.get_output(args, preprocessor, output)
        utils.print_output(args, output, -1)
        utils.write_output(args, output, -1)

