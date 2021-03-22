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

    if args.seed:
        utils.set_seeds(args)

    train_indices, test_indices = preprocessor.get_indices_split_validation(args, event_log)

    training_time_seconds = trainer.train(args, event_log, preprocessor, train_indices)
    predicted_distributions, ground_truths = predictor.test(args, event_log, preprocessor, test_indices)

    output["training_time_seconds"].append(training_time_seconds)
    output = utils.get_output(args, output, predicted_distributions, ground_truths)
    utils.print_output(output, -1)
    utils.write_output(args, output, -1)
