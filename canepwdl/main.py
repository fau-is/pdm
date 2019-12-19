import canepwdl.config as config
import canepwdl.predictor as test_dnn
import canepwdl.trainer as train_dnn
from canepwdl.preprocessor import Preprocessor
import canepwdl.utils as utils
import copy


def execute_experiment(args, experiment_dict, init_data_structure):

    args = utils.set_single_experiment_config(args, experiment_dict)

    preprocessor = Preprocessor(args)
    preprocessor.data_structure = init_data_structure

    preprocessor.create(args)

    output = utils.load_output()
    utils.clear_measurement_file(args)

    if args.cross_validation:

        for iteration_cross_validation in range(0, args.num_folds):
            utils.start_nested_run_mlflow(args, iteration_cross_validation)

            preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

            output["training_time_seconds"].append(train_dnn.train(args, preprocessor))

            test_dnn.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, iteration_cross_validation)
            utils.write_output(args, output, iteration_cross_validation)
            utils.log_at_end_of_nested_run_mlflow(args, iteration_cross_validation)
            utils.end_run_mlflow()

        utils.start_nested_run_mlflow(args, iteration_cross_validation + 1)
        utils.print_output(args, output, iteration_cross_validation + 1)
        utils.write_output(args, output, iteration_cross_validation + 1)
        # utils.log_at_end_of_nested_run_mlflow(args, iteration_cross_validation + 1)
        utils.end_run_mlflow()




    # split validation
    else:
        utils.start_nested_run_mlflow(args, -1)
        output["training_time_seconds"].append(train_dnn.train(args, preprocessor))
        test_dnn.test(args, preprocessor)

        output = utils.get_output(args, preprocessor, output)
        utils.print_output(args, output, -1)
        utils.write_output(args, output, -1)
        utils.end_run_mlflow()

    utils.end_run_mlflow()



if __name__ == '__main__':

    args = config.load()
    experiments_dicts = utils.get_experiments_configuration(args)
    preprocessor = Preprocessor(args)
    init_data_structure = copy.deepcopy(preprocessor.data_structure)
    utils.check_directory_path('./results/')
    utils.check_directory_path('./checkpoints/')

    for experiment_dict in experiments_dicts.items():
        execute_experiment(args, experiment_dict, copy.deepcopy(init_data_structure))

    print(0)


