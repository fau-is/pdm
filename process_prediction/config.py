import os
import argparse
import process_prediction.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--task', default="outcome")

    # dnn
    parser.add_argument('--dnn_num_epochs', default=100, type=int)

    # representation
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)  # for numerical attributes
    parser.add_argument('--encoding_cat', default="onehot", type=str)  # for categorical attributes
    parser.add_argument('--include_time_delta', default=False, type=utils.str2bool)

    # evaluation
    parser.add_argument('--shuffle', default=False, type=utils.str2bool)
    parser.add_argument('--seed', default=False, type=utils.str2bool)
    parser.add_argument('--seed_val', default=1377, type=int)
    parser.add_argument('--num_folds', default=0, type=int)
    parser.add_argument('--split_rate_train', default=0.8, type=float)
    parser.add_argument('--split_rate_train_hpo', default=0.9, type=float)
    parser.add_argument('--val_split', default=0.1, type=float)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)


    # hyper-parameter optimization
    parser.add_argument('--hpo', default=False, type=utils.str2bool)
    parser.add_argument('--hpo_eval_runs', default=10, type=int)
    parser.add_argument('--hpo_units', default=[100], type=list)
    parser.add_argument('--hpo_activation', default=['linear', 'tanh', 'relu'], type=list)
    parser.add_argument('--hpo_optimizer', default=['nadam', 'adam', 'rmsprop'], type=list)
    parser.add_argument('--hpo_kernel_initializer', default=['glorot_normal', 'glorot_uniform'], type=list)

    # data
    parser.add_argument('--data_set', default="bpia17_pdm_shift_inserted.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--model_dir', default="./models/")
    parser.add_argument('--result_dir', default="./results/")

    # event log / data format
    parser.add_argument('--case_id_key', default="case", type=str)
    parser.add_argument('--activity_key', default="event", type=str)
    parser.add_argument('--outcome_key', default="conformance", type=str)
    parser.add_argument('--time_delta_key', default="time_delta", type=str)
    parser.add_argument('--date_format', default="%d.%m.%y-%H:%M:%S", type=str)

    # gpu processing
    parser.add_argument('--gpu_ratio', default=0.2, type=float)
    parser.add_argument('--cpu_num', default=1, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
