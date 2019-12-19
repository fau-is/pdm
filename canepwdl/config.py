import os
import argparse
import canepwdl.utils as utils

"""
Important details!
- if the batch size is set lower than the total number of instances you get the error "AttributeError: 'ProgbarLogger' object has no attribute 'log_values'"
"""


def load():
    parser = argparse.ArgumentParser()

    # dnn
    parser.add_argument('--dnn_num_epochs', default=1, type=int)  # 100
    parser.add_argument('--dnn_architecture', default=3, type=int)

    # pre-processing
    """
    Option: min_max_norm, int, bin, onehot, hash or doc2vec)
    Note doc2vec assumes int encoding for cat attributes and a min_max_norm for num attributes
    Only these values works. In the case of other values, no encoding is performed.
    """
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)  # for numerical attributes
    parser.add_argument('--encoding_cat', default="onehot", type=str)  # for categorical attributes
    parser.add_argument('--num_hash_output', default=10, type=int)  # number of output columns of hash encoding; see work from Mehdiyev et al. (2017)
    parser.add_argument('--doc2vec_num_epochs', default=1, type=int)
    parser.add_argument('--doc2vec_vec_size', default=32, type=int)
    parser.add_argument('--doc2vec_alpha', default=0.025, type=int)
    parser.add_argument('--include_time_delta', default=True, type=utils.str2bool)

    # all models
    parser.add_argument('--task', default="next_event")
    # parser.add_argument('--learning_rate', default=0.002, type=float)

    # evaluation
    parser.add_argument('--num_folds', default=10, type=int)  # 10
    parser.add_argument('--cross_validation', default=True, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.3, type=float)  # only if cross validation is deactivated
    parser.add_argument('--batch_size_train', default=1, type=int)  # 256
    parser.add_argument('--batch_size_test', default=1, type=int)

    # data
    parser.add_argument('--data_set', default="bpi2012_w_converted.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")
    parser.add_argument('--result_dir', default="./results/")
    parser.add_argument('--experiments_config_file', default="../experiments_config.csv")

    # gpu processing
    parser.add_argument('--gpu_ratio', default=1.0, type=float)
    parser.add_argument('--cpu_num', default=6, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
