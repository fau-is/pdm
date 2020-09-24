import os
import argparse
import process_prediction.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--task', default="outcome")

    # dnn
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_architecture', default=0, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)

    # representation
    parser.add_argument('--embedding_dim', default=8, type=int)
    parser.add_argument('--embedding_epochs', default=10, type=int)

    # evaluation
    parser.add_argument('--num_folds', default=0, type=int)
    parser.add_argument('--cross_validation', default=False, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.8, type=float)
    parser.add_argument('--val_split', default=0.1, type=float)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # data
    parser.add_argument('--data_set', default="train_new_hb_pcm_shift_sample.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--model_dir', default="./models/")
    parser.add_argument('--result_dir', default="./results/")

    # gpu processing
    parser.add_argument('--gpu_ratio', default=0.2, type=float)
    parser.add_argument('--cpu_num', default=1, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
