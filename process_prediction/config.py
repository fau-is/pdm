import os
import argparse
import process_prediction.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--explain', default=True, type=utils.str2bool)  # True + outcome2
    parser.add_argument('--task', default="outcome2")  # outcome; outcome2; nextevent

    # dnn
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_architecture', default=0, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)  # dnc 0.0001 #lstm 0.002

    # representation
    parser.add_argument('--embedding_dim', default=100, type=int)
    parser.add_argument('--embedding_epochs', default=10, type=int)

    # evaluation
    parser.add_argument('--num_folds', default=3, type=int)  # 10
    parser.add_argument('--cross_validation', default=True, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.5, type=float)  # only if cross validation is deactivated
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # data
    parser.add_argument('--data_set_out2', default="train_bpia_pcm_sample_2.csv")
    parser.add_argument('--data_set_out1', default="train_bpia_pcm_sample_1.csv")
    parser.add_argument('--data_set_act', default="train_bpia_pcm_sample.csv")
    parser.add_argument('--data_set', default="train_bpia_pcm_sample.csv")

    # explain -> train_hb_pcm
    # outcome -> train_hb_pc_1
    # outcome -> train_hb_pc_2
    # nextevent -> all

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
