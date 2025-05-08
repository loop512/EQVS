"""
DATE: 19/11/2021
AUTHOR: CHENG ZHANG

Entry point of the full program
"""

import platform
import argparse
import torch
import numpy as np
import random
import os

from train import train_functions


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)

def get_args():
    """
    Get the necessary parameters for the model,
    default settings are args for the experiment: 'CNV, Language: Chinese, Embed_rate: 1.0,
                                                    Size_of_the_inspecting_box: 2*5, Epochs: 100, Batch_size: 4'
    :return: The user input args
    """
    parser = argparse.ArgumentParser(description='Please add the necessary parameters')
    parser.add_argument('--language', type=str, default='Chinese', help='\'Chinese\' or \'English\'')
    parser.add_argument('--embed-rate', type=float, default=1.0, help='Choose from (0.1, 0.2, ..., 1.0)')
    parser.add_argument('--epochs', type=int, default=50, help='The number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='The batch size for training and testing')
    parser.add_argument('--sample-length', type=float, default=10.0, help='The length(seconds) of the training sample')
    parser.add_argument('--version', type=str, default='None', help='Version')
    parser.add_argument('--channel', type=int, default=4, help='Number of channel used in the model')
    parser.add_argument('--embed-dim', type=int, default=50, help='Number of embedding dimensions')
    parser.add_argument('--mode', type=str, default='train', help='\'train\', \'speed\', or \'eval\'')
    parser.add_argument('--pretrain', type=bool, default=True, help='\'True\', or \'False\'')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    fix_random_seed(2024)
    args = get_args()
    if args.mode == "train":
        train_functions.full_train(args.language, args.embed_rate, args.sample_length, args.epochs, args.batch_size,
                                   args.pretrain)
    elif args.mode == "speed":
        train_functions.speed_test(args.language, args.embed_rate, args.sample_length, args.batch_size)
    elif args.mode == "eval":
        train_functions.get_inference_value(args.language, args.embed_rate, args.sample_length, 1)
