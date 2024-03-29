
""" argparse configuration""" 

import torch
import os
import argparse

def count_params(model):
    cnt = 0
    for p in model.parameters():
        cnt += p.numel()
    return cnt

def get_args():
    """Get all the args"""
    parser = argparse.ArgumentParser(description="Neural Attention Memory")
    parser.add_argument(
            "--net",
            type=str,
            choices=['tf', 'cnn', 'lstm', 'xlnet', 'nojump', 'dnc', 'stm', 'stack', 'noatt',
                      'namtm', 'ut', 'norwprob', 'onlyjump', 'noerase'],
            default='namtm',
            help='network choices')
    parser.add_argument(
            "--epochs",
            type=int,
            default='50',
            help='number of epochs')
    parser.add_argument(
            "--train_size",
            type=int,
            default='51200',
            help='number of training examples per epoch')
    parser.add_argument(
            "--validation_size",
            type=int,
            default='2560',
            help='number of validation examples')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='128',
            help='batch size')
    parser.add_argument(
            "--model_size",
            type=str,
            default='medium',
            choices=['tiny','mini','small','medium','base','custom'],
            help='Size of the model based on Google\'s bert configurations')
    parser.add_argument(
            "--digits",
            type=int,
            default='10',
            help='Max number of digits')
    parser.add_argument(
            "--seq_type",
            type=str,
            choices= ['add', 'arith', 'reverse', 'copy', 'reduce', 'dyck'],
            default='add',
            help='add: addition / arith: arithmetic / reverse: palindrome / reduce: reduction / dyck: 4-DYCK')
    parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help='Default learning rate')
    parser.add_argument(
            "--log",
            action='store_true',
            help='Save result to file')
    parser.add_argument(
            "--debug",
            action='store_true',
            help='Debugger to detect Nan')
    parser.add_argument(
            "--logmsg",
            type=str,
            default='',
            help='Msg about this experiment')
    parser.add_argument(
            "--exp",
            type=int,
            default=0,
            help='Experiment number')
    return parser.parse_args()

