
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
    parser = argparse.ArgumentParser(description="CS546 Project: Trasnformer Generalization to Arbitary Context Lengths")
    parser.add_argument(
            "--net",
            type=str,
            choices=['tf', 'cnn', 'gru', 'lstm', 'xlnet', 'ibert', 'ibertpos', 'ibert2', 'nam', 'linear', 'dnc', 'lsam', 'namtm', 'ut'],
            default='lsam',
            help='network choices')
    parser.add_argument(
            "--epochs",
            type=int,
            default='100',
            help='number of epochs')
    parser.add_argument(
            "--train_size",
            type=int,
            default='25600',
            help='number of training examples per epoch')
    parser.add_argument(
            "--validation_size",
            type=int,
            default='1024',
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
            choices= ['fib', 'arith', 'palin', 'copy', 'ptbc', 'ptbw', 'scan', 'reduce'],
            default='fib',
            help='fib: fibonacci / arith: arithmetic / palin: palindrome / ptbc: ptb char / ptbw: ptb word')
    parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help='Default learning rate')
    parser.add_argument(
            "--log",
            type=str,
            choices= ['true', 'false'],
            default='false',
            help='Save result to file')
    parser.add_argument(
            "--exp",
            type=int,
            default=0,
            help='Experiment number')
    return parser.parse_args()

