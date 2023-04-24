# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/listops.py
# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generators for custom listops tasks."""

import csv
import random

from absl import app
from absl import flags
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#import tensorflow.compat.v1 as tf

flags.DEFINE_string(
    'task', default='dyck',
    help='Name of task to create.')
flags.DEFINE_integer(
    'num_train_samples', default=12800,
    help=('Number of train samples.'))
flags.DEFINE_integer(
    'num_valid_samples', default=2048,
    help=('Number of test samples.'))
flags.DEFINE_integer(
    'num_test_samples', default=2048,
    help=('Number of test samples.'))
flags.DEFINE_integer(
    'max_depth', default=12,
    help=('maximum tree depth of training sequences.'))
flags.DEFINE_integer(
    'train_depth', default=8,
    help=('maximum tree depth of training sequences.'))
flags.DEFINE_integer(
    'max_length', default=64,
    help=('maximum length per sequence in training sequences.'))
flags.DEFINE_integer(
    'val_length', default=48,
    help=('maximum length per sequence in training sequences.'))
flags.DEFINE_integer(
    'train_length', default=32,
    help=('maximum length per sequence in training sequences.'))
flags.DEFINE_integer(
    'min_length', default=8,
    help=('minimum length per sequence in training sequences.'))
flags.DEFINE_string(
    'output_dir', default='dyckdata',
    help='Directory to output files.')

FLAGS = flags.FLAGS

OPERATORS = ['[', '(', '{', '<']  # , FIRST, LAST]
CLOSE = {'[':']','(':')','{':'}','<':'>'}
VALUE_P = 0.3


def dyck(target_depth, min_length):
    stack = []
    output = []
    while True:
      if len(stack) >= target_depth and len(output) > min_length:
        while len(stack) > target_depth:
          output.append(stack.pop())
        return output, stack[::-1]
      else:
        p = random.random()
        if p > VALUE_P or stack == []:
          op = random.choice(OPERATORS)
          output.append(op)
          stack.append(CLOSE[op])
        else:
          output.append(stack.pop())
          
def write_to_file(data, fp):
  with open(fp + '.txt', 'w+') as f:
    for input,target in data:
        istr = ','.join(input)
        ostr = ','.join(target)
        f.write(istr + '\t' + ostr +'\n')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  #tf.logging.info('Start dataset construction')
  if not os.path.exists(FLAGS.output_dir):
   os.makedirs(FLAGS.output_dir)
  data = []
  iiddata = []
  vdata = []
  tdata = []
#  num_samples = FLAGS.num_train_samples \
#      + FLAGS.num_test_samples + FLAGS.num_valid_samples
  num_samples = FLAGS.num_train_samples
  print('Start creating training samples')
  maxlen = 0
  while len(data) < num_samples:
    curdepth = 2 + (len(data)*(FLAGS.train_depth-1)) // num_samples
    output, target = dyck(curdepth, FLAGS.min_length)
    if len(output) < FLAGS.train_length:
      data.append((output,target))
      maxlen = max(len(output) + len(target), maxlen)
      if len(data) % 1000 == 0:
        #tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(data)))
  print('Max len: {}'.format(maxlen))
  write_to_file(data, FLAGS.output_dir + '/{}_train'.format(FLAGS.task))

  print('Start creating IID valid samples')
  maxlen = 0
  data = []
  while len(data) < FLAGS.num_valid_samples:
    curdepth = FLAGS.train_depth
    output, target = dyck(curdepth, FLAGS.min_length)
    if len(output) < FLAGS.train_length:
      data.append((output,target))
      maxlen = max(len(output) + len(target), maxlen)
      if len(data) % 1000 == 0:
        #tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(data)))
  print('Max len: {}'.format(maxlen))
  write_to_file(data, FLAGS.output_dir + '/{}_val'.format(FLAGS.task))

  print('Start creating length ood samples')
  maxlen = 0
  data = []
  while len(data) < FLAGS.num_valid_samples:
    curdepth = FLAGS.train_depth
    output, target = dyck(curdepth, FLAGS.train_length)
    if len(output) < FLAGS.max_length:
      data.append((output,target))
      maxlen = max(len(output) + len(target), maxlen)
      if len(data) % 1000 == 0:
        #tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(data)))
  print('Max len: {}'.format(maxlen))
  write_to_file(data, FLAGS.output_dir + '/{}_length'.format(FLAGS.task))

  #tf.logging.info('Finished running dataset construction')

  print('Start creating depth ood samples')
  maxlen = 0
  data = []
  while len(data) < FLAGS.num_valid_samples:
    curdepth = FLAGS.train_depth + 1 + ((FLAGS.max_depth - FLAGS.train_depth)*len(data))//FLAGS.num_valid_samples
    output, target = dyck(curdepth, FLAGS.train_length)
    if len(output) < FLAGS.max_length:
      data.append((output,target))
      maxlen = max(len(output) + len(target), maxlen)
      if len(data) % 1000 == 0:
        #tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(data)))
  print('Max len: {}'.format(maxlen))
  write_to_file(data, FLAGS.output_dir + '/{}_test'.format(FLAGS.task))

class DYCKDataset(Dataset):
    def __init__(self, tsv_file, max_len=-1):
        
        vocabs = ['[',']','(',')','{','}','<','>','<SEP>','<MASK>','<PAD>']
        self.dict = {}
        
        for i,v in enumerate(vocabs):
            self.dict[v] = i
        self.wordtoix = self.dict
        self.vocab = vocabs
        self.vocab_size = len(vocabs) 
        self.inputs = []
        self.targets = []
        with open(tsv_file, "r") as fd:
            for l in fd:
                inp, tgt = l.split('\t')
                itokens = inp.strip().split(',')
                otokens = tgt.strip().split(',')
                if len(itokens) + len(otokens) > max_len:
                    max_len = len(itokens) + len(otokens)
                iseq = [self.dict[tok] for tok in itokens]
                tseq = [self.dict[tok] for tok in otokens]
                self.inputs.append(iseq)
                self.targets.append(tseq)
        self.max_len = max_len+1
        self.inp_arr = np.ones([len(self.targets), self.max_len], dtype=np.int64)*(self.dict['<PAD>']) #pre-fill with padding
        self.out_arr = np.ones([len(self.targets), self.max_len], dtype=np.int64)*(self.dict['<PAD>']) #pre-fill with padding

        for idx, inp in enumerate(self.inputs):
            for i,t in enumerate(inp):
                self.inp_arr[idx, i] = t
                self.out_arr[idx, i] = t
            self.inp_arr[idx,len(inp)] = self.dict['<SEP>']
            self.out_arr[idx,len(inp)] = self.dict['<SEP>']
            for j,o in enumerate(self.targets[idx]):
                self.inp_arr[idx,j+len(inp)+1] = self.dict['<MASK>']
                self.out_arr[idx,j+len(inp)+1] = o
        self.size = len(self.targets)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inp_arr[idx], self.out_arr[idx]

if __name__ == '__main__':
  app.run(main)