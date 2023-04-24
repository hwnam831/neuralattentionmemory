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
#import tensorflow.compat.v1 as tf

flags.DEFINE_string(
    'task', default='basic',
    help='Name of task to create.')
flags.DEFINE_integer(
    'num_train_samples', default=25600,
    help=('Number of train samples.'))
flags.DEFINE_integer(
    'num_valid_samples', default=2048,
    help=('Number of test samples.'))
flags.DEFINE_integer(
    'num_test_samples', default=2048,
    help=('Number of test samples.'))
flags.DEFINE_integer(
    'max_depth', default=6,
    help=('maximum tree depth of training sequences.'))
flags.DEFINE_integer(
    'train_depth', default=4,
    help=('maximum tree depth of training sequences.'))
flags.DEFINE_integer(
    'max_args', default=6,
    help=('maximum number of arguments per operator in training sequences.'))
flags.DEFINE_integer(
    'valid_args', default=5,
    help=('maximum number of arguments per operator in training sequences.'))
flags.DEFINE_integer(
    'train_args', default=4,
    help=('maximum number of arguments per operator in training sequences.'))
flags.DEFINE_integer(
    'max_length', default=96,
    help=('maximum length per sequence in training sequences.'))
flags.DEFINE_integer(
    'max_length', default=64,
    help=('maximum length per sequence in training sequences.'))
flags.DEFINE_integer(
    'train_length', default=48,
    help=('maximum length per sequence in training sequences.'))
flags.DEFINE_integer(
    'min_length', default=4,
    help=('minimum length per sequence in training sequences.'))
flags.DEFINE_string(
    'output_dir', default='listopsdata',
    help='Directory to output files.')

FLAGS = flags.FLAGS
BASE = 10
MIN = '[MIN'
MAX = '[MAX'
MED = '[MED'
FIRST = '[FIRST'
LAST = '[LAST'
SUM_MOD = '[SM'
END = ']'

OPERATORS = [MIN, MAX, SUM_MOD, LAST]  # , FIRST, LAST]
VALUES = range(BASE)

VALUE_P = 0.25


def generate_tree(depth, max_depth, max_args, min_args=1):
  """Generate tree-like equations.
  Args:
    depth: current depth of the node, int.
    max_depth: maximum depth of the tree, int.
    max_args: maximum number of arguments per operator, int.
  Returns:
    The root node of a tree structure.
  """
  max_d = 1
  if depth < max_depth:
    r = random.random()
  else:
    r = 1

  if r > VALUE_P:
    value = random.choice(VALUES)
    return value, 1, max_d+1
  else:
    length = 2
    num_values = random.randint(min_args+1, max_args)
    values = []
    for _ in range(num_values):
      sub_t, sub_l, sub_d = generate_tree(depth + 1, max_depth, max_args, 
        min_args=min_args)
      values.append(sub_t)
      length += sub_l
      max_d = sub_d if sub_d > max_d else max_d

    op = random.choice(OPERATORS)
    if depth == 1:
      op = SUM_MOD
    t = (op, values[0])
    for value in values[1:]:
      t = (t, value)
    t = (t, END)
  return t, length, max_d+1


def to_string(t, parens=False):
  if isinstance(t, str):
    return t
  elif isinstance(t, int):
    return str(t)
  else:
    if parens:
      return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'
    else:
      return to_string(t[0]) + ' ' + to_string(t[1])


def to_value(t):
  """Compute the output of equation t.
  Args:
    t: a tree structure that represents equation t, list.
  Returns:
    The result of equation t, int.
  """
  if not isinstance(t, tuple):
    return t
  l = to_value(t[0])
  r = to_value(t[1])
  if l in OPERATORS:  # Create an unsaturated function.
    return (l, [r])
  elif r == END:  # l must be an unsaturated function.
    if l[0] == MIN:
      return min(l[1])
    elif l[0] == MAX:
      return max(l[1])
    elif l[0] == FIRST:
      return l[1][0]
    elif l[0] == LAST:
      return l[1][-1]
    elif l[0] == MED:
      return int(np.median(l[1]))
    elif l[0] == SUM_MOD:
      return np.sum(l[1]) % BASE
  elif isinstance(l, tuple):
    # We've hit an unsaturated function and an argument.
    return (l[0], l[1] + [r])
def stack_solver(t):
  valuestack = [0]
  opstack = [END]
  curop = END
  curval = 0
  tokens = to_string(t).split(" ")
  outstr = ''
  def update_val(curop,curval,val):
    if curop == MIN:
        curval = min(val,curval)
    elif curop == MAX:
        curval = max(val,curval)
    elif curop == LAST:
        curval = val
    elif curop == SUM_MOD:
        curval = (val + curval)%BASE
    return curval
  for tok in tokens:
    if tok == MIN:
      opstack.append(curop)
      valuestack.append(curval)
      curop = MIN
      curval = BASE-1
    elif tok == MAX:
      opstack.append(curop)
      valuestack.append(curval)
      curop = MAX
      curval = 0
    elif tok == LAST:
      opstack.append(curop)
      valuestack.append(curval)
      curop = LAST
      curval = 0
    elif tok == SUM_MOD:
      opstack.append(curop)
      valuestack.append(curval)
      curop = SUM_MOD
      curval = 0
    elif tok == END:
      curop = opstack.pop()
      curval = update_val(curop, curval, valuestack.pop())
    else:
      curval = update_val(curop,curval,int(tok))
    outstr += curop+','+str(curval) + ','
    #outstr += curop+','+str(curval) + ' '
    outstr += opstack[-1]+','+str(valuestack[-1]) + ' '
  return outstr
def write_to_file(data, fp):
  """Write to file output."""
  #tf.logging.info(type(data))
  #tf.logging.info('Writing {} samples to {}'.format(len(data), fp + '.tsv'))
  with open(fp + '.tsv', 'w+') as f:
    writer = csv.writer(f, delimiter='\t')
    #writer.writerow(['Source', 'Target'])
    writer.writerows(data)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  #tf.logging.info('Start dataset construction')

  data = set()
  iiddata = set()
  vdata = set()
  tdata = set()
#  num_samples = FLAGS.num_train_samples \
#      + FLAGS.num_test_samples + FLAGS.num_valid_samples
  num_samples = FLAGS.num_train_samples
  print('Start creating training samples')
  maxlen = 0
  while len(data) < num_samples:
    curdepth = 2 + (len(data)*(FLAGS.train_depth-1)) // num_samples
    tree, length, _ = generate_tree(1, curdepth, FLAGS.train_args)
    if length > FLAGS.min_length and length < FLAGS.train_length:
      data.add(tree)
      maxlen = max(length, maxlen)
      if len(data) % 1000 == 0:
        #tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(data)))
  print('Max len: {}'.format(maxlen))
  train = []
  for example in data:
    train.append([to_string(example), to_value(example), stack_solver(example)])
    #train.append([to_string(example), stack_solver(example)])
  write_to_file(train, FLAGS.output_dir + '/{}_train'.format(FLAGS.task))

  print('Start creating IID valid samples')
  maxlen = 0
  while len(iiddata) < FLAGS.num_valid_samples:
    tree, length, _ = generate_tree(1, FLAGS.train_depth, FLAGS.train_args)
    if length > FLAGS.min_length and length < FLAGS.train_length:
      iiddata.add(tree)
      maxlen = max(length, maxlen)
      if len(iiddata) % 1000 == 0:
        #tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(iiddata)))
  print('Max len: {}'.format(maxlen))
  iid = []
  for example in iiddata:
    iid.append([to_string(example), to_value(example), stack_solver(example)])
  write_to_file(iid, FLAGS.output_dir + '/{}_valid'.format(FLAGS.task))

  print('Start creating arg samples')
  maxlen = 0
  while len(vdata) < FLAGS.num_valid_samples:
    tree, length, _ = generate_tree(1, FLAGS.train_depth, FLAGS.max_args, min_args=FLAGS.train_args)
    if length > FLAGS.min_length and length < FLAGS.max_length:
      vdata.add(tree)
      maxlen = max(length, maxlen)
      if len(vdata) % 1000 == 0:
        #tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(vdata)))
  print('Max len: {}'.format(maxlen))
  val = []
  for example in vdata:
    val.append([to_string(example), to_value(example), stack_solver(example)])
  write_to_file(val, FLAGS.output_dir + '/{}_args'.format(FLAGS.task))

  #tf.logging.info('Finished running dataset construction')

  print('Start creating depth samples')
  maxlen = 0
  while len(tdata) < FLAGS.num_test_samples:
    tree, length, depth = generate_tree(1, FLAGS.max_depth, FLAGS.train_args)
    if length > FLAGS.min_length and length < FLAGS.max_length and depth > FLAGS.train_depth:
      tdata.add(tree)
      maxlen = max(length, maxlen)
      if len(tdata) % 1000 == 0:
        #tf.logging.info('Processed {}'.format(len(data)))
        print('Processed {}'.format(len(tdata)))
  print('Max len: {}'.format(maxlen))
  test = []
  for example in tdata:
    test.append([to_string(example), to_value(example), stack_solver(example)])
  write_to_file(test, FLAGS.output_dir + '/{}_depth'.format(FLAGS.task))



if __name__ == '__main__':
  app.run(main)