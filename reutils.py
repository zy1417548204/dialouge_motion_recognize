# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
EmoTect utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import six
import random

import paddle
import paddle.fluid as fluid
import numpy as np
import sklearn.metrics

def check_cuda(use_cuda, err = \
    "\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
    Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"):
    try:
        if use_cuda == True and fluid.is_compiled_with_cuda() == False:
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """
    Init CheckPoint
    """
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        """
        If existed presitabels
        """
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))


def word2id(word_dict, query):
    """
    Convert word sequence into id list
    """
    unk_id = len(word_dict)
    wids = [word_dict[w] if w in word_dict else unk_id
            for w in query.strip().split(" ")]
    return wids


def data_reader(file_path, word_dict, num_examples, phrase, epoch=1):
    """
    Data reader, which convert word sequence into id list
    """
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            if line.startswith("label"):
                continue
            if phrase == "infer":
                cols = line.strip().split("\t")
                query = cols[-1] if len(cols) != -1 else cols[0]
                wids = word2id(word_dict, query)
                all_data.append((wids,))
            else:
                cols = line.strip().split("\t")
                if len(cols) != 2:
                    sys.stderr.write("[NOTICE] Error Format Line!")
                    continue
                label = int(cols[0])
                query = cols[1].strip()
                wids = word2id(word_dict, query)
                all_data.append((wids, label))
    num_examples[phrase] = len(all_data)

    if phrase == "infer":
        def reader():
            """
            Infer reader function
            """
            for wids in all_data:
                yield wids
        return reader

    def reader():
        """
        Reader function
        """
        for idx in range(epoch):
            if phrase == "train":
                random.shuffle(all_data)
            for wids, label in all_data:
                yield wids, label
    return reader


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as fin:
        wid = 0
        for line in fin:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


def print_arguments(args):
    """
    print arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def query2ids(vocab_path, query):
    """
    Convert query to id list according to the given vocab
    """
    vocab = load_vocab(vocab_path)
    wids = word2id(vocab, query)
    return wids


def accuracy(y_true, y_pred):
    """
    Accuracy classification score.
    """
    score = sklearn.metrics.accuracy_score(y_true, y_pred)
    return score


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix to evaluate the precision/recall/f1 of classification.

    Args:
        y_true: 1d array, Ground truth (correct) labels.
        y_pred: 1d array, Predict labels.

    Returns:
        cm: 2d array
    """
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return cm


def classification_report(y_true, y_pred,
        labels=[0, 1, 2], # negative, neutral, positive
        output_dict=True):
    """
    Build a report showing the main classification metrics(pre, rec, f1-score)
    """
    report = sklearn.metrics.classification_report(y_true, y_pred,
        labels=labels, output_dict=output_dict)
    return report

