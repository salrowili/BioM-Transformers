# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Evaluation metrics for classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import scipy
import sklearn

from finetune import scorer


class SentenceLevelScorer(scorer.Scorer):
  """Abstract scorer for classification/regression tasks."""

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(SentenceLevelScorer, self).__init__()
    self._total_loss = 0
    self._true_labels = []
    self._preds = []

  def update(self, results):
    super(SentenceLevelScorer, self).update(results)
    self._total_loss += results['loss']
    self._true_labels.append(results['label_ids'] if 'label_ids' in results
                             else results['targets'])
    self._preds.append(results['predictions'])

  def get_loss(self):
    return self._total_loss / len(self._true_labels)


class AccuracyScorer(SentenceLevelScorer):

  def _get_results(self):
    correct, count = 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      count += 1
      correct += (1 if y_true == pred else 0)
    return [
        ('accuracy', 100.0 * correct / count),
        ('loss', self.get_loss()),
    ]


class F1Scorer(SentenceLevelScorer):
  """Computes F1 for classification tasks."""

  def __init__(self):
    super(F1Scorer, self).__init__()
    self._positive_label = 1

  def _get_results(self):
    n_correct, n_predicted, n_gold = 0, 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      if pred == self._positive_label:
        n_gold += 1
        if pred == self._positive_label:
          n_predicted += 1
          if pred == y_true:
            n_correct += 1
    if n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = 100.0 * n_correct / n_predicted
      r = 100.0 * n_correct / n_gold
      f1 = 2 * p * r / (p + r)
    return [
        ('precision', p),
        ('recall', r),
        ('f1', f1),
        ('loss', self.get_loss()),
    ]
### Here we add chemprot micro f1 scorer ###
class F1_Chemprot(SentenceLevelScorer):
    def __init__(self):
        super(F1_Chemprot, self).__init__()
        self.label_2_int_mapper = {"CPR:4":0,"CPR:5":1,"CPR:9":2,"CPR:6":3,"CPR:3":4,"false":5}
        self.pos_labels= [0, 1, 2, 3, 4] ## we skip "false [5]" label since we only evaluate chemprot using most common five classes
        
        
    def _get_results(self):
        precision,recall,f1_score,_ = sklearn.metrics.precision_recall_fscore_support(y_pred=self._preds,
                                                                  y_true=self._true_labels,
                                                                  labels=self.pos_labels,
                                                                  average="micro")
        return [('precision', precision*100),('recall', recall*100),('f1', f1_score*100),]


class MCCScorer(SentenceLevelScorer):

  def _get_results(self):
    return [
        ('mcc', 100 * sklearn.metrics.matthews_corrcoef(
            self._true_labels, self._preds)),
        ('loss', self.get_loss()),
    ]


class RegressionScorer(SentenceLevelScorer):

  def _get_results(self):
    preds = np.array(self._preds).flatten()
    return [
        ('pearson', 100.0 * scipy.stats.pearsonr(
            self._true_labels, preds)[0]),
        ('spearman', 100.0 * scipy.stats.spearmanr(
            self._true_labels, preds)[0]),
        ('mse', np.mean(np.square(np.array(self._true_labels) - self._preds))),
        ('loss', self.get_loss()),
    ]