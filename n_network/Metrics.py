'''
__author__ = "Ricardo Montañana Gómez"
__copyright__ = "Copyright 2020, Ricardo Montañana Gómez"
__license__ = "MIT"
Compute metrics for predicted data
'''

import numpy as np
from .Utils import one_hot


class Metrics:
    """
    True Positives (tp), These are the correctly predicted positive values
    True Negatives (tn), These are the correctly predicted negative values
    False Positives (fp), When actual class is target and predicted class is the other
    False Negatives (fn), When actual class is reverse of target but predicted class is target
    """
    _truth = None
    _predicted = None
    _tp = None
    _fp = None
    _fn = None
    _num_classes = 0

    def __init__(self, y=None, yhat=None):
        self._truth = self._adapt(y, update_num=True)
        self._predicted = self._adapt(yhat)
        self._compute_parameters()

    def _adapt(self, data, update_num=False):
        if data.max() > 1 or data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            if update_num:
                self._num_classes = data.max() + 1
            return data
        else:
            if update_num:
                res = np.argmax(data, axis=1)
                self._num_classes = res.max() + 1
            return res

    def _compute_param(self, set_a, set_b):
        return np.sum(np.logical_and(set_a, set_b))

    def _compute_parameters(self):

        self._tp = np.zeros((self._num_classes), dtype=int)
        self._fp = np.zeros((self._num_classes), dtype=int)
        self._fn = np.zeros((self._num_classes), dtype=int)
        for target in range(self._num_classes):
            self._tp[target] = self._compute_param(
                self._truth == target, self._predicted == target)
            self._fp[target] = self._compute_param(
                self._truth != target, self._predicted == target)
            self._fn[target] = self._compute_param(
                self._truth == target, self._predicted != target)

    def parameters(self):
        vmacro, vweigh, _, vmicro = self._compute_metrics()
        return dict(tp=self._tp, fp=self._fp, fn=self._fn, macro=vmacro, weigh=vweigh, micro=vmicro)

    def sets(self):
        return self._truth, self._predicted

    def fp_indices(self, target):
        return np.where(np.logical_and(self._truth != target, self._predicted == target))[0]

    def fn_indices(self, target):
        return np.where(np.logical_and(self._truth == target, self._predicted != target))[0]

    def correct(self):
        """
        Return the number of correct predictions
        """
        return np.sum(self._tp)

    def _get_dict(self, vmacro, vweigh, vmicro):
        return dict(macro=vmacro, weigh=vweigh, micro=vmicro)

    def recall(self, target):
        """
        recall, Recall is the ratio of correctly predicted positive observations to the all observations in positive class
        """
        if target == 'all':
            macro, weigh, _, micro = self._compute_metrics()
            return self._get_dict(macro['rec'], weigh['rec'], micro['rec'])
        else:
            tp = self._tp[target]
            fn = self._fn[target]
            if (tp + fn) > 0:
                return tp / (tp + fn)
            return 0

    def precision(self, target):
        """
        precision, Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
        """
        if target == 'all':
            macro, weigh, _, micro = self._compute_metrics()
            return self._get_dict(macro['prec'], weigh['prec'], micro['prec'])
        else:
            tp = self._tp[target]
            fp = self._fp[target]
            if (tp + fp) > 0:
                return tp / (tp + fp)
            return 0

    def accuracy(self):
        """
        accuracy, Accuracy is a ratio of correctly predicted observations to the total observations
        """
        tp = np.sum(self._tp)
        elements = self._truth.size
        if (elements) > 0:
            return tp / elements
        return 0

    def f1(self, target):
        """
        f1 score, is the weighted average of Precision and Recall
        """
        if target == 'all':
            macro, weigh, _, micro = self._compute_metrics()
            return self._get_dict(macro['f1'], weigh['f1'], micro['f1'])
        else:
            divider = self.recall(target) + self.precision(target)
            if divider != 0:
                return 2 * (self.recall(target) * self.precision(target)) / divider
            return 0

    def confusion_matrix(self):
        """
        Return the confusion matrix associated to the data provided
        """
        result = np.zeros((self._num_classes, self._num_classes), dtype=int)
        for target in reversed(range(self._num_classes)):
            for j in range(self._num_classes):
                result[target][j] = self._compute_param(
                    self._truth == target, self._predicted == j)
        return result

    def debug(self):
        for target in range(self._num_classes):
            tp = self._tp[target]
            fp = self._fp[target]
            fn = self._fn[target]
            print("target=[{0}], tp=[{1}], fp=[{2}], fn=[{3}]".format(
                target, tp, fp, fn))
            print("Truth shape=", self._truth.shape,
                  " Prediction shape=", self._predicted.shape)
            print("Number of classes:", self._num_classes)

    def _compute_micro_metrics(self):
        ttp = np.sum(self._tp)
        tfp = np.sum(self._fp)
        pr = re = ttp / (ttp + tfp)
        if ttp + tfp == 0:
            return 0
        return 2 * (pr * re) / (pr + re), pr, re

    def _compute_metrics(self):
        tf1 = tpr = tre = 0.0
        twf1 = twpr = twre = 0.0
        total_samples = 0
        for target in range(self._num_classes):
            f1 = self.f1(target)
            pr = self.precision(target)
            re = self.recall(target)
            num_samples = len(np.where(self._truth == target)[0])
            tf1 += f1
            tpr += pr
            tre += re
            twf1 += f1 * num_samples
            twpr += pr * num_samples
            twre += re * num_samples
            total_samples += num_samples
        tf1 /= self._num_classes
        tpr /= self._num_classes
        tre /= self._num_classes
        twf1 /= total_samples
        twpr /= total_samples
        twre /= total_samples
        mf1, mpr, mre = self._compute_micro_metrics()
        macro = {}
        weigh = {}
        micro = {}
        macro['f1'] = tf1
        macro['prec'] = tpr
        macro['rec'] = tre
        weigh['f1'] = twf1
        weigh['prec'] = twpr
        weigh['rec'] = twre
        micro['f1'] = mf1
        micro['prec'] = mpr
        micro['rec'] = mre
        return macro, weigh, total_samples, micro

    def classification_report(self, title='', digits=6):
        def format_line(a, b, c, d, e):
            return "[{0:^5}]\t[{1:.{digits}f}]\t[{2:.{digits}f}]\t[{3:.{digits}f}]\t[{4:5d}]".format(a, b, c, d, e, digits=digits)
        print(
            "======================== {0} ========================".format(title))

        header = ['target', 'f1-score', 'precision', 'recall', 'support']
        print("{d[0]:^7}\t{d[1]:^{length}.{length}}\t{d[2]:^{length}.{length}}\t{d[3]:^{length}.{length}}\t{d[4]:^7}".format(
            d=header, length=digits + 4))
        for target in range(self._num_classes):
            f1 = self.f1(target)
            pr = self.precision(target)
            re = self.recall(target)
            num_samples = len(np.where(self._truth == target)[0])
            print(format_line(target, f1, pr, re, num_samples))
        print("")
        macro, weigh, total_samples, micro = self._compute_metrics()
        print(format_line(
            'macro', macro['f1'], macro['prec'], macro['rec'], total_samples))
        print(format_line(
            'weig.', weigh['f1'], weigh['prec'], weigh['rec'], total_samples))
        print("accuracy=[{0:.{digits}f}]".format(
            self.accuracy(), digits=digits))
