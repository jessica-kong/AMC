from prettytable import PrettyTable
import numpy as np
from sklearn.metrics import classification_report


class EarlyStopController(object):
    """
    A controller for early stopping.
    Args:
        patience (int):
            Maximum number of consecutive epochs without breaking the best record.
        higher_is_better (bool, optional, defaults to True):
            Whether a higher record is seen as a better one.
    """

    def __init__(self, patience: int, higher_is_better=True):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.early_stop = False
        self.hit = False
        self.counter = 0

        self.best = None
        self.best_model = None
        self.best_epoch = None

    def __call__(self, score: float, model, epoch: int):
        """Calls this after getting the validation metric each epoch."""
        # first calls
        if self.best is None:
            self.best = score
            self.model = model
            self.hit = True
            self.best_epoch = epoch
        else:
            # not hits the best record
            if (self.higher_is_better and score < self.best) or (not self.higher_is_better and score > self.best):
                self.hit = False
                self.counter += 1
                if self.counter > self.patience:
                    self.early_stop = True
            # hits the best record
            else:
                self.hit = True
                self.counter = 0
                self.best = score
                self.best_model = model
                self.best_epoch = epoch


def acc_for_multilabel(preds, golds):
    assert len(preds) == len(golds)
    total_score = 0
    for pred, gold in zip(preds, golds):
        two_equal = True
        for i in range(len(pred)):
            if int(pred[i]) != int(gold[i]):
                two_equal = False
        if two_equal:
            total_score += 1
    return total_score / len(preds)


def pretty_results(result_dict: dict):
    """
    Post-processes the evaluation result dict, such as generates result table and extracts major score.

    Args:
        result_dict (dict):
            A dict mapping from metric name to its score.
    Returns:
        result_table (PrettyTable):
            A table of results.
    """
    results_table = PrettyTable()
    results_table.field_names = ["Metric", "Score"]
    results_table.align["Metric"] = "c"
    results_table.align["Score"] = "l"
    for metric, score in result_dict.items():
        results_table.add_row([metric, str(score)])
    return results_table


def label_weight(golds):
    weights = [0, 0, 0, 0, 0]
    for gold in golds:
        for i, v in enumerate(gold):
            if int(v) == 1:
                w = weights[i] + 1
                weights[i] = w
    return weights
def multi2binary(labels):
    outputs = [[], [], [], [], []]  # 5个list 分别是每个AM的预测or真实label, e.g.[0,1, 1, 0, 0, 0]
    for label in labels:
        for i, v in enumerate(label):
            lis = outputs[i]
            lis.append(v)
            outputs[i] = lis
    return outputs


def f1_scores(golds, preds):
    weights = label_weight(golds)
    golds, preds = multi2binary(golds), multi2binary(preds)
    f1s = []
    for i in range(5):
        gold, pred = golds[i], preds[i]
        report = classification_report(gold, pred)
        spl = report.split('\n')
        for s in spl:
            if '  1.0   ' in s:
                vlist = s.split()
                f1 = float(vlist[3])
                f1s.append(f1)

    sums = 0
    for i in range(5):
        sums += f1s[i] * weights[i]
    ave = round(sums / sum(weights), 4)
    f1s.insert(0, ave)
    return f1s


def get_mypreds(my_preds, logits, thre):
    for lo in logits:
        my_pred = []
        has_one = False
        for v in lo.tolist():
            if v > thre:
                my_pred.append(1.0)
                has_one = True
            else:
                my_pred.append(0.0)
        if not has_one:
            max_id,max_v = -1, 0
            for k, va in enumerate(lo.tolist()):
                if va > max_v:
                    max_v = va
                    max_id = k
            my_pred[max_id] = 1.0
        my_preds.append(my_pred)

    return my_preds
