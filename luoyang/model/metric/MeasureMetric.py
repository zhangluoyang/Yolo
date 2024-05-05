import torch
import numpy as np
from tqdm import tqdm
from luoyang.model.Layer import MetricLayer
from typing import List, Union, Dict, Any
from sklearn.metrics import f1_score, recall_score, precision_score


class MeasureMetric(MetricLayer):

    def __init__(self, feature_name: str,
                 targets_name: str,
                 name: str = "measure"):
        super(MeasureMetric, self).__init__(name=name)
        self.feature_name = feature_name
        self.targets_name = targets_name

    def metric_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor,
                                                               List[torch.Tensor],
                                                               np.ndarray,
                                                               List[str]]],
                            summary: Dict[str, Any]):
        """

        :param tensor_dict:
        :param summary:
        :return:
        """
        batch_features: np.ndarray = tensor_dict[self.feature_name]

        batch_targets: List[str] = tensor_dict[self.targets_name]

        if "features" not in summary:
            summary["features"] = []

        if "targets" not in summary:
            summary["targets"] = []

        summary["features"].append(batch_features)
        summary["targets"].extend(batch_targets)

    def summary(self, summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """

        :param summary:
        :return:
        """
        # [size, dim]
        features = np.concatenate(summary["features"], axis=0)
        # [size]
        targets = summary["targets"]
        # [size, size]
        similarity = np.dot(features, features.transpose())
        # [size]
        nearst_ids = np.argsort(similarity, axis=-1)
        predicts = [targets[_id[-2]] for _id in nearst_ids]

        labels = set(targets)

        metric_summary = {}
        all_nums = 0
        for label in tqdm(labels):
            y_true = [1 if target == label else 0 for target in targets]
            y_pre = [1 if pre == label else 0 for pre in predicts]
            p_score = precision_score(y_true=y_true,
                                      y_pred=y_pre)
            r_score = recall_score(y_true=y_true,
                                   y_pred=y_pre)
            f_score = f1_score(y_true=y_true,
                               y_pred=y_pre)
            num = sum(y_true)
            metric_summary[label] = {"precision_score": p_score,
                                     "recall_score": r_score,
                                     "f1_score": f_score,
                                     "num": num}

            all_nums += num

        _precision_score = 0
        _recall_score = 0
        _f1_score = 0

        for label in metric_summary.keys():
            _precision_score += (metric_summary[label]["num"] / all_nums) * metric_summary[label]["precision_score"]
            _recall_score += (metric_summary[label]["num"] / all_nums) * metric_summary[label]["recall_score"]
            _f1_score += (metric_summary[label]["num"] / all_nums) * metric_summary[label]["f1_score"]

        return {"mean": {"precision_score": _precision_score,
                         "recall_score": _recall_score,
                         "f1_score": _f1_score}}
