from abc import ABC
from tqdm import tqdm
import numpy as np
import torch
from typing import List, Union, Dict, Any, Tuple
from src.model.Layer import MetricLayer
import src.utils.anchor_utils as anchor_utils
import src.utils.metric_utils as metric_utils


class DetectMetric(MetricLayer, ABC):
    """
    目标检测的评估方法
    """

    def __init__(self,
                 image_size: Tuple[int, int],
                 class_names: List[str],
                 predict_name: str,
                 target_name: str,
                 conf_threshold: float = 0.5,
                 ap_iou_threshold: float = 0.5,
                 nms_threshold: float = 0.5):
        super(DetectMetric, self).__init__(name="")
        self.image_size = image_size
        self.class_names = class_names
        self.class_ids: List[int] = list(range(len(class_names)))
        self.predict_name = predict_name
        self.target_name = target_name
        self.conf_threshold = conf_threshold
        self.ap_iou_threshold = ap_iou_threshold
        self.nms_threshold = nms_threshold

    def _get_batch_statistics(self, predict: List[np.ndarray],
                              targets: List[Union[None, np.ndarray]]) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        List[np.ndarray]]:
        """
        获取匹次统计结果
        :param predict:
        :param targets:
        :return:
        """
        batch_metrics = []
        batch_labels = []
        for sample_i in range(len(predict)):
            if predict[sample_i] is None:
                continue
            output = predict[sample_i]
            # 预测的目标框坐标
            predict_boxes = output[:, :4]
            # 预测的置信度
            predict_confidences = output[:, 4] * output[:, 5]
            # 预测的类别
            predict_labels = output[:, -1]
            # 记录 所有预测框当中 正确的框
            true_positives = np.zeros(predict_boxes.shape[0])
            # [[x, y, w, h, cls], ...]
            ground_true = targets[sample_i]
            if ground_true is None:
                # no ground true
                batch_metrics.append((true_positives, predict_labels, predict_confidences))
            else:
                ground_true = ground_true[np.sum(ground_true, axis=-1) != 0]

                ground_true[:, 0] = ground_true[:, 0] * self.image_size[0]
                ground_true[:, 2] = ground_true[:, 2] * self.image_size[0]
                ground_true[:, 1] = ground_true[:, 1] * self.image_size[1]
                ground_true[:, 3] = ground_true[:, 3] * self.image_size[1]

                ground_boxes = ground_true[:, :4]
                # [[min_x, min_y, max_x, max_y]]
                ground_boxes = anchor_utils.x_y_w_h_to_x_y_x_y(box=ground_boxes)
                ground_true_labels = ground_true[:, -1]
                batch_labels.append(ground_true_labels)
                # 记录已经确定预测框的下表
                detected_predict_boxes_index = []
                # 检测每一个目标框与预测框的最佳匹配
                for ground_true_label, ground_box in zip(ground_true_labels, ground_boxes):
                    # 如果预测的框当中 没有目标对应的类别
                    if ground_true_label not in predict_labels:
                        continue
                    filter_predict_boxes = []
                    filter_predict_boxes_ori_index = []
                    for _id, (predict_label, predict_box) in enumerate(zip(predict_labels, predict_boxes)):
                        if predict_label == ground_true_label and _id not in detected_predict_boxes_index:
                            filter_predict_boxes.append(predict_box)
                            filter_predict_boxes_ori_index.append(_id)
                    if len(filter_predict_boxes) == 0:
                        # 当前 ground_true 没有一个预测正确
                        continue
                    # 计算目标框与所有预测值的最佳匹配
                    iou_scores = anchor_utils.bbox_iou(bbox=np.array(filter_predict_boxes),
                                                       gt=np.array([ground_box]))
                    # 排序找出最大iou值
                    best_match_index = np.argmax(iou_scores)
                    best_iou_score = iou_scores[best_match_index]
                    best_match_target_box_ori_index = filter_predict_boxes_ori_index[best_match_index]
                    # iou值大于阈值的算做预测正确
                    if best_iou_score >= self.ap_iou_threshold:
                        detected_predict_boxes_index.append(best_match_target_box_ori_index)
                        true_positives[best_match_target_box_ori_index] = 1
                batch_metrics.append((true_positives, predict_labels, predict_confidences))
        return batch_metrics, batch_labels

    def metric_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor,
                                                               Any]],
                            summary: Dict[str, Any]):
        """

        :param tensor_dict:
        :param summary: 记录中间过程的评估值
        :return:
        """
        # [b, ?, (min_x, min_y, max_x, max_y, cls)]
        predict: List[Union[np.ndarray, None]] = tensor_dict[self.predict_name]
        # [b, (min_x, min_y, max_x, max_y, cls)]
        targets: List[Union[None, np.ndarray]] = [tensor.cpu().numpy() if tensor is not None else None
                                                  for tensor in tensor_dict[self.target_name]]
        # 临时结果记录
        if "sample_metrics" not in summary:
            summary["sample_metrics"] = []
        if "target" not in summary:
            summary["target"] = []

        batch_metrics, labels = self._get_batch_statistics(predict=predict,
                                                           targets=targets)
        summary["sample_metrics"] = summary["sample_metrics"] + batch_metrics
        summary["target"] = summary["target"] + [labels]

    def _metric_per_class(self, true_positives: np.ndarray,

                          predict_labels: np.ndarray,
                          target_cls: np.ndarray):
        """
        统计每一个类别的 ap 值
        :param true_positives: [?, ]
        :param predict_labels: [?, ]
        :param target_cls: [?, ]
        :return:
        """

        p, r = [], []
        for class_id in tqdm(self.class_ids, desc=""):
            # 对应类别 目标框的实际数目
            num_ground_true = (target_cls == class_id).astype(dtype=np.float32).sum()
            # 对应类别 目标框的预测数目
            num_predict = (predict_labels == class_id).astype(dtype=np.float32).sum()
            # 对应类别 目标框正确预测的数目
            class_true_positives = true_positives[predict_labels == class_id]

            if num_predict == 0 and num_ground_true == 0:
                r.append(0)
                p.append(0)
            elif num_predict == 0 or num_ground_true == 0:
                r.append(0)
                p.append(0)
            else:
                # 预测错误的数目
                fpc = (1 - class_true_positives).sum()
                # 预测正确的数目
                tpc = class_true_positives.sum()
                # 召回率
                recall = tpc / (num_ground_true + 1e-16)
                r.append(recall)
                # 精准度
                precision = tpc / (tpc + fpc)
                p.append(precision)
                # 计算ap值
        p, r = np.array(p), np.array(r)
        f1 = 2 * p * r / (p + r + 1e-16)
        p = [round(_, 4) for _ in p]
        r = [round(_, 4) for _ in r]
        f1 = [round(_, 4) for _ in f1]
        # 精准率 召回率 f1值
        return p, r, f1

    def _metric_(self,
                 true_positives: np.ndarray,
                 predict_confidences: np.ndarray,
                 obj_num: int):
        """
        :param true_positives:
        :param predict_confidences:
        :param obj_num:
        :return:
        """
        sort_index = predict_confidences.argsort()[::-1]
        true_positives = true_positives[sort_index]

        # 预测错误的数目
        fpc = (1 - true_positives).cumsum()
        # 预测正确的数目
        tpc = true_positives.cumsum()
        # 召回率 (recall = (tp)/(tp+tf))
        recall_curve = tpc / obj_num
        # 精准度 (precision = (tp)/(tp+fp))
        precision_curve = tpc / (tpc + fpc)

        ap = metric_utils.compute_ap(recall=recall_curve,
                                     precision=precision_curve)
        return recall_curve[-1], precision_curve[-1], ap

    def summary(self, summary: Dict[str, Any]):
        """
        生成最终的评估结果
        :param summary:
        :return:
        """
        sample_metrics = summary["sample_metrics"]
        if len(sample_metrics) > 0:
            true_positives, predict_labels, predict_confidences = \
                [np.concatenate(x, axis=0) for x in list(zip(*sample_metrics))]
            target_cls = []
            for batch_target in summary["target"]:
                for sample_target in batch_target:
                    for target in sample_target:
                        target_cls.append(int(target))
            precisions, recalls, f1_scores = self._metric_per_class(true_positives=true_positives,
                                                                    predict_labels=predict_labels,
                                                                    target_cls=np.array(target_cls))
            m_recall, m_precision, m_ap = self._metric_(true_positives=true_positives,
                                                        predict_confidences=predict_confidences,
                                                        obj_num=len(target_cls))
            summary_dict = {"m_recall": m_recall,
                            "m_precision": m_precision,
                            "m_ap": m_ap}
            for _id, class_name in enumerate(self.class_names):
                local_summary_dict = {"precision": precisions[_id],
                                      "recall": recalls[_id],
                                      "f1_score": f1_scores[_id]}
                summary_dict[class_name] = local_summary_dict
        else:
            summary_dict = {"m_recall": 0,
                            "m_precision": 0,
                            "m_ap": 0}
            for _id, class_name in enumerate(self.class_names):
                local_summary_dict = {"precision": 0,
                                      "recall": 0,
                                      "f1_score": 0}
                summary_dict[class_name] = local_summary_dict
        return summary_dict
