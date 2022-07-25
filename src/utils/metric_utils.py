import numpy as np


def compute_ap(recall: np.ndarray, precision: np.ndarray):
    """
    计算ap值
    :param recall:
    :param precision:
    :return:
    """
    m_recall = np.concatenate(([0.0], recall, [1.0]))
    m_precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(m_precision.size - 1, 0, -1):
        m_precision[i - 1] = np.maximum(m_precision[i - 1], m_precision[i])
    # 找出recall 不相等的点
    i = np.where(m_recall[1:] != m_recall[:-1])[0]
    # 计算面积 delta_recall * precision
    ap = np.sum((m_recall[i + 1] - m_recall[i]) * m_precision[i + 1])
    return ap
