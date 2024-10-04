import numpy as np
from sklearn.metrics import accuracy_score

def calc_accuracy(y_true, y_pred):
    """
    Вычисляет точность на основе истинных и предсказанных значений.

    Args:
        y_true (np.ndarray): Массив истинных значений меток.
        y_pred (np.ndarray): Массив предсказанных значений меток.

    Returns:
        float: Значение точности (accuracy).
    """
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)
