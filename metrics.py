# from keras import backend as K
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )
#
#
# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
#
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
#
#
# def evaluate_model(model, X, y):
#     predicted_probas = model.predict_proba(X)
#     try:
#         predictions = model.predict_classes(X)
#     except:
#         predictions = model.predict(X)
#     try:
#         auc = roc_auc_score(y, predicted_probas[:, 1], average="weighted")
#     except:
#         auc = roc_auc_score(y, predicted_probas[:, 0], average="weighted")
#     acc = accuracy_score(y, predictions)
#     prec = precision_score(y, predictions)
#     rec = recall_score(y, predictions)
#     f1 = f1_score(y, predictions)
#     return (acc, f1, prec, rec, auc, predicted_probas)



from keras import backend as K
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, average_precision_score
)
import numpy as np

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# def evaluate_model(model, X, y):
#     """
#     统一评估函数，兼容 sklearn 与 keras 模型。
#     """
#     # --- 1. 获取预测概率 ---
#     if hasattr(model, "predict_proba"):
#         predicted_probas = model.predict_proba(X)
#     else:
#         predicted_probas = model.predict(X)
#         # 如果是二维输出(eg. softmax)，取第二列为正类概率
#         if predicted_probas.ndim > 1 and predicted_probas.shape[1] > 1:
#             predicted_probas = predicted_probas[:, 1]
#         else:
#             predicted_probas = predicted_probas.reshape(-1, 1)
#
#     # --- 2. 获取类别预测 ---
#     if hasattr(model, "predict_classes"):
#         predictions = model.predict_classes(X)
#     else:
#         preds = model.predict(X)
#         # 将概率转换为类别（>0.5判为1）
#         predictions = (preds > 0.5).astype(int).ravel()
#
#     # --- 3. 计算各项指标 ---
#     try:
#         auc = roc_auc_score(y, predicted_probas, average="weighted")
#     except:
#         auc = 0.0  # 若 AUC 计算异常
#
#     acc = accuracy_score(y, predictions)
#     prec = precision_score(y, predictions)
#     rec = recall_score(y, predictions)
#     f1 = f1_score(y, predictions)
#
#     return (acc, f1, prec, rec, auc, predicted_probas)


def evaluate_model(model, X_test, y_test):
    predicted_probas = model.predict(X_test)
    y_pred = (predicted_probas > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, predicted_probas)
    aupr = average_precision_score(y_test, predicted_probas)
    mcc = matthews_corrcoef(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sp = tn / (tn + fp)

    return acc, f1, prec, rec, auc, aupr, mcc, sp, predicted_probas

