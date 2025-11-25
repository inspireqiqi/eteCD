import utils
import numpy as np
import os
import argparse
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, precision_recall_curve
import pandas as pd
import models
import metrics


# ==================== Paths ====================
DATA_DIR = "./data/cleaned/"
INPUT_DIR = "./inputs/"
FEATURES_DIR = os.path.join(INPUT_DIR, "features")
OUTPUTS_DIR = "./outputs/"
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results/cv/")

# ==================== Parameters ====================
epochs = 100
batch_size = 32
run_ae = True
encoding_dim = 512
layer_sizes = (1024, 512, 256, 128, 64)

# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        help="Please specify whether you want to run cross validation on CircR2Disease or Circ2Disease dataset",
        required=True,
    )
    args = parser.parse_args()
    data_name = args.data_name

    # check if data name is valid
    assert data_name in ["CircR2Disease", "Circ2Disease"]

    # load data
    all_pairs = utils.load_pickle(INPUT_DIR, f"{data_name}_all_pairs.pkl")
    print(len(all_pairs))
    all_labels = utils.load_pickle(INPUT_DIR, f"{data_name}_all_labels.pkl")
    [unique_circrnas, unique_diseases] = utils.load_pickle(
        INPUT_DIR, f"{data_name}_unique_circrnas_diseases.pkl"
    )

    # load features
    GIP_CC = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_CD.pkl")
    GIP_DD = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DC.pkl")
    SIM_CC = utils.load_pickle(FEATURES_DIR, f"{data_name}_SIM_CC.pkl")
    SIM_DD = utils.load_pickle(FEATURES_DIR, f"{data_name}_SIM_DD.pkl")

    circrna_feature_matrices = [GIP_CC]
    disease_feature_matrices = [GIP_DD,SIM_DD]

    RESULTS = []
    test_scores = []
    roc_curve_data = []
    pr_curve_data = []
    fold = 1

    # ==================== Cross Validation ====================
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in skf.split(all_pairs, all_labels):
        X_train, X_test = all_pairs[train_index], all_pairs[test_index]
        y_train, y_test = all_labels[train_index], all_labels[test_index]

        # ===== Features Construction =====
        X_train_circrna_vecs = np.array([
            utils.get_circrna_vec(unique_circrnas, circrna_feature_matrices, c)
            for c, d in X_train
        ])
        X_train_disease_vecs = np.array([
            utils.get_disease_vec(unique_diseases, disease_feature_matrices, d)
            for c, d in X_train
        ])
        X_train = np.concatenate([X_train_circrna_vecs, X_train_disease_vecs], axis=1)
        ae_input_dim = X_train.shape[1]

        X_test_circrna_vecs = np.array([
            utils.get_circrna_vec(unique_circrnas, circrna_feature_matrices, c)
            for c, d in X_test
        ])
        X_test_disease_vecs = np.array([
            utils.get_disease_vec(unique_diseases, disease_feature_matrices, d)
            for c, d in X_test
        ])
        X_test = np.concatenate([X_test_circrna_vecs, X_test_disease_vecs], axis=1)

        for tr_ind, val_ind in StratifiedShuffleSplit(
            n_splits=1, test_size=0.3, random_state=0
        ).split(X_train, y_train):
            X_train_, X_val = X_train[tr_ind], X_train[val_ind]
            y_train_, y_val = y_train[tr_ind], y_train[val_ind]

        # ===== Train Autoencoder =====
        if run_ae:
            autoencoder, encoder = models.train_autoencoder(
                X_train_,
                X_val,
                X_train_.shape[1],
                encoding_dim=encoding_dim,
                epochs=epochs,
                batch_size=batch_size,
            )

            # reshape to 3D for CNN input
            X_train_3d = X_train.reshape(-1, X_train.shape[1], 1)
            X_train_3d_ = X_train_.reshape(-1, X_train_.shape[1], 1)
            X_val_3d = X_val.reshape(-1, X_val.shape[1], 1)
            X_test_3d = X_test.reshape(-1, X_test.shape[1], 1)

            X_train_encoded = encoder.predict(X_train_3d)
            X_train_encoded_ = encoder.predict(X_train_3d_)
            X_val_encoded = encoder.predict(X_val_3d)
            X_test_encoded = encoder.predict(X_test_3d)
        else:
            autoencoder, ae_input_dim = None, None
            X_train_encoded = X_train
            X_train_encoded_ = X_train_
            X_val_encoded = X_val
            X_test_encoded = X_test
            encoding_dim = X_train_encoded.shape[1]

        # ===== Train Classifier =====
        dnn = models.train_cnn_equivalent(
            X_train_encoded_.reshape(-1, encoding_dim, 1),
            y_train_.reshape(y_train_.shape[0], 1),
            X_val_encoded.reshape(-1, encoding_dim, 1),
            y_val.reshape(y_val.shape[0], 1),
            input_dim=encoding_dim,
            epochs=epochs,
            batch_size=batch_size,

        )
        # dnn = models.train_gru_equivalent(
        #     X_train_encoded_.reshape(-1, encoding_dim, 1),
        #     y_train_.reshape(y_train_.shape[0], 1),
        #     X_val_encoded.reshape(-1, encoding_dim, 1),
        #     y_val.reshape(y_val.shape[0], 1),
        #     input_dim=encoding_dim,
        #     epochs=epochs,
        #     batch_size=batch_size,
        # )

        # ===== Evaluate =====
        acc, f1, prec, rec, auc, aupr, mcc, sp, predicted_probas = metrics.evaluate_model(
            dnn, X_test_encoded.reshape(-1, encoding_dim, 1), y_test.reshape(y_test.shape[0], 1)
        )

        # store
        test_scores.append((acc, f1, prec, rec, auc, aupr, mcc, sp, predicted_probas))

        fpr, tpr, threshold = roc_curve(y_test.reshape(y_test.shape[0], 1), predicted_probas)
        precision, recall, thresholds = precision_recall_curve(
            y_test.reshape(y_test.shape[0], 1), predicted_probas
        )

        roc_curve_data.append((fold, fpr, tpr, threshold))
        pr_curve_data.append((fold, recall, precision, threshold))
        RESULTS.append([fold, acc, f1, prec, rec, auc, aupr, mcc, sp])

        print(f"fold {fold}: acc={acc:.4f}, f1={f1:.4f}, prec={prec:.4f}, rec={rec:.4f}, auc={auc:.4f}, aupr={aupr:.4f}, mcc={mcc:.4f}, sp={sp:.4f}")
        fold += 1

    # ===== Compute Average Metrics =====
    test_acc_scores = np.array([acc for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
    test_f1_scores = np.array([f1 for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
    test_prec_scores = np.array([prec for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
    test_rec_scores = np.array([rec for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
    test_auc_scores = np.array([auc for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
    test_aupr_scores = np.array([aupr for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
    test_mcc_scores = np.array([mcc for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
    test_sp_scores = np.array([sp for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])

    RESULTS.append((
        "average",
        f"{test_acc_scores.mean():.4f} ± {test_acc_scores.std():.3f}",
        f"{test_f1_scores.mean():.4f} ± {test_f1_scores.std():.3f}",
        f"{test_prec_scores.mean():.4f} ± {test_prec_scores.std():.3f}",
        f"{test_rec_scores.mean():.4f} ± {test_rec_scores.std():.3f}",
        f"{test_auc_scores.mean():.4f} ± {test_auc_scores.std():.3f}",
        f"{test_aupr_scores.mean():.4f} ± {test_aupr_scores.std():.3f}",
        f"{test_mcc_scores.mean():.4f} ± {test_mcc_scores.std():.3f}",
        f"{test_sp_scores.mean():.4f} ± {test_sp_scores.std():.3f}",
    ))

    # ===== Print Average =====
    print("\n=== Cross Validation Average Results ===")
    print(f"Accuracy : {test_acc_scores.mean():.4f} ± {test_acc_scores.std():.3f}")
    print(f"F1 Score : {test_f1_scores.mean():.4f} ± {test_f1_scores.std():.3f}")
    print(f"Precision: {test_prec_scores.mean():.4f} ± {test_prec_scores.std():.3f}")
    print(f"Recall   : {test_rec_scores.mean():.4f} ± {test_rec_scores.std():.3f}")
    print(f"AUC      : {test_auc_scores.mean():.4f} ± {test_auc_scores.std():.3f}")
    print(f"AUPR     : {test_aupr_scores.mean():.4f} ± {test_aupr_scores.std():.3f}")
    print(f"MCC      : {test_mcc_scores.mean():.4f} ± {test_mcc_scores.std():.3f}")
    print(f"Specificity: {test_sp_scores.mean():.4f} ± {test_sp_scores.std():.3f}")
    print("======================================\n")

    # ===== Save Results =====
    result_file = os.path.join(RESULTS_DIR, f"{data_name}_results_1.csv")
    roc_file_path = os.path.join(RESULTS_DIR, f"{data_name}_roc_auc_curve_1GGS.png")
    pr_file_path = os.path.join(RESULTS_DIR, f"{data_name}_pr_curve_1GGS.png")
    # result_file = os.path.join(RESULTS_DIR, f"{data_name}_GRU_results.csv")
    # roc_file_path = os.path.join(RESULTS_DIR, f"{data_name}_GRU_roc.png")
    # pr_file_path = os.path.join(RESULTS_DIR, f"{data_name}_GRU_pr.png")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(RESULTS, columns=[
        "Fold", "Accuracy", "F1", "Precision", "Recall", "AUC", "AUPR", "MCC", "Specificity"
    ]).to_csv(result_file, index=False)

    utils.plot_curve(roc_curve_data, roc_file_path, "False Positive Rate", "True Positive Rate")
    utils.plot_curve(pr_curve_data, pr_file_path, "Recall", "Precision")

    print(
        f"Results are saved to:  {result_file}\n"
        f"ROC AUC curve plot is saved to: {roc_file_path}\n"
        f"PR curve plot is saved to: {pr_file_path}"
    )





# import utils
# import numpy as np
# import os
# import argparse
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# import models
# import metrics
#
#
# # ==================== Paths ====================
# DATA_DIR = "./data/cleaned/"
# INPUT_DIR = "./inputs/"
# FEATURES_DIR = os.path.join(INPUT_DIR, "features")
# OUTPUTS_DIR = "./outputs/"
# RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results/cv/")
#
# # ==================== Parameters ====================
# epochs = 100
# batch_size = 32
# run_ae = True
# encoding_dim = 512
# layer_sizes = (1024, 512, 256, 128, 64)
# n_estimators = 30
# max_depth = 2
# random_state = 42
#
# # =====================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--data_name",
#         type=str,
#         help="Please specify whether you want to run cross validation on CircR2Disease or Circ2Disease dataset",
#         required=True,
#     )
#     args = parser.parse_args()
#     data_name = args.data_name
#
#     # check if data name is valid
#     assert data_name in ["CircR2Disease", "Circ2Disease"]
#
#     # load data
#     all_pairs = utils.load_pickle(INPUT_DIR, f"{data_name}_all_pairs.pkl")
#     all_labels = utils.load_pickle(INPUT_DIR, f"{data_name}_all_labels.pkl")
#     [unique_circrnas, unique_diseases] = utils.load_pickle(
#         INPUT_DIR, f"{data_name}_unique_circrnas_diseases.pkl"
#     )
#
#     # load features
#     GIP_CD = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_CD.pkl")
#     GIP_DC = utils.load_pickle(FEATURES_DIR, f"{data_name}_GIP_DC.pkl")
#     SIM_DD = utils.load_pickle(FEATURES_DIR, f"{data_name}_SIM_DD.pkl")
#
#     circrna_feature_matrices = [GIP_CD]
#     disease_feature_matrices = [GIP_DC, SIM_DD]
#
#     RESULTS = []
#     test_scores = []
#     roc_curve_data = []
#     pr_curve_data = []
#     fold = 1
#
#     # ==================== Cross Validation ====================
#     skf = StratifiedKFold(n_splits=5, shuffle=False)
#     for train_index, test_index in skf.split(all_pairs, all_labels):
#         X_train, X_test = all_pairs[train_index], all_pairs[test_index]
#         y_train, y_test = all_labels[train_index], all_labels[test_index]
#
#         # ===== Features Construction =====
#         X_train_circrna_vecs = np.array([
#             utils.get_circrna_vec(unique_circrnas, circrna_feature_matrices, c)
#             for c, d in X_train
#         ])
#         X_train_disease_vecs = np.array([
#             utils.get_disease_vec(unique_diseases, disease_feature_matrices, d)
#             for c, d in X_train
#         ])
#         X_train = np.concatenate([X_train_circrna_vecs, X_train_disease_vecs], axis=1)
#         ae_input_dim = X_train.shape[1]
#
#         X_test_circrna_vecs = np.array([
#             utils.get_circrna_vec(unique_circrnas, circrna_feature_matrices, c)
#             for c, d in X_test
#         ])
#         X_test_disease_vecs = np.array([
#             utils.get_disease_vec(unique_diseases, disease_feature_matrices, d)
#             for c, d in X_test
#         ])
#         X_test = np.concatenate([X_test_circrna_vecs, X_test_disease_vecs], axis=1)
#
#         for tr_ind, val_ind in StratifiedShuffleSplit(
#             n_splits=1, test_size=0.3, random_state=0
#         ).split(X_train, y_train):
#             X_train_, X_val = X_train[tr_ind], X_train[val_ind]
#             y_train_, y_val = y_train[tr_ind], y_train[val_ind]
#
#         # ===== Train Autoencoder =====
#         if run_ae:
#             autoencoder, encoder = models.train_autoencoder(
#                 X_train_,
#                 X_val,
#                 X_train_.shape[1],
#                 encoding_dim=encoding_dim,
#                 epochs=epochs,
#                 batch_size=batch_size,
#             )
#
#             # X_train_encoded = encoder.predict(X_train)
#             # X_train_encoded_ = encoder.predict(X_train_)
#             # X_val_encoded = encoder.predict(X_val)
#             # X_test_encoded = encoder.predict(X_test)
#
#             X_train_encoded = encoder.predict(np.expand_dims(X_train, axis=-1))
#             X_train_encoded_ = encoder.predict(np.expand_dims(X_train_, axis=-1))
#             X_val_encoded = encoder.predict(np.expand_dims(X_val, axis=-1))
#             X_test_encoded = encoder.predict(np.expand_dims(X_test, axis=-1))
#
#         else:
#             X_train_encoded = X_train
#             X_train_encoded_ = X_train_
#             X_val_encoded = X_val
#             X_test_encoded = X_test
#             encoding_dim = X_train_encoded.shape[1]
#
#         # # ===== Train Random Forest Classifier =====
#         # # ===== Train XGBoost Classifier =====
#         # from xgboost import XGBClassifier
#         #
#         # xgb = XGBClassifier(
#         #     n_estimators=n_estimators,
#         #     max_depth=max_depth,
#         #     learning_rate=0.05,
#         #     subsample=0.8,
#         #     colsample_bytree=0.8,
#         #     objective="binary:logistic",
#         #     random_state=random_state,
#         #     eval_metric="logloss",
#         #     n_jobs=-1,
#         #     use_label_encoder=False
#         # )
#         #
#         # xgb.fit(X_train_encoded_, y_train_)
#         #
#         # # ===== Evaluate =====
#         # predicted_probas = xgb.predict_proba(X_test_encoded)[:, 1]
#         # predicted_labels = (predicted_probas >= 0.5).astype(int)
#
#         acc = accuracy_score(y_test, predicted_labels)
#         f1 = f1_score(y_test, predicted_labels)
#         prec = precision_score(y_test, predicted_labels)
#         rec = recall_score(y_test, predicted_labels)
#         auc = roc_auc_score(y_test, predicted_probas)
#         aupr = average_precision_score(y_test, predicted_probas)
#         mcc = matthews_corrcoef(y_test, predicted_labels)
#         tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels).ravel()
#         sp = tn / (tn + fp)
#
#         # store
#         test_scores.append((acc, f1, prec, rec, auc, aupr, mcc, sp, predicted_probas))
#
#         fpr, tpr, threshold = roc_curve(y_test, predicted_probas)
#         precision, recall, thresholds = precision_recall_curve(y_test, predicted_probas)
#         roc_curve_data.append((fold, fpr, tpr, threshold))
#         pr_curve_data.append((fold, recall, precision, threshold))
#         RESULTS.append([fold, acc, f1, prec, rec, auc, aupr, mcc, sp])
#
#         print(f"fold {fold}: acc={acc:.4f}, f1={f1:.4f}, prec={prec:.4f}, rec={rec:.4f}, auc={auc:.4f}, aupr={aupr:.4f}, mcc={mcc:.4f}, sp={sp:.4f}")
#         fold += 1
#
#     # ===== Compute Average Metrics =====
#     test_acc_scores = np.array([acc for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
#     test_f1_scores = np.array([f1 for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
#     test_prec_scores = np.array([prec for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
#     test_rec_scores = np.array([rec for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
#     test_auc_scores = np.array([auc for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
#     test_aupr_scores = np.array([aupr for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
#     test_mcc_scores = np.array([mcc for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
#     test_sp_scores = np.array([sp for (acc, f1, prec, rec, auc, aupr, mcc, sp, _) in test_scores])
#
#     RESULTS.append((
#         "average",
#         f"{test_acc_scores.mean():.4f} ± {test_acc_scores.std():.3f}",
#         f"{test_f1_scores.mean():.4f} ± {test_f1_scores.std():.3f}",
#         f"{test_prec_scores.mean():.4f} ± {test_prec_scores.std():.3f}",
#         f"{test_rec_scores.mean():.4f} ± {test_rec_scores.std():.3f}",
#         f"{test_auc_scores.mean():.4f} ± {test_auc_scores.std():.3f}",
#         f"{test_aupr_scores.mean():.4f} ± {test_aupr_scores.std():.3f}",
#         f"{test_mcc_scores.mean():.4f} ± {test_mcc_scores.std():.3f}",
#         f"{test_sp_scores.mean():.4f} ± {test_sp_scores.std():.3f}",
#     ))
#
#     # ===== Print Average =====
#     print("\n=== Cross Validation Average Results ===")
#     print(f"Accuracy : {test_acc_scores.mean():.4f} ± {test_acc_scores.std():.3f}")
#     print(f"F1 Score : {test_f1_scores.mean():.4f} ± {test_f1_scores.std():.3f}")
#     print(f"Precision: {test_prec_scores.mean():.4f} ± {test_prec_scores.std():.3f}")
#     print(f"Recall   : {test_rec_scores.mean():.4f} ± {test_rec_scores.std():.3f}")
#     print(f"AUC      : {test_auc_scores.mean():.4f} ± {test_auc_scores.std():.3f}")
#     print(f"AUPR     : {test_aupr_scores.mean():.4f} ± {test_aupr_scores.std():.3f}")
#     print(f"MCC      : {test_mcc_scores.mean():.4f} ± {test_mcc_scores.std():.3f}")
#     print(f"Specificity: {test_sp_scores.mean():.4f} ± {test_sp_scores.std():.3f}")
#     print("======================================\n")
#
#     # ===== Save Results =====
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     result_file = os.path.join(RESULTS_DIR, f"{data_name}_XGboost_results.csv")
#     roc_file_path = os.path.join(RESULTS_DIR, f"{data_name}_XGboost_roc.png")
#     pr_file_path = os.path.join(RESULTS_DIR, f"{data_name}_XGboost_pr.png")
#
#     pd.DataFrame(RESULTS, columns=[
#         "Fold", "Accuracy", "F1", "Precision", "Recall", "AUC", "AUPR", "MCC", "Specificity"
#     ]).to_csv(result_file, index=False)
#
#     utils.plot_curve(roc_curve_data, roc_file_path, "False Positive Rate", "True Positive Rate")
#     utils.plot_curve(pr_curve_data, pr_file_path, "Recall", "Precision")
#
#     print(
#         f"Results are saved to:  {result_file}\n"
#         f"ROC AUC curve plot is saved to: {roc_file_path}\n"
#         f"PR curve plot is saved to: {pr_file_path}"
#     )
#
