# ==================== Imports ====================
import os
import random
import numpy as np
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed

from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential, Model
from tensorflow.keras import initializers

# ==================== Random Seed ====================
seed_value = 0
random.seed(seed_value)
set_random_seed(seed_value)
seed(seed_value)
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 只显示致命错误


# TensorFlow session for reproducibility
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# ==================== Optimized CNN Autoencoder ====================
from keras.layers import (
    Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape,
    Dropout, BatchNormalization, Add, Multiply, GlobalAveragePooling1D, Lambda, LeakyReLU
)
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


def se_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation Attention Block"""
    filters = int(input_tensor.shape[-1])
    se = GlobalAveragePooling1D()(input_tensor)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Multiply()([input_tensor, K.expand_dims(se, axis=1)])
    return se

def residual_block(x, filters):
    shortcut = x
    x = Dense(filters)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(filters)(x)
    x = BatchNormalization()(x)

    # 自动匹配 shortcut 维度
    if shortcut.shape[-1] != filters:
        shortcut = Dense(filters)(shortcut)

    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.1)(x)
    return x


def build_autoencoder(input_dim, encoding_dim, seed_value=0):
    """修复后的 CNN 自编码器"""
    K.clear_session()
    input_layer = Input(shape=(input_dim, 1))

    # ===== 编码器 =====
    x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)

    # 保存展平前的形状，用于解码器
    shape_before_flatten = K.int_shape(x)  # (batch_size, timesteps, filters)

    x = Flatten()(x)
    encoded = Dense(encoding_dim, activation='relu', name='encoded')(x)

    # ===== 解码器 =====
    # 首先恢复到卷积层的形状
    x = Dense(shape_before_flatten[1] * shape_before_flatten[2], activation='relu')(encoded)
    x = Reshape((shape_before_flatten[1], shape_before_flatten[2]))(x)

    # 上采样和卷积
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)

    # 最终输出层
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    # 确保输出维度与输入匹配
    decoded = Lambda(lambda z: z[:, :input_dim, :])(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer=Adam(lr=1e-3), loss='mse')
    return autoencoder, encoder


def train_autoencoder(X, X_val, input_dim, encoding_dim, epochs=10000, batch_size=32):
    """
    CNN Autoencoder 训练逻辑 - 修复版本
    """
    # 确保输入是3D的
    X_3d = X.reshape((-1, input_dim, 1))
    X_val_3d = X_val.reshape((-1, input_dim, 1))

    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    hist = autoencoder.fit(
        X_3d,
        X_3d,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(X_val_3d, X_val_3d),
        callbacks=[early_stopping],
        verbose=0,
    )

    optimal_epochs = len(hist.history["loss"])

    # 重新用合并数据训练
    X_full = np.concatenate([X_3d, X_val_3d])
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.fit(
        X_full,
        X_full,
        epochs=optimal_epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
    )

    return autoencoder, encoder



# ==================== CNN Equivalent ====================
from keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, BatchNormalization
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_cnn_equivalent(input_dim, layer_sizes=[128, 128, 128], seed_value=0):
    K.clear_session()
    model_input = Input(shape=(input_dim, 1))
    x = model_input

    for i, filters in enumerate(layer_sizes):
        x = Conv1D(
            filters=filters,
            kernel_size=5 if i % 2 == 0 else 5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            bias_initializer="zeros"
        )(x)
        # x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.01)(x)



    x = GlobalAveragePooling1D()(x)
    # x = Dense(256, activation="relu", kernel_initializer="he_normal", bias_initializer="zeros")(x)
    # x = Dropout(0.6)(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal", bias_initializer="zeros")(x)
    x = Dropout(0.1)(x)
    output = Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform", bias_initializer="zeros")(x)

    model = Model(model_input, output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_cnn_equivalent(
        X,
        y,
        X_val,
        y_val,
        input_dim,
        layer_sizes=[128, 128, 128],
        epochs=1000,
        batch_size=32,
        seed_value=0
):
    # reshape 输入
    X_cnn = X.reshape((-1, input_dim, 1))
    X_val_cnn = X_val.reshape((-1, input_dim, 1))

    model = build_cnn_equivalent(input_dim, layer_sizes, seed_value)

    early_stopping = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)

    hist = model.fit(
        X_cnn,
        y,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_val_cnn, y_val),
        callbacks=[early_stopping, lr_scheduler],
        verbose=0,
    )

    optimal_epochs = len(hist.history["loss"])

    # 合并训练集和验证集重新训练
    X_cnn_full = np.concatenate([X_cnn, X_val_cnn])
    y_full = np.concatenate([y, y_val])

    model = build_cnn_equivalent(input_dim, layer_sizes, seed_value)
    model.fit(
        X_cnn_full,
        y_full,
        epochs=optimal_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    return model

# # ==================== GRU Equivalent ====================
# from keras.layers import GRU
#
# def build_gru_equivalent(input_dim, gru_units=[128, 128], seed_value=0):
#     """
#     与 CNN 分类器结构对应的 GRU 分类器
#     input_dim: 特征长度（时间步）
#     gru_units: GRU 层数量与隐藏单元数
#     """
#     K.clear_session()
#     model_input = Input(shape=(input_dim, 1))
#     x = model_input
#
#     # 堆叠 GRU 层
#     for i, units in enumerate(gru_units):
#         # 最后一层不返回序列
#         return_seq = True if i < len(gru_units) - 1 else False
#         x = GRU(units,
#                 return_sequences=return_seq,
#                 kernel_initializer="he_normal",
#                 bias_initializer="zeros")(x)
#         x = Dropout(0.1)(x)
#
#     # 全连接分类头
#     x = Dense(128, activation="relu",
#               kernel_initializer="he_normal",
#               bias_initializer="zeros")(x)
#     x = Dropout(0.1)(x)
#     output = Dense(1, activation="sigmoid",
#                    kernel_initializer="glorot_uniform",
#                    bias_initializer="zeros")(x)
#
#     model = Model(model_input, output)
#     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#     return model
#
#
#
# def train_gru_equivalent(
#         X,
#         y,
#         X_val,
#         y_val,
#         input_dim,
#         gru_units=[128, 128],
#         epochs=1000,
#         batch_size=32,
#         seed_value=0
# ):
#     """
#     完全匹配 CNN 版的训练逻辑：
#     1) reshape 数据
#     2) 训练（早停）
#     3) 根据最佳 epoch 合并训练集 + 验证集重新训练
#     """
#     # reshape 输入 (samples, timesteps=input_dim, feature=1)
#     X_gru = X.reshape((-1, input_dim, 1))
#     X_val_gru = X_val.reshape((-1, input_dim, 1))
#
#     model = build_gru_equivalent(input_dim, gru_units, seed_value)
#
#     early_stopping = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
#     lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
#
#     hist = model.fit(
#         X_gru,
#         y,
#         epochs=epochs,
#         batch_size=batch_size,
#         shuffle=True,
#         validation_data=(X_val_gru, y_val),
#         callbacks=[early_stopping, lr_scheduler],
#         verbose=0,
#     )
#
#     optimal_epochs = len(hist.history["loss"])
#
#     # ===== 合并训练集和验证集重新训练 =====
#     X_gru_full = np.concatenate([X_gru, X_val_gru])
#     y_full = np.concatenate([y, y_val])
#
#     model = build_gru_equivalent(input_dim, gru_units, seed_value)
#     model.fit(
#         X_gru_full,
#         y_full,
#         epochs=optimal_epochs,
#         batch_size=batch_size,
#         shuffle=True,
#         verbose=0,
#     )
#     return model



# # ==================== Imports ====================
# import os
# import random
# import numpy as np
# import tensorflow as tf
# from numpy.random import seed
# from tensorflow import set_random_seed
#
# from keras import backend as K
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.layers import (
#     Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape,
#     Dropout, BatchNormalization, Add, Multiply, GlobalAveragePooling1D,
#     Lambda, LeakyReLU
# )
# from keras.models import Model
# from keras.optimizers import Adam
#
#
# # ==================== Random Seed ====================
# seed_value = 0
# random.seed(seed_value)
# set_random_seed(seed_value)
# seed(seed_value)
# os.environ["PYTHONHASHSEED"] = "0"
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 只显示致命错误
#
# session_conf = tf.ConfigProto(
#     intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
# )
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
#
#
# # ==================== CNN Autoencoder ====================
# def build_autoencoder(input_dim, encoding_dim):
#     """CNN Autoencoder for feature extraction"""
#     K.clear_session()
#     input_layer = Input(shape=(input_dim, 1))
#
#     # 编码器
#     x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
#     x = MaxPooling1D(2, padding='same')(x)
#     x = Conv1D(64, 3, activation='relu', padding='same')(x)
#     x = MaxPooling1D(2, padding='same')(x)
#
#     shape_before_flatten = K.int_shape(x)
#     x = Flatten()(x)
#     encoded = Dense(encoding_dim, activation='relu', name='encoded')(x)
#
#     # 解码器
#     x = Dense(shape_before_flatten[1] * shape_before_flatten[2], activation='relu')(encoded)
#     x = Reshape((shape_before_flatten[1], shape_before_flatten[2]))(x)
#     x = Conv1D(64, 3, activation='relu', padding='same')(x)
#     x = UpSampling1D(2)(x)
#     x = Conv1D(32, 3, activation='relu', padding='same')(x)
#     x = UpSampling1D(2)(x)
#     decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
#     decoded = Lambda(lambda z: z[:, :input_dim, :])(decoded)
#
#     autoencoder = Model(input_layer, decoded)
#     encoder = Model(input_layer, encoded)
#
#     autoencoder.compile(optimizer=Adam(lr=1e-3), loss='mse')
#     return autoencoder, encoder
#
#
# def train_autoencoder(X, X_val, input_dim, encoding_dim, epochs=500, batch_size=32):
#     """Train CNN Autoencoder"""
#     X_3d = X.reshape((-1, input_dim, 1))
#     X_val_3d = X_val.reshape((-1, input_dim, 1))
#
#     autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
#     early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
#
#     hist = autoencoder.fit(
#         X_3d, X_3d,
#         epochs=epochs,
#         batch_size=batch_size,
#         shuffle=False,
#         validation_data=(X_val_3d, X_val_3d),
#         callbacks=[early_stopping],
#         verbose=0,
#     )
#
#     optimal_epochs = len(hist.history["loss"])
#
#     X_full = np.concatenate([X_3d, X_val_3d])
#     autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
#     autoencoder.fit(
#         X_full, X_full,
#         epochs=optimal_epochs,
#         batch_size=batch_size,
#         shuffle=False,
#         verbose=0,
#     )
#     return autoencoder, encoder
#
#
# # ==================== DNN Classifier ====================
# def build_dnn_equivalent(input_dim, layer_sizes=[64, 64, 64]):
#     """DNN classifier"""
#     K.clear_session()
#     model_input = Input(shape=(input_dim,))
#     x = model_input
#
#     for units in layer_sizes:
#         x = Dense(units, kernel_initializer="he_normal", bias_initializer="zeros")(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(alpha=0.1)(x)
#         x = Dropout(0.01)(x)
#
#     output = Dense(1, activation="sigmoid",
#                    kernel_initializer="glorot_uniform", bias_initializer="zeros")(x)
#
#     model = Model(model_input, output)
#     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#     return model
#
#
# def train_dnn_equivalent(
#         X_encoded, y,
#         X_val_encoded, y_val,
#         input_dim,
#         layer_sizes=[64, 64, 64],
#         epochs=100,
#         batch_size=32
# ):
#     """Train DNN classifier using encoder features"""
#     model = build_dnn_equivalent(input_dim, layer_sizes)
#     early_stopping = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
#     lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
#
#     hist = model.fit(
#         X_encoded, y,
#         epochs=epochs,
#         batch_size=batch_size,
#         shuffle=True,
#         validation_data=(X_val_encoded, y_val),
#         callbacks=[early_stopping, lr_scheduler],
#         verbose=0,
#     )
#
#     optimal_epochs = len(hist.history["loss"])
#
#     X_full = np.concatenate([X_encoded, X_val_encoded])
#     y_full = np.concatenate([y, y_val])
#     model = build_dnn_equivalent(input_dim, layer_sizes)
#     model.fit(
#         X_full, y_full,
#         epochs=optimal_epochs,
#         batch_size=batch_size,
#         shuffle=True,
#         verbose=0,
#     )
#     return model
#


