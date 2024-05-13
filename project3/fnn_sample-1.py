# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:43:04 2019

Updated on Wed Jan 29 10:18:09 2020

@author: created by Sowmya Myneni and updated by Dijiang Huang
"""

########################################
# Part 1 - Data Pre-Processing
#######################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# 根据攻击类型加载训练和测试数据集
def load_dataset(training_attack_types, testing_attack_types, data_folder='NSL-KDD/'):
    training_file_name = "Training-"
    testing_file_name = "Testing-"
    if training_attack_types:
        training_file_name += '-'.join(training_attack_types) + '.csv'
    else:
        training_file_name += 'N.csv'

    if testing_attack_types:
        testing_file_name += '-'.join(testing_attack_types) + '.csv'
    else:
        testing_file_name += 'N.csv'

    training_dataset = pd.read_csv(data_folder + training_file_name, header=None)
    testing_dataset = pd.read_csv(data_folder + testing_file_name, header=None)

    return training_dataset, testing_dataset


# 初始化转换器
column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [1, 2, 3])],
    remainder='passthrough'
)
scaler = StandardScaler()

# 预处理数据集
def preprocess_dataset(dataset, fit_transformers=False):
    X = dataset.iloc[:, 0:-2].values  # 取所有行，除了最后两列
    label_column = dataset.iloc[:, -2].values  # 最后第二列是label
    y = [0 if label == 'normal' else 1 for label in label_column]

    if fit_transformers:
        # 只在训练数据上拟合转换器
        X = column_transformer.fit_transform(X)
        X = scaler.fit_transform(X)
    else:
        # 在测试数据上使用已拟合的转换器
        X = column_transformer.transform(X)
        X = scaler.transform(X)

    return X, np.array(y)



# 构建和编译模型
def build_compile_model(input_dim):
    model = Sequential()
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=input_dim))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 训练模型
def train_model(model, X_train, y_train, batch_size, num_epochs):
    return model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)


# 执行特定场景下的数据加载、预处理、模型构建、训练和评估
def execute_scenario(training_attack_types, testing_attack_types, batch_size, num_epochs, data_folder='NSL-KDD/'):
    print("Executing Scenario with Training: {}, Testing: {}".format(training_attack_types, testing_attack_types))

    # 加载数据集
    training_dataset, testing_dataset = load_dataset(training_attack_types, testing_attack_types, data_folder)

    # 预处理数据集
    X_train, y_train = preprocess_dataset(training_dataset, fit_transformers=True)
    X_test, y_test = preprocess_dataset(testing_dataset, fit_transformers=False)

    # 构建和编译模型
    model = build_compile_model(input_dim=len(X_train[0]))

    # 训练模型
    model_history = train_model(model, X_train, y_train, batch_size, num_epochs)

    # 预测测试集结果
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    # 计算混淆矩阵
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print('[ TN, FP ]')
    print('[ FN, TP ]=')
    print(confusion_matrix_result)

    # 绘制准确度和损失图
    plot_metrics(model_history, training_attack_types, testing_attack_types)

    # 返回混淆矩阵和模型训练历史
    return confusion_matrix_result, model_history


# 定义绘制指标的函数
def plot_metrics(history, training_attack_types, testing_attack_types):
    # 拼接场景名称, 用于图形文件名和图例
    scenario_name = 'Training_' + '_'.join(training_attack_types) + '__Testing_' + '_'.join(testing_attack_types)

    # 绘制准确率图形
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy (' + scenario_name + ')')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('accuracy_' + scenario_name + '.png')
    plt.show()

    # 绘制损失图形
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss (' + scenario_name + ')')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('loss_' + scenario_name + '.png')
    plt.show()


# 定义场景 SA、SB、SC 的攻击类型

scenario_a_training_attacks = ['a1', 'a3']
scenario_a_testing_attacks = ['a2', 'a4']

scenario_b_training_attacks = ['a1', 'a2']
scenario_b_testing_attacks = ['a1']

scenario_c_training_attacks = ['a1', 'a2']
scenario_c_testing_attacks = ['a1', 'a2', 'a3']

# 批大小和epoch数
batch_size = 10
num_epochs = 10

# 执行三个场景
confusion_matrix_a, history_a = execute_scenario(scenario_a_training_attacks, scenario_a_testing_attacks, batch_size, num_epochs)
confusion_matrix_b, history_b = execute_scenario(scenario_b_training_attacks, scenario_b_testing_attacks, batch_size, num_epochs)
confusion_matrix_c, history_c = execute_scenario(scenario_c_training_attacks, scenario_c_testing_attacks, batch_size, num_epochs)
