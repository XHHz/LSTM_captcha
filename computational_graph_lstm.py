# -*- coding: utf-8 -*-
# !/usr/bin/env python
# @Time    : 2018/9/26 14:24
# @Author  : xhh
# @Desc    : 利用RNN(循环神经网络)进行模型的训练
# @File    : computational_graph_lstm.py
# @Software: PyCharm
import tensorflow as tf
from config import *


def computational_graph_lstm (x, y, batch_size=batch_size):
    # 设置权重，和偏差Variable，random_normal并进行高斯初始化，num_units隐层单元，n_classes所属类别
    # weights and  biases of appropriate shape to accomplish above task
    out_weights = tf.Variable(tf.random_normal([num_units, n_classes]), name='out_weight')
    out_bias = tf.Variable(tf.random_normal([n_classes]), name='out_bias')

    # 构建网络,for _ in range(layer_num)进行循环迭代
    lstm_layer = [tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True) for _ in range(layer_num)]    # 创建两层的lstm
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layer, state_is_tuple=True)   # 将lstm连接在一起，即多个网络层进行迭代
    init_state = mlstm_cell.zero_state(batch_size, tf.float32)  # cell的初始状态

    # 输出层
    outputs = list()    # 每个cell的输出
    state = init_state

    # RNN 递归的神经网络
    with tf.variable_scope('RNN'):
        for timestep in range(time_steps):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = mlstm_cell(x[:, timestep, :], state)  # 这里的state保存了每一层 LSTM 的状态
            outputs.append(cell_output)

    # h_state = outputs[-1] #取最后一个cell输出
    # 计算输出层的第一个元素, 获取最后time-step的输出，使用全连接, 得到第一个验证码输出结果，out_bias偏差变量
    prediction_1 = tf.nn.softmax(tf.matmul(outputs[-4], out_weights)+out_bias)
    # 计算输出层的第二个元素, 输出第二个验证码预测结果
    prediction_2 = tf.nn.softmax(tf.matmul(outputs[-3], out_weights)+out_bias)
    # 计算输出层的第三个元素，输出第三个验证码预测结果
    prediction_3 = tf.nn.softmax(tf.matmul(outputs[-2], out_weights)+out_bias)
    # 计算输出层的第四个元素, 输出第四个验证码预测结果,size:[batch,num_class]
    prediction_4 = tf.nn.softmax(tf.matmul(outputs[-1], out_weights)+out_bias)
    # 输出连接
    prediction_all = tf.concat([prediction_1, prediction_2, prediction_3, prediction_4], 1)   #  4 * [batch, num_class] => [batch, 4 * num_class]
    prediction_all = tf.reshape(prediction_all, [batch_size, captcha_num, n_classes], name='prediction_merge')  # [4, batch, num_class] => [batch, 4, num_class]

    # 损失函数reduce_mean函数，计算batch纬度，对算法计算损失值计算方法，loss=-logp
    loss = -tf.reduce_mean(y * tf.log(prediction_all), name='loss')
    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction_all), reduction_indices=1))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_all,labels=y))
    # AdamOptimizer模型优化
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt').minimize(loss)

    # 模型评估
    pre_arg = tf.argmax(prediction_all, 2, name='predict')
    y_arg = tf.argmax(y,2)
    correct_prediction = tf.equal(pre_arg, y_arg)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')

    return opt, loss, accuracy, pre_arg, y_arg




