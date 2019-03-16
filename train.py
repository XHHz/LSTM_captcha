# -*- coding: utf-8 -*-
# !/usr/bin/env python
# @Time    : 2018/9/26 14:24
# @Author  : xhh
# @Desc    : 定义测试集
# @File    : train.py
# @Software: PyCharm

from util import *
from computational_graph_lstm import *


# 定义训练集
def train():
    # 初始化x,y都不是一个特定的值，placeholder是定义占位符
    x = tf.placeholder("float", [None, time_steps, n_input], name="x")  # 输入的图片
    y = tf.placeholder("float", [None, captcha_num, n_classes], name="y")  # 输入图片的标签

    # 计算图
    opt, loss, accuracy, pre_arg, y_arg = computational_graph_lstm(x, y)
    saver = tf.train.Saver()  # 创建训练模型保存类
    init = tf.global_variables_initializer()    # 初始化变量值

    # 创建 tensorflow session，session对象在使用完之后需要关闭资源，
    # 除显示的调用close外，在这里使用with代码块，自动关闭
    with tf.Session() as sess:
        sess.run(init)
        iter = 1
        while iter < iteration:
            batch_x, batch_y = get_batch()
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})   # 只运行优化迭代计算图
            # 让模型进行运行计算，每100次计算一下其损失值
            if iter % 100 == 0:
                los, acc, parg, yarg = sess.run([loss, accuracy, pre_arg, y_arg], feed_dict={x: batch_x, y: batch_y})
                print("For iter ", iter)
                print("Accuracy ", acc)
                print("Loss ", los)
                if iter % 1000 == 0:
                    print("predict arg:", parg[0:10])
                    print("yarg:", yarg[0:10])
                print("__________________")
                if acc > 0.95:
                    print("training complete, accuracy:", acc)
                    break
            if iter % 1000 == 0:   # 保存模型，每迭代1000次，将模型进行保存
                saver.save(sess, model_path, global_step=iter)
            iter += 1
        # 计算验证集准确率
        valid_x, valid_y = get_batch(data_path=validation_path, is_training=False)
        print("Validation Accuracy:", sess.run(accuracy, feed_dict={x: valid_x, y: valid_y}))

        
if __name__ == '__main__':
    train()

