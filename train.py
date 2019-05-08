# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
from datetime import datetime
import time
import os
import tensorflow as tf
import csv
import pandas as pd
from pandas import Series, DataFrame
from numpy import array
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#tf.device('/gpu:1')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True



INPUT_NODE = 2304
OUTPUT_NODE = 7
LAYER1_NODE = 4096
LAYER2_NODE = 512
LAYER3_NODE = 64
BATCH_SIZE = 1024
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.992
REGULARIZER = 0.0001
STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99
TEST_INTERVAL_SECS = 1
MODEL_SAVE_PATH="check_point"
MODEL_NAME="data_model"



def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1,mean=0))
    #损失函数loss含正则化regularization
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] =1，意思是不对样本个数和channel进行卷积
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")  # padding="SAME"用零填充边界


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


def forward(x, regularizer):
 
    x = tf.reshape(x, [-1, 48, 48, 1])
    
    ## convl layer ##
    W_conv1 = get_weight([5,5,1,128],None) # kernel 5*5, channel is 1, out size 32
    b_conv1 = get_bias([128])
    h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)  # output size 28*28*32
    h_pool1 = max_pool_2x2(h_conv1)                          # output size 14*14*32
    
    ## conv2 layer ##
    W_conv2 = get_weight([5,5,128,64],None) # kernel 5*5, in size 32, out size 64
    b_conv2 = get_bias([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)  # output size 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)                          # output size 7*7*64
    
    ## conv3 layer ##
    W_conv3 = get_weight([5,5,64,32],None) # kernel 5*5, in size 32, out size 64
    b_conv3 = get_bias([32])
    h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)  # output size 14*14*64
    h_pool3 = max_pool_2x2(h_conv3)                          # output size 7*7*64
    
    ## funcl layer ##
    W_fc1 = get_weight([1*6*6*32, 576], regularizer)
    b_fc1 = get_bias([576])
    
    h_pool2_flat = tf.reshape(h_pool3, [-1, 1*6*6*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob = 0.5)
    
    ## func2 layer ##
    W_fc2 = get_weight([576, 128], regularizer)
    b_fc2 = get_bias([128])
    y1 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
    
    W_fc3 = get_weight([128, OUTPUT_NODE], regularizer)
    b_fc3 = get_bias([OUTPUT_NODE])
    y = (tf.matmul(y1,W_fc3)+b_fc3)
    tf.add_to_collection('pred_network', y)  # 用于加载模型获取要预测的网络结构
    return W_fc1,b_fc1,h_fc1,W_fc2,b_fc2,y


def backward(new_data):
    step_train = 0
    keep_prob = tf.placeholder(tf.float32)


    
    np.random.shuffle(new_data)


    
    x = tf.placeholder(tf.float32, [None, INPUT_NODE],name='x')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])

    w1, b1, y1, w2, b2,y = forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    #损失函数loss含正则化regularization
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
    #指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        len(new_data) / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)



    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session(config = config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            #每次读入BATCH_SIZE组数据和标签

            if(step_train > len(new_data)):
                step_train = 0

                continue
            else:
                xs = new_data[step_train:step_train+BATCH_SIZE,:2304]

                ys = new_data[step_train:step_train+BATCH_SIZE,2304:]
            step_train = step_train + BATCH_SIZE


            _, loss_value, step, accuracy_train = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: xs, y_: ys, keep_prob:1})

            if i % 10 == 0:

                print("%s after %d training step(s), loss on training batch is %.4f, learning rate is %f, accuracy is %f." % (datetime.now(),step, loss_value,learning_rate.eval(),accuracy_train))
                #print("w1 = ", (sess.run(w1)))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    print("finish！")


def test(new_data):
    

    np.random.shuffle(new_data)
    new_data = new_data[0:1500]
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
        keep_prob = tf.placeholder(tf.float32)
        w1, b1, y1, w2, b2, y = forward(x, None)

        xs = new_data[:, :2304]

        ys = new_data[:, 2304:]

        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session(config=config) as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys, keep_prob : 0.5})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)




def main():
    
    train=True
    tf.device('/gpu:2')
    if train == True: 
        print("train!")
        data = pd.read_csv("fer2013.csv",low_memory=False).values
        data = data.astype(np.float32)  # change to numpy array and float32
        data = data / 255.0
        
        print("finish fer2013!")
        
        print("数组元素总数：",data.size)      #打印数组尺寸，即数组元素总数  
        print("数组形状：",data.shape)         #打印数组形状 
        print("数组的维度数目",data.ndim)      #打印数组的维度数目
        
        backward(data)
        
    else:
        print("test!")
        data = pd.read_csv("test.csv",low_memory=False).values
        data = data.astype(np.float32)  # change to numpy array and float32
        data = data / 255.0
        print("finish test!")

        print("数组元素总数：",data.size)      #打印数组尺寸，即数组元素总数  
        print("数组形状：",data.shape)         #打印数组形状 
        print("数组的维度数目",data.ndim)      #打印数组的维度数目
        
        test(data)

if __name__ == '__main__':

    main()

    
