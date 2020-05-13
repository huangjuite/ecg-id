# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import argparse
import random
import sys
import math
from tqdm import tqdm

input_type = "_I"


##RNN define
def lstm_cell(hidden_units):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)
    # lstm_cell = tf.contrib.rnn.GRUCell(hidden_units)
    return lstm_cell


def RNN(X, weights, biases, inputs, steps, hidden_units):
    X = tf.reshape(X, [-1, inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']  # matmul = matrix multiple
    X_in = tf.reshape(X_in, [-1, steps, hidden_units])

    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(hidden_units) for _ in range(num_layers)])  # num_layers=幾層的LSTM

    outputs, states = tf.nn.dynamic_rnn(cell, X_in, time_major=False, dtype=tf.float32)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # n steps, batch size, ouput size
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


##DNN define (FNN)
def add_layer(inputs, in_size, out_size, activation_function=None):
    DNN_Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    DNN_biases = tf.Variable(tf.zeros([1, out_size]))

    Wx_plus_b = tf.matmul(inputs, DNN_Weights) + DNN_biases;
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


def crossvalid(inputs, output, size, number, nth):
    train_size = math.floor(size / number) * (number - 1)
    valid_size = math.floor(size / number)
    index = np.arange(size)

    trainx = np.delete(inputs, index[(nth - 1) * valid_size:nth * valid_size], 0)
    trainy = np.delete(output, index[(nth - 1) * valid_size:nth * valid_size], 0)

    validx = inputs[(nth - 1) * valid_size:nth * valid_size]
    validy = output[(nth - 1) * valid_size:nth * valid_size]
    return trainx, trainy, validx, validy, train_size, valid_size


if __name__ == "__main__":  # main function
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, required=True, help="Determin to train or test now")
    parser.add_argument('--m', default=True, required=True, help="DNN or RNN")
    parser.add_argument('--n', default=True, required=True, help="nth crossvalid for rnn")
    parser.add_argument('--p', default=True, required=True, help="person")  # False
    parser.add_argument('--ot', default=True, required=True, help="output type")
    args = parser.parse_args()
    person = args.p
    output_type = args.ot

if args.m == "RNN":
    file = "./ECG_Data/" + person + output_type + input_type + "_" + args.m + ".mat"
    datax = scio.loadmat(file)['input']
    datay = scio.loadmat(file)['output']
    # testx = scio.loadmat(file)['testinput']
    # testy = scio.loadmat(file)['testoutput']
    size = scio.loadmat(file)['L'][0, 0]
    # test_size = scio.loadmat(file)['test_size'][0,0]
    windowsize = scio.loadmat(file)['windowsize'][0, 0]
    dim_input = scio.loadmat(file)['dim_input'][0, 0]
    times = scio.loadmat(file)['times'][0, 0]
    # parameter setting
    lr = 0.001  # learning rate
    num_layers = 2
    batch_size = 128

    ncrossvalid = 5
    rnn_inputs = dim_input
    rnn_steps = times
    rnn_hidden_units = 30
    nepochs = 300

    # learning curve
    lr_curve_path = "./ECG_Data/" + person + "/"
    model_path = "./ECG_Data/" + person + "/_" + args.m + "/" + output_type + "/"
    train_lr_curve = np.zeros([nepochs, 1])
    valid_lr_curve = np.zeros([nepochs, 1])
    test_lr_curve = np.zeros([nepochs, 1])

    # define rnn weights
    rnn_weights = {
        # rnn_inputs,rnn_hidden_units
        'in': tf.Variable(tf.random_normal([rnn_inputs, rnn_hidden_units])),
        # rnn_hidden_units, DNN_inputs
        'out': tf.Variable(tf.random_normal([rnn_hidden_units, 1]))
    }
    rnn_biases = {
        # (128,)
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_hidden_units, ])),
        # (10,)
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }
    # for k in range(ncrossvalid):	
    # build model
    xs = [];
    ys = [];
    rnn_out = [];
    layer_out = [];
    loss = [];
    train_step = [];
    temp_min_test_loss = 100
    xs = tf.placeholder(tf.float32, [None, rnn_steps, rnn_inputs])
    ys = tf.placeholder(tf.float32, [None, 1]);

    rnn_out = RNN(xs, rnn_weights, rnn_biases, rnn_inputs, rnn_steps, rnn_hidden_units)
    layer_out = rnn_out
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - layer_out), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # random from matlab
        crossloss = np.zeros([1, 1])
        trainx, trainy, testx, testy, train_size, valid_size = crossvalid(datax, datay, size, ncrossvalid,
                                                                          int(args.n))  # 第n次交叉驗正
        index = np.arange(trainx.shape[0])
        for j in range(nepochs):
            # np.random.shuffle(index)
            print("Epoch : %d" % (j + 1))
            for i in tqdm(range(0, trainx.shape[0], batch_size), ncols=60):
                sess.run(train_step,
                         feed_dict={xs: trainx[index[i:i + batch_size]], ys: trainy[index[i:i + batch_size]]})
            train_loss = sess.run(loss, feed_dict={xs: trainx, ys: trainy})
            train_lr_curve[j] = train_loss
            print("Train loss = " + str(train_loss))
            # vali_loss = sess.run(loss, feed_dict={xs: validx, ys: validy})
            # valid_lr_curve[j] = vali_loss
            # print("Valid loss = " + str(vali_loss))
            # _grad = sess.run(gvs, feed_dict={xs: trainx, ys: trainy})
            testloss = sess.run(loss, feed_dict={xs: testx, ys: testy})
            test_lr_curve[j] = testloss
            print("Test loss = " + str(testloss))
        # if temp_min_test_loss>testloss:
        #	temp_min_test_loss = testloss
        #	tf.train.Saver().save(sess, model_path)

        crossloss[0] = sess.run(loss, feed_dict={xs: testx, ys: testy})
        estimation = sess.run(layer_out, feed_dict={xs: testx})

    min_test_loss = np.zeros([1, 1])
    min_test_epoch = np.zeros([1, 1])
    min_test_loss[0] = np.sqrt(np.amin(test_lr_curve))
    min_test_epoch[0] = np.argmin(test_lr_curve)

    print(str(min_test_epoch))
    print(str(min_test_loss))
    # print(testloss)
    results = {'train_lr_curve': train_lr_curve.tolist(), 'test_lr_curve': test_lr_curve.tolist(),
               'min_test_loss': min_test_loss.tolist(), 'estimation': estimation.tolist()}
    if num_layers == 2:
        scio.savemat(lr_curve_path + "1Time_Results_" + args.n + "_" + output_type + input_type + "_" + args.m, results)
    else:
        scio.savemat(lr_curve_path + "1layer_Results_" + args.n + "_" + output_type + input_type + "_" + args.m, results)
else:
    print("model input error")







