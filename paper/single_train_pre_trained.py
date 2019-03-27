#! /usr/bin/env python

import os
import argparse
import gensim
import tensorflow as tf
import numpy as np
from data_helpers import DataHelper
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ModelTrain(DataHelper):
    def __init__(self, dev_sample_percentage, data_file, dropout_keep_prob,
                 batch_size, num_epochs, evaluate_every, checkpoint_every, num_checkpoints, allow_soft_placement,
                 log_device_placement, resize_n, l2_reg_lambda):
        self.dev_sample_percentage = dev_sample_percentage
        self.data_file = data_file
        self.resize_n = resize_n
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.num_checkpoints = num_checkpoints
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement

    def get_data(self):
        print("Loading data...")
        if not os.path.exists(self.data_file):
            print("start get data from hive...")
            self.get_title_data(days=15, data_size=1000, category_num=20)

        train_data_path = './data/train_data.npy'
        if os.path.exists(train_data_path):
            train_data = np.load(train_data_path).tolist()
            self.default_w2v = train_data['default_w2v']
            self.x_train = train_data['x_train']
            self.x_dev = train_data['x_dev']
            self.y_train = train_data['y_train']
            self.y_dev = train_data['y_dev']
            self.len_train = train_data['len_train']
            self.len_dev = train_data['len_dev']
        else:
            x_text, y = self.load_data_and_labels(self.data_file)
            vocabulary = {}
            for text in x_text:
                for word in text:
                    if word not in vocabulary:
                        vocabulary[word] = len(vocabulary)
            vec_path = './data/pre_trained_model.vec'
            if not os.path.exists(vec_path):
                hdfs_path = '/user/kehuangsong/pre_trained_vec/pre_trained_model.vec'
                os.system('hdfs dfs -copyToLocal %s %s' % (hdfs_path, vec_path))
            default_w2v = [[]] * len(vocabulary)
            model = gensim.models.KeyedVectors.load_word2vec_format(vec_path)
            for word in vocabulary:
                index = vocabulary[word]
                if not default_w2v[index]:
                    try:
                        default_w2v[index] = list(model[word.decode('utf-8')])
                    except:
                        default_w2v[index] = list(np.random.uniform(-0.65, 0.65, 100))
            x = []
            for text in x_text:
                x.append([vocabulary[n] for n in text])
            x = np.array(x)
            y = np.array(y)
            self.default_w2v = np.array(default_w2v)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            dev_sample_index = -1 * int(self.dev_sample_percentage * float(len(y)))
            self.x_train, self.x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
            self.y_train, self.y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
            self.len_train = [len(v) for v in self.x_train]
            self.len_dev = [len(v) for v in self.x_dev]
            train_data = dict(default_w2v=self.default_w2v,
                              x_train=self.x_train,
                              x_dev=self.x_dev,
                              y_train=self.y_train,
                              y_dev=self.y_dev,
                              len_train=self.len_train,
                              len_dev=self.len_dev)
            np.save(train_data_path, train_data)
            print("Vocabulary Size: {:d}".format(len(vocabulary)))
        print("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))

    def train_step(self, sess, cnn, lr, train_op, x_batch, y_batch, len_batch, lrate):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.len_x: len_batch,
            cnn.dropout_keep_prob: self.dropout_keep_prob,
            lr: lrate
        }
        _, loss, accuracy = sess.run(
            [train_op, cnn.loss, cnn.accuracy],
            feed_dict)

    def dev_step(self, cnn, sess, x_batch, y_batch, len_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.len_x: len_batch,
            cnn.dropout_keep_prob: 1.0
        }
        loss, accuracy = sess.run(
            [cnn.loss, cnn.accuracy],
            feed_dict)

        return (loss, accuracy)

    def my_plot(self, loss_list1, loss_list2):
        plt.subplot(1, 1, 1)
        plt.cla()
        plt.plot(loss_list1)
        plt.plot(loss_list2)
        plt.draw()
        plt.pause(0.0001)

    def train(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                    allow_soft_placement=self.allow_soft_placement,
                    log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    batch_size=self.batch_size,
                    num_classes=self.y_train.shape[1],
                    embedding_size=self.default_w2v.shape[1],
                    resize_n=self.resize_n,
                    w2v=self.default_w2v,
                    l2_reg_lambda=self.l2_reg_lambda)

                lr = tf.placeholder(tf.float32, [])

                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(lr)  # AdagradOptimizer, AdamOptimizer, RMSPropOptimizer
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                sess.run(tf.global_variables_initializer())

                batches = self.batch_iter(
                    self.x_train, self.y_train, self.len_train, self.batch_size, self.num_epochs)

                i = 0
                plt.ion()
                plt.figure()
                plt.show()
                train_accu = []
                dev_accu = []
                turn = 0
                for x_batch, y_batch, len_batch in batches:
                    _lr = 0.001#+(0.001-0.0001)*np.exp(-i/2000)
                    self.train_step(sess, cnn, lr, train_op, x_batch, y_batch, len_batch, _lr)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.evaluate_every == 0:
                        if 1:
                            feed_dict = {
                                cnn.input_x: x_batch,
                                cnn.input_y: y_batch,
                                cnn.len_x: len_batch,
                                cnn.dropout_keep_prob: 1.0
                            }
                            argmax, position_weight, position_w = sess.run(
                                [cnn.max_condition, cnn.position_weight, cnn.position_w],
                                feed_dict)
                            print [list(v) for v in argmax[:1]]
                            print [v[0] for v in position_weight]
                            print [list(v) for v in position_w[:1]]
                        dev_batches = self.batch_iter(
                                  self.x_dev, self.y_dev, self.len_dev, self.batch_size, 1)
                        train_batches = self.batch_iter(
                                  self.x_train, self.y_train, self.len_train, self.batch_size, 1)
                        dev_pre = []
                        train_pre = []
                        turn += 1
                        print("\nEvaluation: %s" % turn)
                        for x_dev_batch, y_dev_batch, len_dev_batch in dev_batches:
                            dev_loss, dev_accuracy = self.dev_step(cnn, sess, x_dev_batch, y_dev_batch, len_dev_batch)
                            dev_pre.append(dev_accuracy)
                        for x_train_batch, y_train_batch, len_train_batch in train_batches:
                            train_loss, train_accuracy = self.dev_step(cnn, sess, x_train_batch, y_train_batch,
                                                                       len_train_batch)
                            train_pre.append(train_accuracy)
                        print('train accuray: '+str(sum(train_pre)/len(train_pre)))
                        print('test accuray: '+str(sum(dev_pre)/len(dev_pre)))
                        dev_accu.append(sum(dev_pre)/len(dev_pre))
                        train_accu.append(sum(train_pre)/len(train_pre))
                        self.my_plot(train_accu, dev_accu)
                plt.ioff()
                plt.show()

    def start(self):
        self.get_data()
        self.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-model', type=str, default='origin')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5)
    parser.add_argument('-l2_reg_lambda', type=float, default=0)
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-refresh_data', type=bool, default=False)
    args = parser.parse_args()
    if args.model == 'origin':
        from origin_text_cnn_pre_trained import TextCNN
    if args.model == 'weighted_origin':
        from weighted_origin_text_cnn_pre_trained import TextCNN
    if args.model == 'origin_pro':
        from origin_condition_text_cnn_pre_trained import TextCNN
    elif args.model == 'resized':
        from resized_text_cnn_pre_trained import TextCNN
    elif args.model == 'both':
        from both_text_cnn_pre_trained import TextCNN
    elif args.model == 'ssp':
        from ssp_text_cnn_pre_trained import TextCNN
    if args.refresh_data:
        os.system('rm -r ./data')
    if not os.path.exists('./data'):
        os.system('mkdir ./data')
    print args
    data_file = "./data/raw_title_data.txt"
    dev_sample_percentage = 0.1
    resize_n = 60
    dropout_keep_prob = args.dropout_keep_prob
    l2_reg_lambda = args.l2_reg_lambda
    batch_size = args.batch_size
    num_epochs = 200
    evaluate_every = 100
    checkpoint_every = 100
    num_checkpoints = 5
    allow_soft_placement = True
    log_device_placement = False
    model_train = ModelTrain(dev_sample_percentage, data_file, dropout_keep_prob,
                 batch_size, num_epochs, evaluate_every, checkpoint_every, num_checkpoints, allow_soft_placement,
                 log_device_placement, resize_n, l2_reg_lambda)
    model_train.start()

