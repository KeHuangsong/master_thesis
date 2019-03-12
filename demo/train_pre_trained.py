#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import DataHelper
from text_cnn_pre_trained import TextCNN
from tensorflow.contrib import learn
import matplotlib.pyplot as plt


class ModelTrain(DataHelper, TextCNN):
    def __init__(self):
        self.dev_sample_percentage = 0.1
        self.positive_data_file = "./data/data.pos"
        self.negative_data_file = "./data/data.neg"
        self.resize_n = 100
        self.dropout_keep_prob = 0.01
        self.l2_reg_lambda = 5
        self.batch_size = 50
        self.num_epochs = 200
        self.evaluate_every = 100
        self.checkpoint_every = 100
        self.num_checkpoints = 5
        self.allow_soft_placement = True
        self.log_device_placement = False

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

vocabulary = {}
for x in x_text:
    w = x.split(" ")
    for v in w:
        if v not in vocabulary:
            vocabulary[v] = len(vocabulary)

W2V = [0] * len(vocabulary)

'''
with open('../glove1.txt') as f:
    line = f.readline().rstrip().split(' ')
    while len(line)>1:
         W2V[vocabulary[line[0]]] = [float(v) for v in line[1:]]
         line = f.readline().rstrip().split(' ')
'''

for i in xrange(len(W2V)):
    if W2V[i] == 0:
        W2V[i] = list(np.random.uniform(-0.65, 0.65, 300))
'''
x = []
with open('doc_vocabulary.txt') as f:
    line = f.readline().strip()
    while len(line)>0:
        x.append([int(v)-1 for v in line.split(' ')])
        line = f.readline().strip()
'''
x = []
for xt in x_text:
    w = xt.split(" ")
    x.append([vocabulary[n] for n in w])# if n in set(W)])


x = np.array(x)
W2V = np.array(W2V)

# Randomly shuffle data
# np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
len_train = [len(v) for v in x_train]
len_dev = [len(v) for v in x_dev]
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

print(x_train[0])
print(x_dev[0])


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            batch_size=FLAGS.batch_size,
            num_classes=y_train.shape[1],
            embedding_size=W2V.shape[1],
            resize_n=FLAGS.resize_n,
            w2v=W2V,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        lr = tf.placeholder(tf.float32, [])

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(lr) #AdagradOptimizer  AdamOptimizer RMSPropOptimizer
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, len_batch, lrate):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.len_x: len_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              lr: lrate
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            #return(loss, accuracy)

        def dev_step(x_batch, y_batch, len_batch):
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
            return(loss, accuracy)
            
        def plot(loss_list1):#, loss_list2):
            plt.subplot(1, 1, 1)
            plt.cla()
            plt.plot(loss_list1)
            #plt.plot(loss_list2)
            plt.draw()
            plt.pause(0.0001)

        # Generate batches
        batches = data_helpers.batch_iter(
            x_train, y_train, len_train, FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        i = 0
        plt.ion()
        plt.figure()
        plt.show()
        train_accu = []
        dev_accu = []
        for x_batch, y_batch, len_batch in batches:
            _lr = 0.0001+(0.001-0.0001)*np.exp(-i/2000)
            train_step(x_batch, y_batch, len_batch, _lr)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                dev_batches = data_helpers.batch_iter(
                          x_dev, y_dev, len_dev, FLAGS.batch_size, 1)
                train_batches = data_helpers.batch_iter(
                          x_train, y_train, len_train, FLAGS.batch_size, 1)
                dev_pre = []
                train_pre = []
                print("\nEvaluation:")
                for x_dev_batch, y_dev_batch, len_dev_batch in dev_batches:
                    dev_loss, dev_accuracy = dev_step(x_dev_batch, y_dev_batch, len_dev_batch)
                    dev_pre.append(dev_accuracy)
                #for x_train_batch, y_train_batch, len_train_batch in train_batches:
                #    train_loss, train_accuracy = dev_step(x_train_batch, y_train_batch, len_train_batch)
                #    train_pre.append(train_accuracy)
                #print('train accuray: '+str(sum(train_pre)/len(train_pre))+'\n')
                print('test accuray: '+str(sum(dev_pre)/len(dev_pre))+'\n')
                dev_accu.append(sum(dev_pre)/len(dev_pre))
                #train_accu.append(sum(train_pre)/len(train_pre))
                #plot(train_accu, dev_accu)
                plot(dev_accu)
        plt.ioff()
        plt.show()
