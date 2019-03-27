import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max_pooling and softmax layer.
    """
    def __init__(self, batch_size, num_classes, embedding_size, resize_n, w2v, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
        self.len_x = tf.placeholder(tf.int32, [batch_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        BS = tf.split(self.input_x, self.len_x, num=batch_size)

        self.W = tf.Variable(w2v, name="W")
        # self.W = tf.constant(w2v, name="W")
        # self.W = tf.Variable(tf.random_uniform(list(w2v.shape), -1.0, 1.0), name="W")

        resized_images = []
        for i in range(batch_size):
            with tf.name_scope("embedding-%s" % i):
                bs = tf.reshape(BS[i], [1, -1])
                embedded_chars = tf.nn.embedding_lookup(self.W, bs)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
                pad_dim = tf.reduce_max(self.len_x) - tf.reduce_max(tf.slice(self.len_x, [i], [1]))
                zeros = tf.zeros([1, pad_dim, embedding_size, 1], dtype=tf.float32)
                resized_images.append(tf.concat([tf.cast(embedded_chars_expanded, tf.float32), zeros], 1))
        resized_embedded_chars_expanded = tf.concat(resized_images, 0)

        pooled_outputs = []
        argmax = []
        num_filters = 50
        divider = tf.expand_dims(self.len_x, 1)
        divider = tf.expand_dims(divider, 2)
        divider = tf.concat([divider for _ in range(num_filters)], 2)
        filter_sizes = [2, 3, 4]
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    resized_embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.pooled = tf.reduce_max(h, [1])
                self.max_condition = tf.cast(tf.argmax(h, 1), tf.float32)/tf.cast(divider, tf.float32)
                pooled_outputs.append(self.pooled)
                argmax.append(self.max_condition)

        with tf.name_scope("max_value"):
            self.h_concat = tf.concat(pooled_outputs, 2)

            self.h_pool_flat = tf.reshape(self.h_concat, [-1, num_filters * len(filter_sizes)])
            self.h_pool_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            w = tf.get_variable(
                "w",
                shape=[num_filters * len(filter_sizes), num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)

            self.pool_scores = tf.nn.xw_plus_b(self.h_pool_drop, w, b, name="scores")

        with tf.name_scope("argmax"):
            self.h_argmax = tf.concat(argmax, 2)
            batch_mean, batch_var = tf.nn.moments(self.h_argmax, [0, 1], keep_dims=True)
            shift = tf.Variable(tf.zeros([1]))
            scale = tf.Variable(tf.ones([1]))
            epsilon = 1e-3
            self.h_argmax = tf.nn.batch_normalization(self.h_argmax, batch_mean, batch_var,
                                                      shift, scale, epsilon)
            self.h_argmax_flat = tf.reshape(self.h_argmax, [-1, num_filters * len(filter_sizes)])
            self.h_argmax_drop = tf.nn.dropout(self.h_argmax_flat, self.dropout_keep_prob)
            w1 = tf.get_variable(
                "w1",
                shape=[num_filters * len(filter_sizes), num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b1")

            l2_loss += tf.nn.l2_loss(w1)
            l2_loss += tf.nn.l2_loss(b1)

            self.argmax_scores = tf.nn.xw_plus_b(self.h_argmax_drop, w1, b1, name="scores")

        if 0:
            'concat'
            with tf.name_scope("output"):
                self.bind_score = tf.concat([self.pool_scores, self.argmax_scores], 1)
                self.w2 = tf.get_variable(
                    "w2",
                    shape=[2*num_classes, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")

                l2_loss += tf.nn.l2_loss(self.w2)
                l2_loss += tf.nn.l2_loss(self.b2)

                self.scores = tf.nn.xw_plus_b(self.bind_score, self.w2, self.b2, name="scores")
                # self.scores = self.pool_scores
        if 1:
            'multiply'
            with tf.name_scope("output"):
                self.bind_score = tf.multiply(self.pool_scores, self.argmax_scores)
                self.w2 = tf.get_variable(
                    "w2",
                    shape=[num_classes, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")

                l2_loss += tf.nn.l2_loss(self.w2)
                l2_loss += tf.nn.l2_loss(self.b2)

                self.scores = tf.nn.xw_plus_b(self.bind_score, self.w2, self.b2, name="scores")
                # self.scores = self.pool_scores

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
