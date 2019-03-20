import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, spatial_pyramid_pooling and softmax layer.
    """
    def __init__(self, batch_size, num_classes, embedding_size, resize_n, w2v, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
        self.len_x = tf.placeholder(tf.int32, [batch_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        BS = tf.split(self.input_x, self.len_x, num=batch_size)

        # self.W = tf.Variable(w2v, name="W")
        # self.W = tf.constant(w2v, name="W")
        self.W = tf.Variable(tf.random_uniform(list(w2v.shape), -1.0, 1.0), name="W")

        pooled_outputs = []
        num_filters = 50
        filter_sizes = [3, 4, 5]
        ssp_sizes = [1, 2, 3]
        for filter_size in filter_sizes:
            with tf.name_scope("ssp-%s" % filter_size):
                filter_pool = []
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                for i in range(batch_size):
                    pools = []
                    bs = tf.reshape(BS[i], [1, -1])
                    embedded_chars = tf.nn.embedding_lookup(self.W, bs)
                    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        w,
                        strides=[1, 1, embedding_size, 1],
                        padding="SAME",
                        name="conv")
                    conv = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    conv_flat = tf.reshape(conv, [-1, 1, num_filters])
                    text_len = tf.reduce_max(tf.slice(self.len_x, [i], [1]))  # - tf.constant(filter_size - 1)
                    for ssp_size in ssp_sizes:
                        for j in range(ssp_size):
                            start_point = tf.cast(tf.ceil(tf.cast(text_len/ssp_size * j, tf.float32)), tf.int32)
                            end_point = tf.cast(tf.ceil(tf.cast(text_len/ssp_size * (j + 1), tf.float32)), tf.int32)
                            local_area = tf.slice(conv_flat, [start_point, 0, 0], [end_point - start_point, -1, -1])
                            local_max = tf.reduce_max(local_area, 0)
                            token = tf.clip_by_value(local_max, 0, 2)
                            pools.append(token)
                    filter_pool.append(tf.concat(pools, 1))
            pooled_outputs.append(tf.concat(filter_pool, 0))

        self.h_concat = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = self.h_concat

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            w = tf.get_variable(
                "w",
                shape=[num_filters * sum(ssp_sizes) * len(filter_sizes), num_classes],
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)

            # self.scores = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop, w, b, name="scores"))
            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
