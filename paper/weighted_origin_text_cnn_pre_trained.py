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
        position_weight_split = 10
        self.position_weight = tf.Variable(tf.random_uniform([position_weight_split, 1], 1.0, 1.0),
                                           name="position_weight")
        l2_loss += tf.nn.l2_loss(self.position_weight-tf.ones([position_weight_split, 1]))
        self.word_weight = tf.Variable(tf.random_uniform([list(w2v.shape)[0], 1], 1.0, 1.0), name="position_weight")
        l2_loss += tf.nn.l2_loss(self.position_weight - tf.ones([position_weight_split, 1]))

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
        num_filters = 100
        divider = tf.expand_dims(self.len_x, 1)
        divider = tf.expand_dims(divider, 2)
        divider = tf.concat([divider for _ in range(num_filters)], 2)
        filter_sizes = [2, 3, 4, 5]
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
                pooled = tf.reduce_max(h, [1])
                max_condition = tf.argmax(h, 1)
                split_max_condition = tf.unstack(max_condition, axis=0)
                word_weight = []
                for i in range(batch_size):
                    pool_max_conditions = tf.concat([split_max_condition[i]+i for i in range(filter_size)], 0)
                    pool_max_conditions = tf.clip_by_value(tf.cast(pool_max_conditions, tf.int32), 0,
                                                           tf.slice(self.len_x, [i], [1]) - 1)
                    max_word = tf.nn.embedding_lookup(BS[i], pool_max_conditions)
                    batch_word_w = tf.reduce_mean(tf.nn.embedding_lookup(self.word_weight, max_word), [0, 2, 3])
                    word_weight.append(tf.expand_dims(tf.expand_dims(batch_word_w, 0), 0))
                self.word_w = tf.concat(word_weight, 0)
                self.max_condition = tf.rint(tf.cast(max_condition, tf.float32)/tf.cast(divider, tf.float32) * position_weight_split)
                self.max_condition = tf.cast(tf.clip_by_value(self.max_condition, 0, position_weight_split-1), tf.int32)
                self.position_w = tf.reduce_max(tf.nn.embedding_lookup(self.position_weight, self.max_condition), 3)
                self.weight = tf.multiply(self.position_w, self.word_w)
                print self.position_w
                print self.word_w
                print self.weight
                pooled_outputs.append(tf.multiply(pooled, self.weight))

        self.h_concat = tf.concat(pooled_outputs, 2)
        self.h_pool_flat = tf.reshape(self.h_concat, [-1, num_filters * len(filter_sizes)])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            w = tf.get_variable(
                "w",
                shape=[num_filters * len(filter_sizes), num_classes],
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
