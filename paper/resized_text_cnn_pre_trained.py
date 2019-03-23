import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a resized, convolutional, pooling and softmax layer.
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

        resized_images = []
        for i in range(batch_size):
            with tf.name_scope("embedding-%s" % i):
                bs = tf.reshape(BS[i], [1, -1])
                embedded_chars = tf.nn.embedding_lookup(self.W, bs)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
                em = tf.image.resize_images(embedded_chars_expanded,
                                            [resize_n, embedding_size], method=tf.image.ResizeMethod.BILINEAR)
                # BILINEAR  NEAREST_NEIGHBOR  BICUBIC  AREA
                resized_images.append(em)
        resized_embedded_chars_expanded = tf.concat(resized_images, 0)

        '''
        batch_mean, batch_var = tf.nn.moments(self.resized_embedded_chars_expanded, [0, 1, 2], keep_dims=True)
        shift = tf.Variable(tf.zeros([1]))
        scale = tf.Variable(tf.ones([1]))
        epsilon = 1e-3
        self.resized_embedded_chars_expanded = tf.nn.batch_normalization(self.resized_embedded_chars_expanded,
                                             batch_mean, batch_var, shift, scale, epsilon)
        '''

        pooled_outputs = []
        num_filters = 50
        filter_sizes = [5, 10, 25]
        for filter_size in filter_sizes:
            with tf.name_scope("conv-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                conv = tf.nn.conv2d(
                    resized_embedded_chars_expanded,
                    w,
                    strides=[1, filter_size, 1, 1],
                    padding="VALID",
                    name="conv")
                print conv
                pooled_outputs.append(tf.reshape(conv, [batch_size, -1]))

        self.h_concat = tf.concat(pooled_outputs, 1)

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_concat, self.dropout_keep_prob)

        out_len = 0
        for filter_size in filter_sizes:
            out_len += resize_n / filter_size
        out_len = out_len * num_filters

        with tf.name_scope("output"):
            w = tf.get_variable(
                "w",
                shape=[out_len, num_classes],
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
