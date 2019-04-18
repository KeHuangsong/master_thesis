import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN model for text classification.
    Uses an embedding layer, followed by a convolutional, max_pooling and softmax layer.
    """
    def __init__(self, batch_size, num_classes, embedding_size, w2v, l2_reg_lambda=0.0, sentence_w=0.5, w_f_len=100):
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
        l2_loss += tf.nn.l2_loss(self.position_weight - tf.ones([position_weight_split, 1]))
        self.word_weight = tf.Variable(tf.random_uniform([list(w2v.shape)[0], 1], 1.0, 1.0), name="position_weight")
        l2_loss += tf.nn.l2_loss(self.word_weight - tf.ones([list(w2v.shape)[0], 1]))

        sentence_weight = []
        resized_images = []
        for i in range(batch_size):
            with tf.name_scope("embedding-%s" % i):
                sentence_len = tf.reduce_max(tf.slice(self.len_x, [i], [1]))
                bs = tf.reshape(BS[i], [1, -1])
                embedded_chars = tf.nn.embedding_lookup(self.W, bs)
                word_weight = tf.nn.embedding_lookup(self.word_weight, bs)
                top_k = tf.nn.top_k(tf.reshape(word_weight, [1, -1]), k=sentence_len).values
                slice_len = tf.cond(tf.less(sentence_len, w_f_len), lambda: sentence_len, lambda: w_f_len)
                top_k_mean = tf.reduce_mean(tf.slice(top_k, [0, 0], [1, slice_len]), [1])
                sentence_weight.append(top_k_mean)
                self.word_w = tf.concat([word_weight for _ in range(embedding_size)], 2)
                seq = tf.range(start=0, limit=tf.cast(sentence_len, tf.float32), delta=1, dtype=tf.float32)
                word_position = tf.cast(tf.clip_by_value(tf.rint(seq/tf.cast(sentence_len, tf.float32)
                                                                      * position_weight_split), 0, 9), tf.int32)
                position_weight = tf.nn.embedding_lookup(self.position_weight, word_position)
                self.position_w = tf.expand_dims(tf.concat([position_weight for _ in range(embedding_size)], 1), 0)
                self.weight = tf.multiply(self.word_w, self.position_w)
                embedded_chars_expanded = tf.expand_dims(tf.multiply(tf.cast(embedded_chars, tf.float32), self.weight), -1)
                pad_dim = tf.reduce_max(self.len_x) - sentence_len
                zeros = tf.zeros([1, pad_dim, embedding_size, 1], dtype=tf.float32)
                resized_images.append(tf.concat([embedded_chars_expanded, zeros], 1))
        resized_embedded_chars_expanded = tf.concat(resized_images, 0)

        pooled_outputs = []
        num_filters = 100
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
                pooled_outputs.append(pooled)

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
            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            sentence_weight = tf.pow(tf.concat(sentence_weight, 0), sentence_w)
            sentence_weight = tf.stop_gradient(sentence_weight)
            self.loss = tf.reduce_mean(tf.multiply(losses, sentence_weight)) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
