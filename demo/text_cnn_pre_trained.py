import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a resized, convolutional, pooling and softmax layer.
    """
    def __init__(self, batch_size, num_classes, embedding_size, resize_n, w2v, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
        self.len_x = tf.placeholder(tf.int32, [batch_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # split input
        BS = tf.split(self.input_x, self.len_x, num=batch_size)

        # Embedding vector
        self.W = tf.Variable(w2v, name="W")
        # self.W = tf.constant(w2v, name="W")
        # self.W = tf.Variable(tf.random_uniform(list(w2v.shape), -1.0, 1.0), name="W")

        # deal with every sentences
        for i in range(batch_size):
            
            # Embedding layer
            bs = tf.reshape(BS[i], [1, -1])
            self.embedded_chars = tf.nn.embedding_lookup(self.W, bs)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            if i == 0:
                self.resized_embedded_chars_expanded = tf.image.resize_images(self.embedded_chars_expanded,
                                         [resize_n, embedding_size], method=tf.image.ResizeMethod.BILINEAR)
            else:
                em = tf.image.resize_images(self.embedded_chars_expanded,
                                         [resize_n, embedding_size], method=tf.image.ResizeMethod.BILINEAR)
                # BILINEAR  NEAREST_NEIGHBOR  BICUBIC  AREA
                self.resized_embedded_chars_expanded = tf.concat(
                                         [self.resized_embedded_chars_expanded, em], 0)

        '''
        batch_mean, batch_var = tf.nn.moments(self.resized_embedded_chars_expanded, [0, 1, 2], keep_dims=True)
        shift = tf.Variable(tf.zeros([1]))
        scale = tf.Variable(tf.ones([1]))
        epsilon = 1e-3
        self.resized_embedded_chars_expanded = tf.nn.batch_normalization(self.resized_embedded_chars_expanded, 
                                             batch_mean, batch_var, shift, scale, epsilon)
        '''

        filter_shape1 = [5, 300, 1, 50]
        w1 = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="W1")

        conv1 = tf.nn.conv2d(
            self.resized_embedded_chars_expanded,
            w1,
            strides=[1, 5, 1, 1],
            padding="VALID",
            name="conv1")
        h1 = tf.reshape(conv1, [batch_size, -1])

        filter_shape2 = [10, 300, 1, 50]
        w2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W3")

        conv2 = tf.nn.conv2d(
            self.resized_embedded_chars_expanded,
            w2,
            strides=[1, 10, 1, 1],
            padding="VALID",
            name="conv2")
        h2 = tf.reshape(conv2, [batch_size, -1])

        filter_shape3 = [20, 300, 1, 50]
        w3 = tf.Variable(tf.truncated_normal(filter_shape3, stddev=0.1), name="W3")

        conv3 = tf.nn.conv2d(
            self.resized_embedded_chars_expanded,
            w3,
            strides=[1, 20, 1, 1],
            padding="VALID",
            name="conv3")
        h3 = tf.reshape(conv3, [batch_size, -1])

        h = tf.concat([h1, h2, h3], 1)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(h, self.dropout_keep_prob)

        w = tf.get_variable(
            "w",
            shape=[1750, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

        # Final (unnormalized) scores and predictions
        l2_loss += tf.nn.l2_loss(w)
        l2_loss += tf.nn.l2_loss(b)

        self.scores = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop, w, b, name="scores"))
        # self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
