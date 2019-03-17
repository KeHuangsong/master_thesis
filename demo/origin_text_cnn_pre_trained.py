import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max_pooling and softmax layer.
    """
    def __init__(self, batch_size, num_classes, embedding_size, w2v, l2_reg_lambda=0.0):
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

        pooled_outputs = []

        # deal with every sentences
        for i in range(batch_size):
            
            # Embedding layer
            bs = tf.reshape(BS[i], [1, -1])
            embedded_chars = tf.nn.embedding_lookup(self.W, bs)
            embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            t_pools = []

            for filter_size in [2, 3, 5]:
                filter_shape1 = [filter_size, embedding_size, 1, 20]
                w1 = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="w1")
                b1 = tf.Variable(tf.constant(0.1, shape=[20]), name="b1")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    w1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b1), name="relu")
                # Maxpooling over the outputs
                pooled = tf.reduce_max(h, reduction_indices=[0, 1])
                t_pools.append(pooled)
            pooled_outputs.append(tf.concat(t_pools, 1))

        h_pool_flat = tf.concat(pooled_outputs, 0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        w = tf.get_variable(
            "w",
            shape=[60, num_classes],
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
