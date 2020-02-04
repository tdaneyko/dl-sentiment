# Author: Thora Daneyko, 3822667
# Honor Code:  I pledge that this program represents my own work.

from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn

class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2

class Model:
    def __init__(
            self,
            config,
            labels,
            word_ids,
            char_vecs,
            feature_vecs,
            sent_lens,
            word_lens,
            n_chars,
            embeddings,
            phase=Phase.Predict):
        # Get shapes
        dict_size = embeddings.shape[0]
        embed_size = embeddings.shape[1]
        n_of_batches = word_ids.shape[0]
        batch_size = word_ids.shape[1]
        input_size = word_ids.shape[2]
        word_size = char_vecs.shape[3]
        feat_size = feature_vecs.shape[3]
        label_size = labels.shape[2]

        # Word embedding lookup table
        self._embeddings = tf.placeholder(tf.float32, shape=[dict_size, embed_size])

        # The label distribution
        if phase != Phase.Predict:
            self._y = tf.placeholder(tf.int32, shape=[batch_size, label_size])

        # Placeholders for the data
        self._word_ids = tf.placeholder(tf.int32, shape=[batch_size, input_size])
        self._char_vecs = tf.placeholder(tf.int32, shape=[batch_size, input_size, word_size])
        self._feat_vecs = tf.placeholder(tf.float32, shape=[batch_size, input_size, feat_size])
        self._sent_lens = tf.placeholder(tf.int32, shape=[batch_size])
        self._word_lens = tf.placeholder(tf.int32, shape=[batch_size, input_size])

        # Create Tensorflow lookups for char and word embeddings
        # https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
        char_lookup = tf.get_variable(name="char_lookup", shape=[n_chars, config.char_embed_size])
        word_embed = tf.nn.embedding_lookup(self._embeddings, self._word_ids)
        char_embed = tf.nn.embedding_lookup(char_lookup, self._char_vecs)

        # Get single char-level vector for each word
        char_embed = tf.reshape(char_embed, [batch_size * input_size, word_size, -1]) # reshape to list of char-level word representations
        word_lens = tf.reshape(self._word_lens, [-1])
        char_embed = self.get_sequence_representation(char_embed, word_lens, config, config.word_hidden_size, "word")
        char_embed = tf.reshape(char_embed, [batch_size, input_size, -1]) # group back into tweets

        # Concat embeddings and feature vectors for each word
        word_vecs = tf.concat([word_embed, char_embed, self._feat_vecs], axis=2)

        # Apply input dropout during training
        if phase == Phase.Train:
            word_vecs = tf.nn.dropout(word_vecs, config.input_dropout)

        # Apply RNN to get single vector for each Tweet
        hidden = self.get_sequence_representation(word_vecs, self._sent_lens, config, config.sent_hidden_size, "sent")

        # Apply hidden layer(s)
        for (i, hidden_size) in enumerate(config.hidden_sizes):
            if phase == Phase.Train:
                hidden = tf.nn.dropout(hidden, config.hidden_dropout)
            W = tf.get_variable("W_hidden_%d" % i, shape=[hidden.shape[1], hidden_size])
            b = tf.get_variable("b_hidden_%d" % i, shape=[hidden_size])
            hidden = config.activation(tf.matmul(hidden, W) + b)

        # Final linear transformation
        w = tf.get_variable("w", shape=[hidden.shape[1], label_size])
        b = tf.get_variable("b", shape=[label_size])

        # Get logits
        logits = tf.matmul(hidden, w) + b
        logits = tf.reshape(logits, [batch_size, label_size])

        # Compute losses
        if phase == Phase.Train or Phase.Validation:
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._y, logits=logits)
            self._loss = loss = tf.reduce_sum(losses)

        # Train parameters
        if phase == Phase.Train:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(config.start_lr, global_step, n_of_batches,
                                                       config.lr_decay_rate)
            self._train_op = config.optimizer(learning_rate) \
                .minimize(losses, global_step=global_step)
            self._probs = probs = tf.nn.softmax(logits)

        # Get label predictions
        if phase == Phase.Validation:
            self._labels = (tf.argmax(self.y, axis=1), tf.argmax(logits, axis=1))

    """
    Applies a bidirectional RNN and max pooling to the input data
    to get a sequence representation.
    """
    def get_sequence_representation(self, seqs, lens, config, hidden_size, name):
        with tf.variable_scope(name):
            batch_size = seqs.shape.as_list()[0]

            # RNN
            fw_cell = rnn.GRUCell(hidden_size * config.pool_size, activation=config.activation)
            bw_cell = rnn.GRUCell(hidden_size * config.pool_size, activation=config.activation)
            _, bi_hidden = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, seqs, sequence_length=lens,
                                                           dtype=tf.float32)
            bi_hidden = (tf.expand_dims(bi_hidden[0], -1), tf.expand_dims(bi_hidden[1], -1))
            seqs = tf.concat(bi_hidden, axis=2)
            seqs = tf.expand_dims(seqs, -1)

            # Max pooling
            pool_shape = [1, config.pool_size, seqs.shape[2], 1]
            pool = tf.nn.max_pool(seqs, ksize=pool_shape, strides=pool_shape, padding="SAME")
            seqs = tf.reshape(pool, [batch_size, hidden_size])

            return seqs


    @property
    def embeddings(self):
        return self._embeddings

    @property
    def word_ids(self):
        return self._word_ids

    @property
    def char_vecs(self):
        return self._char_vecs

    @property
    def feat_vecs(self):
        return self._feat_vecs

    @property
    def y(self):
        return self._y

    @property
    def labels(self):
        return self._labels

    @property
    def sent_lens(self):
        return self._sent_lens

    @property
    def word_lens(self):
        return self._word_lens

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op