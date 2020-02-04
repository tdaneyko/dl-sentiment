# Author: Thora Daneyko, 3822667
# Honor Code:  I pledge that this program represents my own work.

import tensorflow as tf

class DefaultConfig:

    n_epochs = 14
    batch_size = 64
    hidden_sizes = [32]

    char_embed_size = 16
    word_hidden_size = 32
    sent_hidden_size = 64
    pool_size = 1

    start_lr = 0.0005
    lr_decay_rate = 1.0

    input_dropout = 1.0
    hidden_dropout = 0.5

    def activation(self, tensor):
        return tf.tanh(tensor)

    def optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)