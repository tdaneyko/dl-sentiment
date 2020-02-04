# Author: Thora Daneyko, 3822667
# Honor Code:  I pledge that this program represents my own work.

import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support as scores
from sklearn.metrics import f1_score

from config import DefaultConfig
from model import Model, Phase
from numberer import Numberer
from preprocessing import prepare_data, read_emolex

"""
Splits data into batches of a certain size. The remainder is discarded.
"""
def generate_batches(labels, word_ids, char_vecs, feature_vecs, sent_lens, word_lens, batch_size=128):
    # Get dimensions
    n_of_batches = word_ids.shape[0] // batch_size
    n_of_labels = labels.shape[1]
    sent_len = word_ids.shape[1]
    word_len = char_vecs.shape[2]
    feat_len = feature_vecs.shape[2]

    # Discard final incomplete batch
    used_data = n_of_batches * batch_size

    # Reshape data into batches
    label_batches = np.reshape(labels[:used_data], [n_of_batches, batch_size, n_of_labels])
    word_batches = np.reshape(word_ids[:used_data], [n_of_batches, batch_size, sent_len])
    char_batches = np.reshape(char_vecs[:used_data], [n_of_batches, batch_size, sent_len, word_len])
    feat_batches = np.reshape(feature_vecs[:used_data], [n_of_batches, batch_size, sent_len, feat_len])
    sent_len_batches = np.reshape(sent_lens[:used_data], [n_of_batches, batch_size])
    word_len_batches = np.reshape(word_lens[:used_data], [n_of_batches, batch_size, sent_len])

    return (label_batches, word_batches, char_batches, feat_batches, sent_len_batches, word_len_batches)

"""
Sets up the models, feeds them the training and validation batches and computes
the scores.
"""
def train_model(config, train_batches, validation_batches, embeddings, verbose=True):
    (train_labels, train_words, train_chars, train_feats, train_slens, train_wlens) = train_batches
    (valid_labels, valid_words, valid_chars, valid_feats, valid_slens, valid_wlens) = validation_batches

    # Get number of characters in data to set up lookup table
    n_chars = max(np.max(train_chars), np.max(valid_chars))

    # Create models
    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=False):
            train_model = Model(
                config,
                train_labels,
                train_words,
                train_chars,
                train_feats,
                train_slens,
                train_wlens,
                n_chars,
                embeddings,
                phase=Phase.Train)

        with tf.variable_scope("model", reuse=True):
            validation_model = Model(
                config,
                valid_labels,
                valid_words,
                valid_chars,
                valid_feats,
                valid_slens,
                valid_wlens,
                n_chars,
                embeddings,
                phase=Phase.Validation)

        # Initialize Tensorflow graph
        sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            train_loss = 0.0
            validation_loss = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            micro_f1 = 0.0

            # Train on all batches
            for batch in range(train_words.shape[0]):
                loss, _ = sess.run([train_model.loss, train_model.train_op], {
                    train_model.y: train_labels[batch],
                    train_model.word_ids: train_words[batch],
                    train_model.char_vecs: train_chars[batch],
                    train_model.feat_vecs: train_feats[batch],
                    train_model.sent_lens: train_slens[batch],
                    train_model.word_lens: train_wlens[batch],
                    train_model.embeddings: embeddings
                })
                train_loss += loss

            # Validation on all batches
            for batch in range(valid_words.shape[0]):
                loss, labels = sess.run([validation_model.loss, validation_model.labels], {
                    validation_model.y: valid_labels[batch],
                    validation_model.word_ids: valid_words[batch],
                    validation_model.char_vecs: valid_chars[batch],
                    validation_model.feat_vecs: valid_feats[batch],
                    validation_model.sent_lens: valid_slens[batch],
                    validation_model.word_lens: valid_wlens[batch],
                    validation_model.embeddings: embeddings
                })
                validation_loss += loss
                (prec, rec, mac_f1, _) = scores(labels[0], labels[1], average="macro")
                precision += prec
                recall += rec
                f1 += mac_f1
                micro_f1 += f1_score(labels[0], labels[1], average="micro")

            # Compute and print scores for this epoch
            train_loss /= train_words.shape[0]
            validation_loss /= valid_words.shape[0]
            precision = precision / valid_words.shape[0] * 100
            recall = recall / valid_words.shape[0] * 100
            f1 = f1 / valid_words.shape[0] * 100
            micro_f1 = micro_f1 / valid_words.shape[0] * 100

            if verbose:
                print(
                    "epoch %d - train loss: %.2f, validation loss: %.2f; prec: %.2f, recall: %.2f, f1: %.2f (micro: %.2f)" %
                    (epoch, train_loss, validation_loss, precision, recall, f1, micro_f1))


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        sys.stderr.write("Usage: %s [option] TRAIN_DATA TEST_DATA EMBEDDINGS EMOLEX\n" % sys.argv[0])
        sys.stderr.write("Options:\n")
        sys.stderr.write("\t-write: Save pre-processed training and test data to file\n")
        sys.stderr.write("\t-read: Training and test data was already pre-processed\n")
        sys.stderr.write("\t(default): Pre-process training and test data, do not write to file\n")
        sys.exit(1)

    # Interpret options
    w = False
    r = False
    if len(sys.argv) == 5:
        args = sys.argv[1:]
    else:
        option = sys.argv[1]
        w = option == "-write"
        r = option == "-read"
        args = sys.argv[2:]

    # Load embeddings
    (word_ids, embeddings) = pickle.load(open(args[2], "rb"))

    # Load or pre-process data
    if r:
        train_data = pickle.load(open(args[0], "rb"))
        valid_data = pickle.load(open(args[1], "rb"))
    else:
        char_map = Numberer()
        label_map = Numberer()
        emolex = read_emolex(args[3])
        train_data = prepare_data(args[0], word_ids, emolex, char_map, label_map)
        valid_data = prepare_data(args[1], word_ids, emolex, char_map, label_map)
        if w:
            with open("traindata", "wb") as train_file:
                pickle.dump(train_data, train_file)
            with open("testdata", "wb") as test_file:
                pickle.dump(valid_data, test_file)

    # Get batches
    config = DefaultConfig()
    train_batches = generate_batches(*train_data, batch_size=config.batch_size)
    validation_batches = generate_batches(*valid_data, batch_size=config.batch_size)

    # Execute the model
    train_model(config, train_batches, validation_batches, embeddings)