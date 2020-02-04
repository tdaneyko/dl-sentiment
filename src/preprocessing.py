# Author: Thora Daneyko, 3822667
# Honor Code:  I pledge that this program represents my own work.

import sys
import pickle
import numpy as np
from nltk.tokenize import TweetTokenizer
from numberer import Numberer

"""
Reads the original GloVe embeddings and produces
    1) a list with the words in the dictionary, where the index
    of a word corresponds to its index in the lookup table, and
    2) the lookup table with the word embedding vectors.
This method is not used by train.py, but was employed to obtain
the embeddings file shipped with this project.
"""
def get_embeddings(embeddings_file, dict_size=sys.maxsize):
    # Read embeddings
    word_ids = list()
    embeddings = list()
    with open(embeddings_file, "r", encoding="utf8") as embed_file:
        for (n, line) in enumerate(embed_file):
            if n >= dict_size:
                break
            items = line.split(" ")
            word = items[0]
            word_ids.append(word)
            embed_vec = [float(i) for i in items[1:]]
            embeddings.append(embed_vec)

    # Convert to numpy matrix
    embed_size = len(embeddings[0])
    embed_matrix = np.empty((dict_size+1, embed_size))
    embed_matrix[0:dict_size] = embeddings

    # Get <unknown> vector by taking mean of vectors of rare words
    rare_start = int(dict_size*0.9)
    embed_matrix[dict_size] = np.mean(embed_matrix[rare_start:dict_size], axis=0)

    embed_data = (word_ids, embed_matrix)
    return embed_data

"""
Reads the EmoLex data into a map from word to feature vector.
"""
def read_emolex(emolex_file):
    # Read EmoLex
    emolex = {}
    with open(emolex_file, "r", encoding="utf8") as emo_file:
        emos = []
        for line in emo_file:
            fields = line.split("\t")
            emos.append(float(fields[2]))
            if len(emos) == 10:
                word = fields[0]
                emolex[word] = emos
                emos = []
    return emolex

"""
The actual pre-processing. Accepts a data file with annotated tweets,
a list of words where the index of a word corresponds to its id (as
produced by get_embeddings), an EmoLex map (as produced by read_emolex),
and token-to-int mappings for characters and labels.
It will produce numpy matrices containing
    1) the labels of the tweets,
    2) the lookup ids of the words in the tweets,
    3) the character id vectors of the words in the tweets,
    4) the all-caps + EmoLex feature vectors of the words in the tweets,
    5) the actual lengths of the words in the tweets, and
    6) the actual lengths of the tweets.
"""
def prepare_data(data_file, id_to_word_mapping, emolex, char_map=Numberer(), label_map=Numberer()):
    # Feature vector for words not in EmoLex
    emolex_default = [0.0] * 10
    # Count unknown tokens for statistics
    overall_tokens = 0
    unknown_tokens = 0

    # Create word-to-id mapping from word list
    word_map = {}
    for (i, word) in enumerate(id_to_word_mapping):
        word_map[word] = str(i)
    unknown_id = len(word_map)

    tokenizer = TweetTokenizer()
    data = ([], [], [], [], [], [])
    sent_len = 0 # stores maximal tweet length
    word_len = 0 # stores maximal word length
    feat_len = 11

    # Process data tweet by tweet, token by token
    with open(data_file, "r", encoding="utf8") as file:
        for line in file:
            split1 = line.find("\t")
            split2 = line.rfind("\t")
            label = label_map.number(line[split2:])
            sentence = line[split1:split2].strip()
            tokenized = tokenizer.tokenize(sentence)

            word_ids = []
            char_vecs = []
            feature_vecs = []
            word_lens = []

            for token in tokenized:
                char_vec = [char_map.number(c) for c in token]
                word_len = max(word_len, len(char_vec))
                word_lens.append(len(char_vec))
                char_vecs.append(char_vec)

                feature_vec = [float(token.isupper())]

                if token[0] == "#":
                    token = "<hashtag>"
                elif token[0] == "@":
                    token = "<user>"
                elif token.isdigit():
                    token = "<number>"
                elif token.startswith("http://") or token.startswith("https://") or token.startswith("www."):
                    token = "<url>"
                else:
                    token = token.lower()

                feature_vec += emolex.get(token, emolex_default)
                feature_vecs.append(feature_vec)

                token = word_map.get(token, unknown_id)
                overall_tokens += 1
                if token == unknown_id: unknown_tokens += 1

                word_ids.append(token)

            data[0].append(label)
            data[1].append(word_ids)
            data[2].append(char_vecs)
            data[3].append(feature_vecs)
            data[4].append(len(word_ids))
            data[5].append(word_lens)
            sent_len = max(sent_len, len(word_ids))

    # Print statistics
    print("%.2f percent of tokens not in dictionary" % (unknown_tokens / overall_tokens * 100))

    # Convert collected data to numpy matrices
    data_len = len(data[0])
    n_of_labels = label_map.max_number()
    data_tensors = (np.zeros([data_len, n_of_labels]),
                     np.zeros([data_len, sent_len]),
                     np.zeros([data_len, sent_len, word_len]),
                     np.zeros([data_len, sent_len, feat_len]),
                     np.array(data[4]),
                     np.zeros([data_len, sent_len]))
    for i in range(data_len):
        data_tensors[0][i,data[0][i]] = 1
        data_tensors[1][i,0:len(data[1][i])] = data[1][i]
        data_tensors[5][i,0:len(data[5][i])] = data[5][i]
        for j in range(len(data[1][i])):
            data_tensors[2][i,j,0:len(data[2][i][j])] = data[2][i][j]
            data_tensors[3][i,j,0:len(data[3][i][j])] = data[3][i][j]

    return data_tensors

"""
This main method was only used by myself to produce pre-processed
embedding and data files.
"""
if __name__ == "__main__":
    embed_data = get_embeddings(sys.argv[1], 100000)
    with open("embeddings", "wb") as embed_file:
        pickle.dump(embed_data, embed_file)

    if len(sys.argv) == 4:
        char_map = Numberer()
        label_map = Numberer()
        emolex = read_emolex("emolex.txt")
        train_tensors = prepare_data(sys.argv[2], embed_data[0], emolex, char_map, label_map)
        with open("traindata", "wb") as train_file:
            pickle.dump(train_tensors, train_file)
        test_tensors = prepare_data(sys.argv[3], embed_data[0], emolex, char_map, label_map)
        with open("testdata", "wb") as test_file:
            pickle.dump(test_tensors, test_file)