# Deep Learning: Sentiment Analysis

This is the semester project I did for the course "Deep Learning" taught by Daniël de Kok at the University of Tübingen in the summer semester 2017.

I have implemented a bidirectional recurrent neural network (biRNN) that is classifying tweets according to their conveyed emotion (anger, disgust, fear, joy, sadness and surprise) leveraging information from word embeddings, entries in an emotion-annotated lexicon and character-level embeddings. My model reaches a micro-averaged F1 score of 62.81 and a macro-averaged F1 score of 52.91.

Details on the design and implementation can be found in the accompanying project plan/proposal and report (under _doc_).

### External resources

* This project uses the (word-level) [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), created by Saif M. Mohammad and Peter D. Turney at the National Research Council Canada.
* It also uses the pre-trained Twitter word vectors from [GloVe](https://nlp.stanford.edu/projects/glove/) which are made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/.

## Preprocessing

The original GloVe embeddings were far too verbose for my purpose, so I stripped the unused information in a first preprocessing step, also to reduce their enormous size. Download the pre-trained Twitter word vectors from [their website](https://nlp.stanford.edu/projects/glove/) (I used the 200d vectors), apply `get_embeddings(<filename>, 100000)` from `preprocessing.py` to them and `pickle`the result into a file (cf. `preprocessing.main`).

## Data format

The training and validation data has to be provided in a very simple tab-separated format with three columns; the first being completely irrelevant (the original data provided by the lecturer contained a tweet id), the second column containg the Tweet text and the third column containing two colons followed by the emotion class. For example:

<code><tweet_id>&nbsp;&nbsp;&nbsp;&nbsp;i'm SO happy today!!! #yippie&nbsp;&nbsp;&nbsp;&nbsp;:: joy</code>

## Usage

To run the model, use the following command:

`python train.py TRAIN_DATA TEST_DATA EMBEDDINGS EMOLEX`

where `TRAIN_DATA` is the data the model should be trained on and `TEST_DATA` is
the data to evaluate on. `EMBEDDINGS` should be the ﬁle containing the preprocessed
word embeddings and `EMOLEX` the one with the NRC Emotion Lexicon
data.

For more options, refer to _doc/project_report.pdf_.
