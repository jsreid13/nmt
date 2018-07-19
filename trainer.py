from pickle import load
from numpy import array
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import Sequence
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

import sys


class corpus_sequence(Sequence):

    """Docstring for corpus_sequence.
    This creates a generator to process the corpus in batches
    using Keras Sequence to allow for multiprocessing
    """

    def __init__(self, dataset, train, test, batch_size):
        self.batch_size = batch_size
        # load datasets
        self._dataset = dataset
        self._train = train
        self._test = test

    def __len__(self):
        return int(np.ceil(len(self.train) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

    def text_generator(self, text):
        """Generate lines of text for tokenizer so it doesn't eat up all my RAM
        :param text: Large text file
        :returns: line of text

        """
        for line in text:
            yield line


# load a clean dataset
def load_clean_sentences(filename):
    return load(open('pickle/' + filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


target_language = 'french'
# load datasets
dataset = np.asarray(load_clean_sentences('english-%s-both.pkl' % target_language))
train = np.asarray(load_clean_sentences('english-%s-train.pkl' % target_language))
test = np.asarray(load_clean_sentences('english-%s-test.pkl' % target_language))

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
#  eng_tokenizer = load(open('/mnt/E4A696A5A69677AE/en_tokenizer.pkl', 'rb'))
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
#  ger_tokenizer = load(open('/mnt/E4A696A5A69677AE/fr_tokenizer.pkl', 'rb'))
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('%s Vocabulary Size: %d' % (target_language, ger_vocab_size))
print('%s Max Length: %d' % (target_language, ger_length))

# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
ger_tokenizer = 0
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
eng_tokenizer = 0
testY = encode_output(testY, eng_vocab_size)


# define model
model = define_model(ger_vocab_size,
                     eng_vocab_size,
                     ger_length,
                     eng_length,
                     256
                     )
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
#  plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'english_%s_model.h5' % target_language
checkpoint = ModelCheckpoint('models/' + filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min'
                             )
model.fit(trainX,
          trainY,
          epochs=30,
          batch_size=64,
          validation_data=(testX, testY),
          callbacks=[checkpoint],
          verbose=2
          )
#  model.fit_generator(corpus_sequence(trainX, trainY, batch_size=64),
#                      epochs=30,
#                      validation_data=(testX, testY),
#                      callbacks=[checkpoint],
#                      verbose=2,
#                      use_multiprocessing=True,
#                      workers=4
#                      )
