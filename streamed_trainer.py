import json
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

from pprint import pprint


class corpus_sequence(Sequence):

    """
    This creates a generator to process the corpus in batches
    extending upon Keras' Sequence to allow for multiprocessing
    """

    def __init__(self,
                 train,
                 test,
                 batch_size,
                 batches_per_epoch,
                 src_vocab,
                 tar_vocab,
                 line_length_french,
                 line_length_english
                 ):
        self.batch_size = batch_size
        # load datasets
        self.train = train
        self.test = test
        self.batches_per_epoch = batches_per_epoch
        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab
        self.line_length_french = line_length_french
        self.line_length_english = line_length_english
        self.batch_counter = 0

    def __len__(self):
        """
        Returns the number of batches processed per epoch as the length
        :returns: batches_per_epoch

        """
        return self.batches_per_epoch

    def __getitem__(self, idx):
        """
        Returns a tuple of (source text, target translation) pairs, with the
        target translation as a sparse matrix
        :returns: tuple with 2 numpy arrays

        """
        train_sample = np.empty((self.batch_size, self.line_length_french), dtype=int)
        test_sample = np.empty((self.batch_size, self.line_length_english), dtype=int)
        for i in range(self.batch_size):
            train_line = self.train.readline()
            split_train_line = [int(weight) for weight in train_line.strip().split(' ')
                                if weight != '']
            len_train = len(split_train_line)
            # Pad with zeros
            [split_train_line.append(0.0) for i in range(self.line_length_french - len_train)]

            test_line = self.test.readline()
            split_test_line = [int(weight) for weight in test_line.strip().split(' ')]
            len_test = len(split_test_line)
            [split_test_line.append(0.0) for i in range(self.line_length_english - len_test)]

            train_sample[i] = array(split_train_line)
            test_sample[i] = array(split_test_line)

        #  self.batch_counter += 1
        #  print(self.batch_counter / self.batches_per_epoch)

        return train_sample, to_categorical(test_sample, num_classes=self.src_vocab)

    def on_epoch_end(self):
        """Restart at the beginning on the text file so we're not always training on new data
        :returns: None

        """
        self.train.seek(0)
        self.test.seek(0)
        #  self.batch_counter = 0


def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    """
    The Keras neural network created for this translation. This uses the encoder-decoder
    architecture with LSTM encoders and decoders to convert the sentences into a fixed
    length word vector

    :param int src_vocab: Number of unique words tokenized in the source language
    :param int tar_vocab: Number of unique words tokenized in the target language
    :param int src_timesteps: Maximum number of words in a phrase in the source language
    :param int tar_timesteps: Maximum number of words in a phrase in the target language
    :param int n_units: Number of LSTM neurons in the encoder and decoder
    :returns: A Keras model
    """
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


target_language = 'french'

corpra_stats = json.load(open('corpra/english_%s_stats.json' % target_language, 'r'))
lines_per_batch = 256
batches_per_epoch = corpra_stats['number_of_sentences'] // lines_per_batch
src_num_unique_words = corpra_stats['english_vocabulary']
tar_num_unique_words = corpra_stats['target_vocabulary']
max_line_length_english = corpra_stats['longest_english_sentence']
max_line_length_french = corpra_stats['longest_target_sentence']

# define model
model = define_model(tar_num_unique_words,
                     src_num_unique_words,
                     max_line_length_french,
                     max_line_length_english,
                     256
                     )

model.compile(optimizer='adam', loss='categorical_crossentropy')

# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

# fit model
filename = 'english_%s_model.h5' % target_language
checkpoint = ModelCheckpoint('models/' + filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min'
                             )

#  model.fit(trainX,
#            trainY,
#            epochs=10,
#            batch_size=64,
#            validation_data=(testX, testY),
#            callbacks=[checkpoint],
#            verbose=2
#            )

with open('corpra/encoded_en.txt', 'r') as english_encoded\
        , open('corpra/encoded_%s.txt' % target_language, 'r') as target_encoded:
    gen = corpus_sequence(target_encoded,
                          english_encoded,
                          lines_per_batch,
                          batches_per_epoch,
                          src_num_unique_words,
                          tar_num_unique_words,
                          max_line_length_french,
                          max_line_length_english
                          )
    model.fit_generator(generator=gen,
                        epochs=30,
                        validation_data=gen,
                        callbacks=[checkpoint],
                        verbose=1,
                        use_multiprocessing=True,
                        workers=1
                        )
