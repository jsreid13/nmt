import json
from pickle import load
from numpy import array
from random import shuffle
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


class corpus_sequence(Sequence):

    """
    This creates a generator to process the corpus in batches
    extending upon Keras' Sequence to allow for multiprocessing
    """

    def __init__(self,
                 pairs,
                 batch_size,
                 batches_per_epoch,
                 src_vocab,
                 tar_vocab,
                 line_length_src,
                 line_length_tar
                 ):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.src_vocab = src_vocab
        # Target are the labels
        self.tar_vocab = tar_vocab
        self.line_length_src = line_length_src
        self.line_length_tar = line_length_tar
        # load datasets
        self.pairs = np.genfromtxt(pairs, autostrip=True, dtype=np.int16)

    def __len__(self):
        """
        Returns the number of batches processed per epoch as the length
        :returns: batches_per_epoch

        """
        return self.batches_per_epoch

    def encode_output(self, sequences):
        """TODO: Docstring for encode_output.
        :returns: TODO

        """
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=self.tar_vocab)
            ylist.append(encoded)
        y = array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], self.tar_vocab)
        return y

    def __getitem__(self, idx):
        """
        Returns a tuple of (source text, target translation) pairs, with the
        target translation as a sparse matrix
        :returns: tuple with 2 numpy arrays

        """
        source_sample = np.empty((self.batch_size, self.line_length_src), dtype=int)
        target_sample = np.empty((self.batch_size, self.line_length_tar), dtype=int)

        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            source_sample[i - (idx * self.batch_size)] = array(self.pairs[i, :self.line_length_src])
            target_sample[i - (idx * self.batch_size)] = array(self.pairs[i, self.line_length_src:])

        return source_sample, self.encode_output(target_sample)

    def on_epoch_end(self):
        """Shuffle the datasets each epoch
        :returns: None

        """
        np.random.shuffle(self.pairs)


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


source_language = 'french'
target_language = 'english'

corpra_stats = json.load(open('corpra/%s_to_%s_stats.json' % (source_language, target_language), 'r'))
lines_per_batch = 512
# Minus 1 because the generator takes lines from the current batch_count to batch_count+1
# so it overflows the end of the dataset otherwise
train_batches_per_epoch = corpra_stats['number_of_train_phrases'] // lines_per_batch - 1
val_batches_per_epoch = corpra_stats['number_of_val_phrases'] // lines_per_batch - 1
src_num_unique_words = corpra_stats['source_vocabulary'] + 1
tar_num_unique_words = corpra_stats['target_vocabulary'] + 1
max_line_length_src = corpra_stats['longest_source_phrase']
max_line_length_tar = corpra_stats['longest_target_phrase']

# define model
model = define_model(src_num_unique_words,
                     tar_num_unique_words,
                     max_line_length_src,
                     max_line_length_tar,
                     256
                     )

model.compile(optimizer='adam', loss='categorical_crossentropy')

# summarize defined model
print(model.summary())
#  plot_model(model, to_file='model.png', show_shapes=True)

# fit model
filename = '%s_%s_model.h5' % (source_language, target_language)
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

train_gen = corpus_sequence('corpra/%s_to_%s_train.txt' % (source_language, target_language),
                            lines_per_batch,
                            train_batches_per_epoch,
                            src_num_unique_words,
                            tar_num_unique_words,
                            max_line_length_src,
                            max_line_length_tar
                            )
val_gen = corpus_sequence('corpra/%s_to_%s_val.txt' % (source_language, target_language),
                          lines_per_batch,
                          val_batches_per_epoch,
                          src_num_unique_words,
                          tar_num_unique_words,
                          max_line_length_src,
                          max_line_length_tar
                          )
model.fit_generator(generator=train_gen,
                    epochs=30,
                    validation_data=val_gen,
                    callbacks=[checkpoint],
                    verbose=1,
                    use_multiprocessing=True,
                    workers=6
                    )
