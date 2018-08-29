import string
import re
import json
from pickle import load
from pickle import dump
from random import shuffle
from unicodedata import normalize
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open('corpra/' + filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


def clean_phrase(phrase, trans_table, re_print):
    """TODO: Docstring for clean_phrase.

    :phrase: Phrase of text
    :returns: Cleaned phrase of text
    """
    # normalize unicode characters
    #  line = normalize('NFD', phrase).encode('ascii', 'ignore')
    #  line = line.decode('UTF-8')
    # tokenize on white space
    line = phrase.split()
    # convert to lowercase
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [word.translate(trans_table) for word in line]
    # remove non-printable chars form each token
    #  line = [re_print.sub('', w) for w in line]
    # remove tokens with numbers in them
    line = [word for word in line if word.isalpha()]
    # combine cleaned words back into phrase
    return ' '.join(line)


# clean a list of lines
def clean_pairs(parallel_text):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for tar_phrase, src_phrase in parallel_text:
        tar_clean = clean_phrase(tar_phrase, table, re_print)
        src_clean = clean_phrase(src_phrase, table, re_print)
        cleaned.append([tar_clean, src_clean])
    return array(cleaned)


# save a list of clean sentences to file
def save_pickle(sentences, filename):
    dump(sentences, open('pickle/' + filename, 'wb'))
    print('Saved: %s' % filename)


# fit a tokenizer
def create_tokenizer(lines, vocab=None):
    tokenizer = Tokenizer(num_words=vocab)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# map an integer to a word
def word_for_id(integer, tokenizer):
    if not isinstance(integer, int):
        raise ValueError
    if integer == 0:
        return ''
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


source_language = 'french'
target_language = 'english'
# Clean the data
# load dataset
filename = 'fra.txt'
doc = load_doc(filename)
# split into english-target language pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# spot check
#  for i in range(20):
#      print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))

# Tokenize the words
vocab_length = None
phrase_length = 5
validation_ratio = 0.1
test_ratio = 0.1
if validation_ratio + test_ratio > 0.5:
    raise ValueError('Need at least half the data for training')
# Select all phrases containing phrase_length words
idx_phrase = [len(sentence.split(' ')) for sentence in clean_pairs[:, 0]].index(phrase_length + 1)
# First column is english, second is french
target_tokenizer = create_tokenizer(clean_pairs[:idx_phrase, 0], vocab_length)
source_tokenizer = create_tokenizer(clean_pairs[:idx_phrase, 1], vocab_length)
save_pickle(source_tokenizer, '%s_tokenizer.pkl' % source_language)
save_pickle(target_tokenizer, '%s_tokenizer.pkl' % target_language)

stats = {'longest_source_phrase': max([len(sentence.split(' ')) for sentence in clean_pairs[:idx_phrase, 1]]),
         'longest_target_phrase': phrase_length,
         'source_vocabulary': max(source_tokenizer.word_index.values()),
         'target_vocabulary': max(target_tokenizer.word_index.values()),
         }

encoded_target = encode_sequences(target_tokenizer, stats['longest_target_phrase'], clean_pairs[:idx_phrase, 0])
encoded_target = [' '.join([str(word) for word in phrase]) for phrase in encoded_target]
encoded_source = encode_sequences(source_tokenizer, stats['longest_source_phrase'], clean_pairs[:idx_phrase, 1])
encoded_source = [' '.join([str(word) for word in phrase]) for phrase in encoded_source]

stats['number_of_train_phrases'] = int(idx_phrase * (1-validation_ratio-test_ratio))
stats['number_of_val_phrases'] = int(idx_phrase * validation_ratio)
stats['number_of_test_phrases'] = int(idx_phrase * test_ratio)

pairs = list(zip(encoded_source, encoded_target))
shuffle(pairs)
for i in range(10):
    source_ex = ' '.join(filter(None, [word_for_id(int(token), source_tokenizer) for token in pairs[i][0].split(' ')]))
    target_ex = ' '.join(filter(None, [word_for_id(int(token), target_tokenizer) for token in pairs[i][1].split(' ')]))
    print('[%s] => [%s]' % (source_ex, target_ex))

# Combine pairs
train_pairs = pairs[:stats['number_of_train_phrases']]
val_pairs = pairs[stats['number_of_train_phrases']:stats['number_of_train_phrases']+stats['number_of_val_phrases']]
test_pairs = pairs[stats['number_of_train_phrases']+stats['number_of_val_phrases']:]
# Write to file as new lines
with open('corpra/%s_to_%s_train.txt' % (source_language, target_language), 'w') as train_file,\
        open('corpra/%s_to_%s_val.txt' % (source_language, target_language), 'w') as val_file,\
        open('corpra/%s_to_%s_test.txt' % (source_language, target_language), 'w') as test_file:
    [train_file.write(' '.join(pair) + '\n') for pair in train_pairs]
    [val_file.write(' '.join(pair) + '\n') for pair in val_pairs]
    [test_file.write(' '.join(pair) + '\n') for pair in test_pairs]

# Get stats on the dataset for padding later
print(stats)
with open('corpra/%s_to_%s_stats.json' % (source_language, target_language), 'w') as f:
    json.dump(stats, f)
