import string
import re
from pickle import load
from pickle import dump
from numpy.random import shuffle
from unicodedata import normalize
from numpy import array


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


# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open('pickle/' + filename, 'wb'))
    print('Saved: %s' % filename)


target_language = 'french'
# Clean the data
# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into english-target language pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# spot check
for i in range(20):
    print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))

# Split the data
# reduce dataset size
n_sentences = 25000
dataset = clean_pairs[:n_sentences]
# Divide the dataset into 90% training, 10% test data
split = int(n_sentences * 0.9)
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:split], dataset[split:]
# save
save_clean_data(dataset, 'english-%s-both.pkl' % target_language)
save_clean_data(train, 'english-%s-train.pkl' % target_language)
save_clean_data(test, 'english-%s-test.pkl' % target_language)
