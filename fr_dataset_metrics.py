import re
import string
import time
from pickle import dump
from unicodedata import normalize
from keras.preprocessing.text import Tokenizer


def analyze_encoded_data(line, highest_int, longest_sentence):
    """TODO: Docstring for analyze_encoded_data.

    :arg1: TODO
    :returns: TODO

    """
    return longest_sentence, highest_int


def write_stats(highest_int, longest_sentence):
    """TODO: Docstring for write_stats.

    :param int highest_int: TODO
    :returns: TODO

    """
    with open('french_dataset_info.txt', 'w') as f:
        f.write('Highest encoded int in dataset: ')
        f.write(str(highest_int))
        f.write('\nLongest sentence in dataset: ')
        f.write(str(longest_sentence))


def get_longest_and_highest(path):
    """TODO: Docstring for get_longest_and_highest.
    :param str path: TODO
    :returns: TODO

    """
    with open(path, 'r') as dataset:
        highest_int = 0
        longest_sentence = 0
        while True:
            line = dataset.readline()
            if line == '':
                break
            text_sample = [int(weight) for weight in line.split()]
            len_sample = len(text_sample)
            highest_num_in_sample = max(text_sample)
            if len_sample > longest_sentence:
                longest_sentence = len_sample
            if highest_num_in_sample > highest_int:
                highest_int = highest_num_in_sample
    return longest_sentence, highest_int


# fit a tokenizer
def tokenize(tokenizer, line):
    encoded_sentence = tokenizer.texts_to_sequences(line)
    return encoded_sentence


# clean a list of lines
def clean_text(line, _re_print, _table):
    # normalize unicode characters
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    # tokenize on white space
    line = line.split()
    # convert to lowercase
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [word.translate(_table) for word in line]
    # remove non-printable chars form each token
    line = [_re_print.sub('', w) for w in line]
    # remove tokens with numbers in them
    line = [word for word in line if word.isalpha()]
    # return as string
    return ' '.join(line)


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open('pickle/' + filename, 'wb'))
    print('Saved: %s' % filename)


t1 = time.time()

path_to_encoded_french = '/mnt/E4A696A5A69677AE/giga-fren.release2.fixed.fr.ids40000'
path_to_corpra_french = '/mnt/E4A696A5A69677AE/giga-fren.release2.fixed.fr'
path_to_encoded_english = '/mnt/E4A696A5A69677AE/giga-fren.release2.fixed.en.ids40000'
path_to_corpra_english = '/mnt/E4A696A5A69677AE/giga-fren.release2.fixed.en'
reduced_corpra_path_french = '/mnt/E4A696A5A69677AE/french-reduced-dataset.txt'
reduced_corpra_path_english = '/mnt/E4A696A5A69677AE/english-reduced-dataset.txt'
encoded_reduced_corpra_path_french = '/mnt/E4A696A5A69677AE/encoded-french-reduced-dataset.txt'
encoded_reduced_corpra_path_english = '/mnt/E4A696A5A69677AE/encoded-english-reduced-dataset.txt'

#  longest_sentence, highest_int = get_longest_and_highest(encoded_reduced_corpra_path_french)
#  print(longest_sentence)
#  input('hello')
#  write_stats(highest_int, longest_sentence)
#  print(str(time.time() - t1))

with open(reduced_corpra_path_english, 'w') as english_dataset,\
        open(reduced_corpra_path_french, 'w') as french_dataset,\
        open(encoded_reduced_corpra_path_french, 'w') as encoded_french_dataset,\
        open(encoded_reduced_corpra_path_english, 'w') as encoded_english_dataset,\
        open(path_to_corpra_french, 'r') as raw_french_dataset,\
        open(path_to_corpra_english, 'r') as raw_english_dataset:
    # Initialize tokenizers
    english_tokenizer = Tokenizer()
    french_tokenizer = Tokenizer()

    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    counter = 0
    longest_french_line = 0
    while True:
        english_text_line = raw_english_dataset.readline()
        if english_text_line == '':
            break
        cleaned_english_text_line = clean_text(english_text_line, re_print, table)
        french_text_line = raw_french_dataset.readline()
        cleaned_french_text_line = clean_text(french_text_line, re_print, table)

        if cleaned_english_text_line == '' or cleaned_french_text_line == '':
            continue
        split_english_text = cleaned_english_text_line.split(' ')
        split_french_text = cleaned_french_text_line.split(' ')

        # only want lines shorter than 10 words
        if len(split_english_text) > 10 or len(split_french_text) >15:
            continue
        if len(split_french_text) > longest_french_line:
            longest_french_line = len(split_french_text)
        counter += 1
        if counter % 1000 == 0:
            print(counter)
        if counter % 10000 == 0:
            break

        english_dataset.write(cleaned_english_text_line + '\n')
        english_tokenizer.fit_on_texts(split_english_text)

        french_dataset.write(cleaned_french_text_line + '\n')
        french_tokenizer.fit_on_texts(split_french_text)

    # Redo it now that tokenizer has weights
    counter = 0
    raw_english_dataset.seek(0)
    raw_french_dataset.seek(0)
    combined_dataset = []
    while True:
        english_text_line = raw_english_dataset.readline()
        if english_text_line == '':
            break
        cleaned_english_text_line = clean_text(english_text_line, re_print, table)
        french_text_line = raw_french_dataset.readline()
        cleaned_french_text_line = clean_text(french_text_line, re_print, table)

        if cleaned_english_text_line == '' or cleaned_french_text_line == '':
            continue
        split_english_text = cleaned_english_text_line.split(' ')
        split_french_text = cleaned_french_text_line.split(' ')

        # only want lines shorter than 10 words
        if len(split_english_text) > 10 or len(split_french_text) >15:
            continue
        if len(split_french_text) > longest_french_line:
            longest_french_line = len(split_french_text)
        counter += 1
        if counter % 1000 == 0:
            print(counter)
        if counter % 10000 == 0:
            break

        #  english_dataset.write(cleaned_english_text_line + '\n')
        #  encoded_english_line = tokenize(english_tokenizer, split_english_text)
        #  encoded_english_dataset.write(' '.join([str(token[0]) for token in encoded_english_line]) + '\n')

        #  french_dataset.write(cleaned_french_text_line + '\n')
        #  encoded_french_line = tokenize(french_tokenizer, split_french_text)
        #  encoded_french_dataset.write(' '.join([str(token[0]) for token in encoded_french_line]) + '\n')
        combined_dataset.append([cleaned_english_text_line, cleaned_french_text_line])

    save_clean_data(combined_dataset, 'english-french.pkl')
    print(str(time.time() - t1))
    print('longest line is: ' + str(longest_french_line))
    print(max(list(french_tokenizer.word_index.values())))

    with open('/mnt/E4A696A5A69677AE/fr_tokenizer.pkl', 'wb') as french_pkl,\
            open('/mnt/E4A696A5A69677AE/en_tokenizer.pkl', 'wb') as english_pkl:
        dump(french_tokenizer, french_pkl)
        dump(english_tokenizer, english_pkl)
