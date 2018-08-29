import json
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from pprint import pprint


# load a clean dataset
def load_clean_sentences(filename):
    return load(open('pickle/' + filename, 'rb'))


# fit a tokenizer
def tokenize(tokenizer, line):
    tokenizer.fit_on_texts(line)
    encoded_sentence = tokenizer.texts_to_sequences(line)
    return encoded_sentence


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


# map an integer to a word
def word_for_id(integer, tokenizer):
    if integer == 0:
        return ''
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is '':
            break
        target.append(word)
    return ' '.join(target)


# evaluate the skill of the model
def evaluate_model(model, src_tokenizer, tar_tokenizer, source, target):
    actual, predicted = list(), list()
    with open('fr_en_translations.txt', 'w') as all_translations:
        for i, encoded_phrase in enumerate(source):
            # translate encoded source text
            # convert vector to array for predict method
            encoded_phrase = array(encoded_phrase)
            encoded_phrase = encoded_phrase.reshape((1, encoded_phrase.shape[0]))
            translation = predict_sequence(model, tar_tokenizer, encoded_phrase)
            raw_tar = ' '.join([word_for_id(token, tar_tokenizer) for token in target[i]]).strip()
            raw_src = ' '.join([word_for_id(token, src_tokenizer) for token in source[i]]).strip()
            if i < 10:
                print('src=[%s], target=[%s], predicted=[%s]' %
                      (raw_src, raw_tar, translation))
            all_translations.write('src=[%s], target=[%s], predicted=[%s]\n' %
                                   (raw_src, raw_tar, translation))
            actual.append([raw_tar.split(' ')])
            predicted.append(translation.split(' '))
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# evaluate the skill of the model
def user_evaluate_model(model, tokenizer, source, raw_src):
    # translate encoded source text
    #  source = source.reshape((1, source.shape[0]))  # convert vector to array for predict method
    translation = predict_sequence(model, tokenizer, source)
    print('src=[%s], predicted=[%s]' % (raw_src, translation))


source_language = 'french'
target_language = 'english'
# load datasets
src_tokenizer = load(open('pickle/%s_tokenizer.pkl' % source_language, 'rb'))
tar_tokenizer = load(open('pickle/%s_tokenizer.pkl' % target_language, 'rb'))

test_file = open('corpra/%s_to_%s_test.txt' % (source_language, target_language), 'r')

# load model
model = load_model('models/%s_%s_model.h5' % (source_language, target_language))
# test on some training sequences
#  print('train')
#  evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
corpra_stats = json.load(open('corpra/%s_to_%s_stats.json' % (source_language, target_language),
                              'r'))
num_samples = corpra_stats['number_of_test_phrases']
raw_source = []
raw_target = []
for i in range(num_samples):
    pair = test_file.readline().split(' ')
    if pair == '':
        break
    src_sentence = pair[:corpra_stats['longest_source_phrase']]
    tar_sentence = pair[corpra_stats['longest_source_phrase']:]
    # Convert string numbers to integers for tokenizer
    raw_source.append([int(token) for token in src_sentence])
    raw_target.append([int(token) for token in tar_sentence])

evaluate_model(model, src_tokenizer, tar_tokenizer, raw_source, raw_target)
#  phrase = 'tu'
#  user_evaluate_model(model,
#                      tar_tokenizer,
#                      encode_sequences(src_tokenizer, corpra_stats['longest_source_phrase'], [phrase]),
#                      phrase
#                      )
