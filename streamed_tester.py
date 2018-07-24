from json import load
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
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, target, src):
    actual, predicted = list(), list()
    with open('fr_en_translations.txt', 'w') as all_translations:
        for i, source in enumerate(sources):
            # translate encoded source text
            source = source.reshape((1, source.shape[0]))  # convert vector to array for predict method
            translation = predict_sequence(model, tokenizer, source)
            print(source)
            raw_target = target[i].split()
            raw_src = src[i].split()
            if i < 10:
                print('src=[%s], target=[%s], predicted=[%s]' %
                      (raw_src, raw_target, translation))
            all_translations.write('src=[%s], target=[%s], predicted=[%s]\n' %
                                   (raw_src, raw_target, translation))
            actual.append(raw_target)
            predicted.append(translation)
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


target_language = 'french'
# load datasets
eng_tokenizer = load(open('/mnt/E4A696A5A69677AE/en_tokenizer.pkl', 'rb'))
fr_tokenizer = load(open('/mnt/E4A696A5A69677AE/fr_tokenizer.pkl', 'rb'))
#  print(eng_tokenizer.word_index.values())
#  eng_tokenizer = Tokenizer()
#  fr_tokenizer = Tokenizer()
train = open('/mnt/E4A696A5A69677AE/french-reduced-dataset.txt', 'r')
test = open('/mnt/E4A696A5A69677AE/english-reduced-dataset.txt', 'r')

# load model
model = load_model('models/english_%s_model.h5' % target_language)
# test on some training sequences
#  print('train')
#  evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
print('test')
num_samples = 10
raw_source = [train.readline().strip() for i in range(num_samples)]
raw_target = [test.readline().strip() for i in range(num_samples)]
evaluate_model(model,
               eng_tokenizer,
               encode_sequences(fr_tokenizer, 15, raw_source),
               raw_target,
               raw_source
               )
