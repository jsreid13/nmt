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
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    print(integers)
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
            raw_tar = ' '.join([word_for_id(token, tokenizer) for token in target[i]])
            raw_src = ' '.join([word_for_id(token, fr_tokenizer) for token in src[i]])
            if i < 10:
                print('src=[%s], target=[%s], predicted=[%s]' %
                      (raw_src, raw_tar, translation))
            all_translations.write('src=[%s], target=[%s], predicted=[%s]\n' %
                                   (raw_src, raw_tar, translation))
            actual.append(raw_tar)
            predicted.append(translation)
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


target_language = 'french'
# load datasets
eng_tokenizer = load(open('pickle/english_tokenizer.pkl', 'rb'))
fr_tokenizer = load(open('pickle/%s_tokenizer.pkl' % target_language, 'rb'))
#  print(eng_tokenizer.word_index.values())
#  eng_tokenizer = Tokenizer()
#  fr_tokenizer = Tokenizer()
train = open('corpra/encoded_%s.txt' % target_language, 'r')
test = open('corpra/encoded_en.txt', 'r')

# load model
model = load_model('models/english_%s_model.h5' % target_language)
# test on some training sequences
#  print('train')
#  evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
print('test')
corpra_stats = json.load(open('corpra/english_%s_stats.json' % target_language, 'r'))
num_samples = 10
raw_source = []
raw_target = []
for i in range(num_samples):
    src_sentence = train.readline().strip().split(' ')
    tar_sentence = test.readline().strip().split(' ')
    if src_sentence[0] is '' or tar_sentence[0] is '':
        continue
    # Convert string numbers to integers for tokenizer
    raw_source.append([int(token) for token in src_sentence])
    raw_target.append([int(token) for token in tar_sentence])
    print(pad_sequences(raw_source, corpra_stats['longest_target_sentence'], padding='post'))

evaluate_model(model,
               eng_tokenizer,
               pad_sequences(raw_source, corpra_stats['longest_target_sentence'], padding='post'),
               raw_target,
               raw_source
               )
