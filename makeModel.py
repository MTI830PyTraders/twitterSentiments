import json
from tensorflow import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import csv
# extract data from a csv
# notice the cool options to skip lines at the beginning
# and to only take data from certain columns
# training = np.genfromtxt('training.1600000.processed.noemoticon.head.csv', delimiter=',', skip_header=1, usecols=(0, 5), dtype=None, encoding='latin-1')
# training = np.random.choice(training)
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', verbose="true", keep_default_na=False, na_values=[], delimiter=',', dtype=[('target', np.uint8),('ids', np.uint8), ('date', str), ('flag', str), ('user', str), ('text', str)], names=['target', 'ids', 'date', 'flag', 'user', 'text'] )
# max_words=3000
training=df.sample(130000)
# df.to_csv("training.1600000.processed.noemoticon.1000.csv", header='false', index='false')
# print(training.values)
# # create our training data from the tweets
# # np.array(np.rec.fromrecords(df.values))
# train_x = np.asarray(df_sampled.get("text"), dtype=str)
# print(train_x[5])
# # index all the sentiment labels
# # np.rec.fromrecords(df.values)
# # train_y = np.array(np.rec.fromrecords(df_sampled.get("target").values))
# train_y = np.asarray(df_sampled.get("target"), dtype=int)
# print(train_y[2])
# only work with the 3000 most popular words found in our dataset
# df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=['target', 'ids', 'date', 'flag', 'user', 'text'] )
# np.random.shuffle(training)
max_words = 10000

# create our training data from the tweets
# train_x = [str(x[1]) for x in training]
train_x = np.array(training.get('text'))
# index all the sentiment labels
# lst = [x[0].strip('"') for x in training]
train_y = np.array(training.get('target'))
# lst = [x[0] for x in training]
# # print(lst[0:10])
# np.array(lst)
# train_y = lst

# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)
# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, 5)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(train_x, train_y,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_split=0.4,
    shuffle=True)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

print('saved model!')
