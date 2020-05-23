import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Embedding
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import json
import pickle

path = 'data/training_corpus.txt'
text = open(path).read().lower()
print('Tamanho do corpus:', len(text))

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)
print("Qtd. de palavras:", len(words))

unique_words = np.unique(words)
print("Qtd. de palavras únicas:", len(unique_words))
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

a = open('data/vocab.json', 'w')
a.write(json.dumps(unique_words.tolist()))



SEQUENCE_LENGTH = 3
prev_words = []
next_words = []
for i in range(len(words) - SEQUENCE_LENGTH):
    prev_words.append(words[i:i + SEQUENCE_LENGTH])
    next_words.append(words[i + SEQUENCE_LENGTH])




X = np.zeros((len(prev_words), SEQUENCE_LENGTH), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j] = unique_word_index[each_word]
    Y[i, unique_word_index[next_words[i]]] = 1

# load the whole embedding into memory
embeddings_index = dict()
f = open('./data/glove_s50.txt')
for line in f:
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(unique_words), 50))
for i in range(len(unique_words)):
    word = unique_words[i]
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

e = Embedding(len(unique_words), 50, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

model = Sequential()
model.add(e)
model.add(LSTM(256))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

print(model.summary())

optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X, Y, validation_split=0.05,
                    batch_size=128, epochs=120, shuffle=True).history

print(history)

model.save('saved_models/word_prediction.h5')
pickle.dump(history, open("saved_models/history.p", "wb"))

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

