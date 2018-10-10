from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras import backend as K
from keras.utils import multi_gpu_model
import numpy as np

maxlen = 125
batch_size = 32

def load_data(csv_file):
    current_data = []
    file = open(csv_file)
    for line in file:
        line_stats = []
        split_line = line.split(";", 7)
        for i in range(2, 7):
            line_stats.append(float(split_line[i].strip("\n")))
        current_data.append(line_stats)
    return current_data 

current_data = load_data("aaba-1m.csv")
x_train = np.zeros((len(current_data)-maxlen, maxlen, 5), dtype=float)
y_train = np.zeros((len(current_data)-maxlen, 1), dtype=float)
for i in range(0, len(current_data)-(maxlen+1)):
    for j in range(0, maxlen):
        x_train[i][j][0] = current_data[i+j][0]
        x_train[i][j][1] = current_data[i+j][1]
        x_train[i][j][2] = current_data[i+j][2]
        x_train[i][j][3] = current_data[i+j][3]
        x_train[i][j][4] = current_data[i+j][4]
    y_train[i][0] = current_data[i+(maxlen+1)][2]

print('Build model...')
model = Sequential()
model.add(LSTM(1, input_shape=(125,5), dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='linear'))

parallel = multi_gpu_model(model, gpus=2)

# try using different optimizers and different optimizer configs
parallel.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
parallel.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_train, y_train))

score, acc = parallel.evaluate(x_train, y_train,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)