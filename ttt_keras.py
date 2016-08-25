import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD

import connect_four_data as ttt_data

# training_data = ttt_data.reshape_training_data_1d(ttt_data.get_training_data_2d())
# target_data = ttt_data.reshape_target_data_1d(ttt_data.get_target_data_2d());
# validation_data = ttt_data.reshape_validation_data_1d(ttt_data.get_validation_data_2d())

training_data = ttt_data.get_training_data_2d()
target_data = ttt_data.get_target_data_2d()
validation_data = ttt_data.get_validation_data_2d()

model = Sequential()
model.add(Convolution2D(4, 2, 2, input_shape=(1, 6, 6), activation='relu'))
model.add(Convolution2D(4, 2, 2, activation='relu'))
model.add(Convolution2D(4, 2, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

history = model.fit(training_data, target_data, nb_epoch=10000, batch_size=4, verbose=0)

print(model.predict(validation_data))
