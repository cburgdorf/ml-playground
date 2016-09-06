import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D
from keras.optimizers import SGD

import data_gen
import connect_four_data as ttt_data

# training_data = ttt_data.reshape_training_data_1d(ttt_data.get_training_data_2d())
# target_data = ttt_data.reshape_target_data_1d(ttt_data.get_target_data_2d());
# validation_data = ttt_data.reshape_validation_data_1d(ttt_data.get_validation_data_2d())

training_data, target_data, target_data_simple = data_gen.generate_samples(500)
#training_data = ttt_data.get_training_data_2d()
#target_data = ttt_data.get_target_data_per_pixel()
validation_data = ttt_data.get_validation_data_2d()

model = Sequential()
# model.add(Convolution2D(6, 2, 2, input_shape=(1,6,6), activation='relu', border_mode='same'))
# model.add(Convolution2D(6, 2, 2, activation='relu', border_mode='same'))
# model.add(Dropout(0.25))
# model.add(Convolution2D(6, 2, 2, activation='relu', border_mode='same'))
# model.add(Convolution2D(6, 2, 2, activation='relu', border_mode='same'))
# model.add(Convolution2D(1, 2, 2, activation='sigmoid', border_mode='same'))

model.add(Convolution2D(4, 2, 2, input_shape=(1, 6, 6), activation='relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(4, 2, 2, activation='relu'))
model.add(Convolution2D(4, 2, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(training_data, target_data_simple, nb_epoch=10000, batch_size=32, verbose=0)

prediction = model.predict(validation_data);
print "validation data"
print validation_data
print "prediction"
print(prediction)
print "rounded prediction"
print(prediction.round())
