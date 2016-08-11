    import numpy as np
    from keras.models import Sequential
    from keras.layers.core import Activation, Dense
    from keras.optimizers import SGD

    training_data = np.array([[0,0,0,
                               0,0,0,
                               0,0,0],
                              [0,0,1,
                               0,1,0,
                               0,0,1],
                              [0,0,1,
                               0,1,0,
                               1,0,0],
                              [0,1,0,
                               0,1,0,
                               0,1,0]], "float32")

    target_data = np.array([[0],[0],[1],[1]], "float32")

    validation_data = np.array([[0,0,0,
                                 0,0,0,
                                 0,0,0],
                                [1,0,0,
                                 0,1,0,
                                 1,0,0],
                                [1,0,0,
                                 0,1,0,
                                 0,0,1],
                                [0,0,1,
                                 0,0,1,
                                 0,0,1]], "float32")

    model = Sequential()
    model.add(Dense(2, input_dim=9, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    history = model.fit(training_data, target_data, nb_epoch=10000, batch_size=4, verbose=0)

    print(model.predict(validation_data))
