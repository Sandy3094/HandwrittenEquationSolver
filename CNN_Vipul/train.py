import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

df = pd.read_csv("data/processedData.csv")

dataY = df["784"].to_numpy()
dataX = df.drop(["784"], axis=1).to_numpy()
print(dataX.shape)
dataX = dataX.reshape((dataX.shape[0], 28, 28, 1))
print(dataY)
dataY = to_categorical(dataY)
print(dataY)


def define_model():
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(14, activation="softmax"))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def custom_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(14, activation="softmax"))
    # Compile model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories, models = list(), list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = custom_model()
        # select rows for train and test
        trainX, trainY, testX, testY = (
            dataX[train_ix],
            dataY[train_ix],
            dataX[test_ix],
            dataY[test_ix],
        )
        # fit model
        print(trainX.shape, trainY.shape, testX.shape, testY.shape)
        history = model.fit(
            trainX,
            trainY,
            epochs=10,
            batch_size=32,
            validation_data=(testX, testY),
            verbose=0,
        )
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print("> %.3f" % (acc * 100.0))
        # stores scores
        models.append(model)
        scores.append(acc)
        histories.append(history)
    return scores, histories, models


# scores, histories, models = (evaluate_model(dataX, dataY))

model = custom_model()
model.fit(
    dataX,
    dataY,
    epochs=10,
    batch_size=32,
    verbose=0,
)
model_json = model.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_final.h5")