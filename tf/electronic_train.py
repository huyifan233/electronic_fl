import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


DATA_PATH_WIN = "C:\\Users\\yifanhu\\Documents\\smart_grid_stability_augmented.csv"
DATA_PATH = "/home/chunxin.hyf/smart_grid_stability_augmented.csv"
TRAIN_DATA_LEN = 54000
BATCH_SIZE = 32
EPOCH = 10
LEARNING_RATE = 0.0001

def consturct_dataset():
    data = pd.read_csv(DATA_PATH)
    map1 = {'unstable': 0, 'stable': '1'}
    data['stabf'] = data['stabf'].replace(map1)
    data = data.sample(frac=1)
    data2 = data
    # f_most_correlated = data.corr().nlargest(14, 'stabf')['stabf'].index
    # print(data2.corr())


    X = data.iloc[:, :12]
    Y = data.iloc[:, 13]

    X_train = X.iloc[:TRAIN_DATA_LEN, :]
    Y_train = Y.iloc[:TRAIN_DATA_LEN]

    X_test = X.iloc[TRAIN_DATA_LEN:, :]
    Y_test = Y.iloc[TRAIN_DATA_LEN:]
    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, Y_train, X_test, Y_test

def grad(model, inputs, targets, loss_fn):
  with tf.GradientTape() as tape:
    pred = model(inputs)
    loss_value = loss_fn(y_true=targets, y_pred=pred)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def main():

    X_train, Y_train, X_test, Y_test = consturct_dataset()

    # classifier = keras.Sequential()
    #
    # # Input layer and first hidden layer
    # classifier.add(layers.Dense(units=24, kernel_initializer='uniform', activation='relu', input_dim=12))
    #
    # # Second hidden layer
    # classifier.add(layers.Dense(units=24, kernel_initializer='uniform', activation='relu'))
    #
    # # Third hidden layer
    # classifier.add(layers.Dense(units=12, kernel_initializer='uniform', activation='relu'))
    #
    # # Single-node output layer
    # classifier.add(layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    #
    # # ANN compilation
    # classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model = tf.keras.Sequential([
        layers.Dense(units=24, kernel_initializer='uniform', activation='relu', input_dim=12),
        layers.Dense(units=24, kernel_initializer='uniform', activation='relu'),
        layers.Dense(units=12, kernel_initializer='uniform', activation='relu'),
        layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'),
    ])

    cross_val_round = 1
    print(f'Model evaluation\n')

    # model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(),metrics=["accuracy"],)
    loss_func = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for train_index, val_index in KFold(10, shuffle=True, random_state=10).split(X_train):

        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        x_val = np.array(x_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)
        # print(np.array(x_train).shape, np.array(y_train).shape)
        loss_value, grads = grad(model, x_train, y_train, loss_func)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # model.fit(x_train, y_train, epochs=50, verbose=0)
        # classifier.fit(x_train, y_train, epochs=50, verbose=0)
        # classifier_loss, classifier_accuracy = model.evaluate(x_val, y_val, verbose=2)
        # classifier_loss, classifier_accuracy = classifier.evaluate(x_val, y_val)
        # print(f'Round {cross_val_round} - Loss: {classifier_loss:.4f} | Accuracy: {classifier_accuracy * 100:.2f} %')
        print("loss: ", loss_value.numpy())
        cross_val_round += 1



if __name__ == "__main__":
    main()