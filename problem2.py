import matplotlib.pyplot as plt
import numpy as np

import config
import scipy.io
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model


def plot_and_prep_data(filepath):
    data = scipy.io.loadmat(filepath)['data']
   
    features = data[:, :2]
    target = data[:, 2]

    negative_indices = target == 0
    positive_indices = target != 0

    negative_features = features[negative_indices]
    positive_features = features[positive_indices]

    plt.scatter(negative_features[:, 0], negative_features[:, 1], c='red')
    plt.scatter(positive_features[:, 0], positive_features[:, 1], c='blue')
    plt.savefig('images/p2_data_vis.png', dpi=300)

    negative_target = np.zeros((negative_features.shape[0], 1))
    negative_inputs = np.concatenate([negative_features, negative_target], axis=1)
    np.random.shuffle(negative_inputs)

    positive_target = np.ones((positive_features.shape[0], 1))
    positive_inputs = np.concatenate([positive_features, positive_target], axis=1)
    np.random.shuffle(positive_inputs)

    inputs = np.concatenate([negative_inputs, positive_inputs], axis=0)
    np.random.shuffle(inputs)

    features = inputs[:, :2]
    target = inputs[:, 2]

    return features, target


def create_underfitting_model():
    model = Sequential()

    model.add(Dense(256, input_dim=2, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))

    return model


def create_optimal_model():
    model = Sequential()

    # model.add(Dense(256, input_dim=1, activation='relu'))
    # model.add(Dense(256), activation='relu')
    # model.add(Dense(256), activation='relu')
    # model.add(Dense(512), activation='relu')
    # model.add(Dense(512), activation='relu')
    # model.add(Dense(1024), activation='relu')
    # model.add(Dense(1024), activation='relu')
    # model.add(Dense(1))

    return model


def create_overfitting_model():
    model = Sequential()

    model.add(Dense(256, input_dim=1, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1))

    return model


def train_and_evaluate_model(model, features, target):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    # tensorboard = TensorBoard(
    #     log_dir='logs',
    #     histogram_freq=1
    # )
    # keras_callbacks = [
    #     tensorboard
    # ]

    #for _ in range(conf.PROBLEM1.EPOCHS):
    model.fit(x=features,
              y=target,
              validation_split=0.2,
              epochs=1,
              batch_size=120)
              #callbacks=keras_callbacks)

    # plot decision boundary

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-1, 6, 200)
    x_grid, y_grid = np.meshgrid(x, y)
    grid = np.stack([x_grid, y_grid], axis=-1)

    grid = np.reshape(grid, (-1, 2))

    prediction = model.predict(grid)

    print(prediction.shape)


def main():
    # alias
    conf = config.C.PROBLEM2

    features, target = plot_and_prep_data(conf.DATASET_PATH)

    underfit_model = create_underfitting_model()
    train_and_evaluate_model(underfit_model, features, target)

    # optimal_model = create_optimal_model()
    # train_and_evaluate_model(optimal_model)

    # overfit_model = create_overfitting_model()
    # train_and_evaluate_model(overfit_model)
    

if __name__ == '__main__':
    main()
