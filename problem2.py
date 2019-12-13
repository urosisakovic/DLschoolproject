import matplotlib.pyplot as plt
import numpy as np

import config
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
    plt.savefig('images/problem2/data_vis.png', dpi=300)

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

    model.add(Dense(64, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    return model


def create_optimal_model():
    model = Sequential()

    model.add(Dense(256, input_dim=2, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    return model


def create_overfitting_model():
    model = Sequential()

    model.add(Dense(256, input_dim=2, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    return model


def train_and_evaluate_model(model, features, target, model_name):
    model.fit(x=features,
              y=target,
              validation_split=0.2,
              epochs=1000,
              batch_size=120)

    acc = accuracy_score(target, model.predict(features) > 0.5)

    print('Accuracy: {}'.format(acc))

    # plot decision boundary
    x = np.linspace(-4, 4, 1000)
    y = np.linspace(-2, 6, 1000)
    xx, yy = np.meshgrid(x, y)
    # Predict the function value for the whole gid
    grid = np.stack([xx, yy], axis=-1)
    grid = np.reshape(grid, (-1, 2))
    prediction = model.predict(grid)

    print(prediction.max())
    print(prediction.min())
    print(np.average(prediction))

    prediction = prediction.reshape(xx.shape)
    
    prediction = prediction > .5

    plt.clf()
    plt.contourf(xx, yy, prediction, cmap=plt.cm.Spectral)
    plt.scatter(features[:, 0], features[:, 1], c=target, cmap=plt.cm.Spectral)
    plt.title("Logistic Regression")
    plt.savefig('images/problem2/{}_decision_boundary.png'.format(model_name), dpi=300)


def main():
    # alias
    conf = config.C.PROBLEM2

    features, target = plot_and_prep_data(conf.DATASET_PATH)

    # split into training and test set
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

    if conf.UNDERFIT:
        underfit_model = create_underfitting_model()
        train_and_evaluate_model(underfit_model, features_train, target_train, 'underfit')

    if conf.OPTIMAL:
        optimal_model = create_optimal_model()
        train_and_evaluate_model(optimal_model, features_train, target_train, 'optimal')

    if conf.OVERFIT:
        overfit_model = create_overfitting_model()
        train_and_evaluate_model(overfit_model, features_train, target_train, 'overfit')
    

if __name__ == '__main__':
    main()
