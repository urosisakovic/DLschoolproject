import matplotlib.pyplot as plt
import numpy as np

import config
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model


def scatter_plot(features, target):
    negative_indices = target == 0
    positive_indices = target != 0

    negative_features = features[negative_indices]
    positive_features = features[positive_indices]

    plt.scatter(negative_features[:, 0], negative_features[:, 1], c='red')
    plt.scatter(positive_features[:, 0], positive_features[:, 1], c='blue')
    plt.savefig('images/problem2/data_vis.png', dpi=300)


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


def plot_decision_boundary(model, features, target, xmin, xmax, x_density, ymin, ymax, y_density, filepath, title):
    # create a grid of points whose class will be predicted by the model
    x = np.linspace(xmin, xmax, x_density)
    y = np.linspace(ymin, ymax, y_density)
    x_grid, y_grid = np.meshgrid(x, y)
    grid = np.stack([x_grid, y_grid], axis=-1)
    grid = np.reshape(grid, (-1, 2))
    
    # consider class positive if prediction is > 0.5, otherwise class negative
    prediction = model.predict(grid) > .5
    prediction = prediction.reshape(x_grid.shape)
    
    # plot decision boundary and save it using the given filepath
    plt.clf()
    plt.contourf(x_grid, y_grid, prediction, cmap=plt.cm.Spectral)
    plt.scatter(features[:, 0], features[:, 1], c=target, cmap=plt.cm.Spectral)
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.title(title)
    plt.savefig(filepath, dpi=300)


def train_and_evaluate_model(model, features, target, name, epochs, batch_size):
    model.fit(x=features,
              y=target,
              validation_split=0.2,
              epochs=epochs,
              batch_size=batch_size)

    acc = accuracy_score(target, model.predict(features) > 0.5)
    print('Accuracy of {} model: {}'.format(name, acc))

    plot_decision_boundary(model=model,
                           features=features,
                           target=target,
                           xmin=-4.5,
                           xmax=4.5,
                           x_density=1000,
                           ymin=-4.5,
                           ymax=4.5,
                           y_density=1000,
                           filepath='images/problem2/{}_decision_boundary.png'.format(name),
                           title='Decision boundary: {}'.format(name))


def main():
    # alias
    conf = config.C.PROBLEM2

    # load data from .mat format
    data = scipy.io.loadmat(conf.DATASET_PATH)['data']

    # split data onto features and target
    features = data[:, :2]
    target = data[:, 2]

    scatter_plot(features, target)

    # split data into training and test set
    features_train, \
    features_test,  \
    target_train,   \
    target_test = train_test_split(features, target, test_size=0.2)

    if conf.UNDERFIT:
        underfit_model = create_underfitting_model()
        train_and_evaluate_model(model=underfit_model,
                                 features=features_train, 
                                 target=target_train,
                                 name='underfit',
                                 epochs=conf.EPOCHS,
                                 batch_size=conf.BATCH_SIZE)

    if conf.OPTIMAL:
        optimal_model = create_optimal_model()
        train_and_evaluate_model(model=optimal_model,
                                 features=features_train,
                                 target=target_train,
                                 name='optimal',
                                 epochs=conf.EPOCHS,
                                 batch_size=conf.BATCH_SIZE)

    if conf.OVERFIT:
        overfit_model = create_overfitting_model()
        train_and_evaluate_model(model=overfit_model,
                                 features=features_train,
                                 target=target_train,
                                 name='overfit',
                                 epochs=conf.EPOCHS,
                                 batch_size=conf.BATCH_SIZE)
    

if __name__ == '__main__':
    main()
