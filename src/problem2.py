import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

import config


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

    model.add(Dense(2, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    return model


def create_optimal_model():
    model = Sequential()

    model.add(Dense(16, input_dim=2, activation='relu'))
    model.add(Dense(16, activation='relu'))
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


def plot_confusion_matrix(model, features, target, name, mode):
    filepath = 'images/problem2/{}_{}_confusion_matrix.png'.format(name, mode)

    prob_pred = model.predict(features)
    class_pred = prob_pred > 0.5

    # plot confusion matrix
    cm = np.array(confusion_matrix(target, class_pred))
    cm = cm / np.sum(cm)
    cm *= 100 # convert to percentage
    tp = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print('Confusion matrix for {} model on {} data:\n {}'.format(name, mode, cm))
    print('Precision for {} model on {} data: {}'.format(name, mode, precision))
    print('Recall for {} model on {} data: {}\n'.format(name, mode, recall))

    plt.clf()
    plt.matshow(cm, cmap=plt.cm.Blues)
    for i in range(2):
        for j in range(2):
                plt.text(i, j, str(cm[i][j]) + "%", va='center', ha='center')
    plt.title('Confusion matrix: {} model, {} data'.format(name, mode))
    plt.savefig(filepath, dpi=300)


def train_and_evaluate_model(model, features, target, name, epochs, batch_size):
    # split dataset onto train, validation and test part
    features_train, \
    features_test,  \
    target_train,   \
    target_test = train_test_split(features, target, test_size=0.2)
    
    history = model.fit(x=features_train,
                        y=target_train,
                        validation_data=(features_test, target_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot training and validation losses
    plt.clf()
    plt.plot(train_loss, 'r-', label='training loss')
    plt.plot(val_loss, 'b-', label='validation loss')
    plt.xlabel('epochs')
    plt.title("Train and validation loss: {}".format(name))
    plt.legend(loc='upper right')
    plt.savefig('images/problem2/{}_model_loss.png'.format(name), dpi=300)

    plot_decision_boundary(model=model,
                           features=features,
                           target=target,
                           xmin=-4.5,
                           xmax=4.5,
                           x_density=1000,
                           ymin=-5.5,
                           ymax=5.5,
                           y_density=1000,
                           filepath='images/problem2/{}_decision_boundary.png'.format(name),
                           title='Decision boundary: {}'.format(name))

    plot_confusion_matrix(model=model,
                          features=features_train,
                          target=target_train,
                          name=name,
                          mode='train')

    plot_confusion_matrix(model=model,
                          features=features_test,
                          target=target_test,
                          name=name,
                          mode='test')


def main():
    # alias
    conf = config.C.PROBLEM2

    # load data from .mat format
    data = scipy.io.loadmat(conf.DATASET_PATH)['data']

    # split data onto features and target
    features = data[:, :2]
    target = data[:, 2]

    scatter_plot(features, target)

    if conf.UNDERFIT:
        underfit_model = create_underfitting_model()
        train_and_evaluate_model(model=underfit_model,
                                 features=features, 
                                 target=target,
                                 name='underfit',
                                 epochs=conf.EPOCHS,
                                 batch_size=conf.BATCH_SIZE)

    if conf.OPTIMAL:
        optimal_model = create_optimal_model()
        train_and_evaluate_model(model=optimal_model,
                                 features=features,
                                 target=target,
                                 name='optimal',
                                 epochs=conf.EPOCHS,
                                 batch_size=conf.BATCH_SIZE)

    if conf.OVERFIT:
        overfit_model = create_overfitting_model()
        train_and_evaluate_model(model=overfit_model,
                                 features=features,
                                 target=target,
                                 name='overfit',
                                 epochs=conf.EPOCHS,
                                 batch_size=conf.BATCH_SIZE)
    

if __name__ == '__main__':
    main()
