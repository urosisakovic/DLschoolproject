import matplotlib.pyplot as plt
import numpy as np

import config
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def generate_data(A, B, f1, f2):
    x = np.linspace(0, 0.7, 240)
    
    # true function
    h = A * np.sin(2 * np.pi * f1 * x) \
        + B * np.sin(2 * np.pi* f2 *x)

    # function with added gaussian noise
    y = h + np.random.normal(scale=0.2*min(A, B), size=h.shape)

    return x, h, y


def create_regression_model():
    model = Sequential()

    model.add(Dense(256, input_dim=1, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1))

    return model


def main():
    # convenience alias
    conf = config.C.PROBLEM1

    sampled_points, true_function, noisy_function = generate_data(conf.A,
                                                                  conf.B,
                                                                  conf.f1,
                                                                  conf.f2)
    
     # save true and noisy function graph as an image
    plt.plot(sampled_points, true_function, 'b-', linewidth=2, alpha=1, label='true function')    
    plt.plot(sampled_points, noisy_function, 'r-', linewidth=2, alpha=0.8, label='noisy function')
    plt.legend(loc='upper left')
    plt.savefig('images/problem1/noisy_function.png', dpi=300)

    # ML conventional aliases
    features = sampled_points
    target = noisy_function

    # split dataset onto train, validation and test part
    features_train, \
    features_test,  \
    target_train,   \
    target_test = train_test_split(features, target, test_size=0.2)

    features_train, \
    features_val,   \
    target_train,   \
    target_val  = train_test_split(features_train, target_train, test_size=0.25)
    
    # plot ground truth function (noisy function)
    plt.clf()
    plt.plot(features_train, target_train, 'ro', linewidth=2, alpha=1, label='training points')  
    plt.plot(features, noisy_function, 'r-', linewidth=2, alpha=1, label='true function')  
    plt.plot(features_val, target_val, 'bo', linewidth=2, alpha=1, label='validation points')   
    plt.legend(loc='upper left')
    plt.savefig('images/problem1/training_val_split.png', dpi=300)

    # uncompiled model
    model = create_regression_model()
    plot_model(model, to_file='images/problem1/model.png')

    # these lists will hold train and validation losses per epoch
    train_loss = []
    val_loss = []

    # compile and fit the model
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='mse')
    history = model.fit(x=features_train,
                        y=target_train,
                        validation_data=(features_val, target_val),
                        epochs=conf.COARSE_EPOCHS,
                        batch_size=conf.BATCH_SIZE)
    # store training and validation losses
    train_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])

    # recompile the model in order to use smaller learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='mse')
    history = model.fit(x=features_train,
                        y=target_train,
                        validation_data=(features_val, target_val),
                        epochs=conf.FINE_EPOCHS,
                        batch_size=conf.BATCH_SIZE)
    train_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])

    # plot training and validation losses
    plt.clf()
    plt.plot(train_loss, 'r-', label='training loss')
    plt.plot(val_loss, 'b-', label='validation loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.savefig('images/problem1/train_val_loss.png', dpi=300)

    # find predictions for every point the dataset
    pred = model.predict(sampled_points, batch_size=conf.BATCH_SIZE)
    pred = np.squeeze(pred, axis=-1)

    # plot function trying to be fitted and function which the model has learneds
    plt.clf()
    # plt.plot(features_train, target_train, 'ro', linewidth=2, alpha=1, label='training points')  
    plt.plot(features, noisy_function, 'r-', linewidth=2, alpha=1, label='true function')  
    # plt.plot(features_val, target_val, 'bo', linewidth=2, alpha=1, label='validation points')   
    plt.plot(sampled_points, pred, 'r-', linewidth=2, alpha=0.8, label='model prediction', color='green')
    plt.legend(loc='upper left')
    plt.savefig('images/problem1/model_prediction.png', dpi=300)

    plt.clf()
    plt.plot(sampled_points, pred, 'r-', linewidth=2, alpha=0.8, label='model prediction', color='green')
    plt.savefig('images/problem1/regression_curve.png', dpi=300)

if __name__ == '__main__':
    main()
