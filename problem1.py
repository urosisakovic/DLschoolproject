import matplotlib.pyplot as plt
import numpy as np

import config
import tensorflow as tf
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


def split_dataset(features, target, ratio_train=0.6, ratio_val=0.2, ratio_test=0.2):
    assert ratio_train + ratio_val + ratio_test == 1, 'Sum of ratios must be 1.'
    
    features_size = features.shape[0]
    target_size = target.shape[0]
    assert features_size == target_size, 'Feature and target dataset must be of equal sizes.'

    # alias
    dataset_size = features_size # (= target_size)

    train_size = (int)(dataset_size * ratio_train)
    val_size = (int)(dataset_size * ratio_val)
    # no (feature, target) pair is discarded due to float number rounding
    test_size = dataset_size - train_size - val_size 

    # generate a random permutation
    indices = np.arange(features_size, dtype=np.int32)
    np.random.shuffle(indices)

    # shuffle datasets
    features = features[indices]
    target = target[indices]

    features_train = features[0:train_size]
    features_val = features[train_size : train_size+val_size]
    features_test = features[train_size+val_size:]

    target_train = target[0:train_size]
    target_val = target[train_size : train_size+val_size]
    target_test = target[train_size+val_size:]
    
    return features_train,  \
           features_val,    \
           features_test,   \
           target_train,    \
           target_val,      \
           target_test,     


def main():
    # alias
    conf = config.C

    sampled_points, true_function, noisy_function = generate_data(conf.A,
                                                                  conf.B,
                                                                  conf.f1,
                                                                  conf.f2)
    
    # save true function graph as an image
    plt.plot(sampled_points, true_function, 'b-', linewidth=2, alpha=1, label='true function')
    plt.savefig('images/p1_true_function.png', dpi=300)
    
    # save true and noisy function graph as an image
    plt.plot(sampled_points, noisy_function, 'r-', linewidth=2, alpha=0.8, label='noisy function')
    plt.legend(loc='upper left')
    plt.savefig('images/p1_noisy_function.png', dpi=300)

    # ML conventional aliases
    features = sampled_points
    target = noisy_function

    # train, validation, test split
    features_train, \
    features_val,   \
    features_test,  \
    target_train,   \
    target_val,     \
    target_test = split_dataset(features,
                                target,
                                ratio_train=0.6,
                                ratio_val=0.2,
                                ratio_test=0.2)

    # clear current figure
    plt.clf()
    # plot ground truth function (noisy function)
    plt.plot(features_train, target_train, 'ro', linewidth=2, alpha=1, label='training points')  
    plt.plot(features, noisy_function, 'r-', linewidth=2, alpha=1, label='true function')  
    plt.plot(features_val, target_val, 'bo', linewidth=2, alpha=1, label='validation points')   
    plt.legend(loc='upper left')
    plt.savefig('images/p1_training_val_split.png', dpi=300)


    # create a regression model
    model = Sequential()

    model.add(Dense(256, input_dim=1, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1))

    # plot_model(model, to_file='images/p1_model.png')

    train_loss = []
    val_loss = []

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='mse')
    history = model.fit(x=features_train,
                        y=target_train,
                        validation_data=(features_val, target_val),
                        epochs=5000,
                        batch_size=120)
    train_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])


    model.compile(optimizer=Adam(learning_rate=1e-6),
                  loss='mse')
    history = model.fit(x=features_train,
                        y=target_train,
                        validation_data=(features_val, target_val),
                        epochs=5000,
                        batch_size=120)
    train_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])

    plt.clf()
    plt.plot(train_loss, 'r-', label='training loss')
    plt.plot(val_loss, 'b-', label='validation loss')
    plt.xlabel('epochs')
    plt.savefig('images/p1_train_val_loss.png', dpi=300)

    pred = model.predict(sampled_points, batch_size=100)
    pred = np.squeeze(pred, axis=-1)

    # clear current figure
    plt.clf()
    # plot ground truth function (noisy function)
    plt.plot(features_train, target_train, 'ro', linewidth=2, alpha=1, label='training points')  
    plt.plot(features, noisy_function, 'r-', linewidth=2, alpha=1, label='true function')  
    plt.plot(features_val, target_val, 'bo', linewidth=2, alpha=1, label='validation points')   
    # plot model prediction and save it as an image
    plt.plot(sampled_points, pred, 'r-', linewidth=2, alpha=0.8, label='model prediction', color='green')
    plt.legend(loc='upper left')
    plt.savefig('images/p1_model_prediction.png', dpi=300)

if __name__ == '__main__':
    main()
