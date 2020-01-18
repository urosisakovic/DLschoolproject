import matplotlib.pyplot as plt
import numpy as np

import config
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical


def build_model():
    model = Sequential()

    model.add(Dense(128, input_dim=11, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    # convenience alias
    conf = config.C.PROBLEM3

    # load train data
    train_data = np.loadtxt(open('/home/uros/src/nn_project/data3/Wine/winequality-white.csv', 'rb'), delimiter=';', skiprows=1)
    print('Loaded training data with shape: ', train_data.shape)

    # split train data into features and target
    features = train_data[:, :-1]
    target = train_data[:, -1]

    target -= 3

    # plot train data class distribution
    plt.hist(target, bins=np.arange(10) - .5, rwidth=0.8)
    plt.xticks(range(9))
    plt.xlabel('Class')
    plt.ylabel('Training samples per class')
    plt.savefig('images/temp/class_distribution.png', dpi=300)
    plt.clf()

    target = to_categorical(target)

    model = build_model()

    history = model.fit(x=features,
                        y=target,
                        validation_split=0.2,
                        epochs=500,
                        batch_size=64,
                        verbose=1)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot training and validation losses
    plt.clf()
    plt.plot(train_loss, 'r-', label='training loss')
    plt.plot(val_loss, 'b-', label='validation loss')
    plt.xlabel('epochs')
    plt.title('Train and validation loss')
    plt.legend(loc='upper right')
    plt.savefig('images/temp/model_loss.png', dpi=300)

    pred = model.predict(features)
    pred = np.rint(pred)

    test_acc = accuracy_score(target, pred)

    print('Test accuracy: {}'.format(test_acc))

if __name__ == '__main__':
    main()
