import matplotlib.pyplot as plt
import numpy as np

import config
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def build_model():
    model = Sequential()

    model.add(Dense(256, input_dim=16, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
	model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    # convenience alias
    conf = config.C.PROBLEM3

    # load train data
    train_data = np.loadtxt(open(conf.TRAIN_DATASET_PATH, 'rb'), delimiter=',')
    print('Loaded training data with shape: ', train_data.shape)

    # split train data into features and target
    features_train = train_data[:, :-1]
    target_train = train_data[:, -1]

    # plot train data class distribution
    plt.hist(target_train, bins=np.arange(11) - .5, rwidth=0.8)
    plt.xticks(range(10))
    plt.xlabel('Class')
    plt.ylabel('Training samples per class')
    plt.savefig('images/p3_train_class_distribution.png', dpi=300)
    plt.clf()

    # load test data
    test_data = np.loadtxt(open(conf.TEST_DATASET_PATH, 'rb'), delimiter=',')
    print('Loaded test data with shape: ', test_data.shape)

    # split test data into features and target
    features_test = test_data[:, :-1]
    target_test = test_data[:, -1]

    estimator = KerasClassifier(build_fn=build_model, epochs=200, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, features_train, target_train, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


if __name__ == '__main__':
    main()
