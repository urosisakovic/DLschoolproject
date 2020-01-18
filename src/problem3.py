import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import config


def build_model():
    model = Sequential()

    reg = 0.001

    model.add(Dense(256, input_dim=16, activation='relu', activity_regularizer=regularizers.l2(reg)))
    model.add(Dense(256, activation='relu', activity_regularizer=regularizers.l2(reg)))
    model.add(Dense(256, activation='relu', activity_regularizer=regularizers.l2(reg)))
    model.add(Dense(10, activation='softmax', activity_regularizer=regularizers.l2(reg)))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def cross_validation(model_fn, features, target):
    estimator = KerasClassifier(build_fn=model_fn, epochs=10, batch_size=5, verbose=1)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, features, target, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


def plot_confusion_matrix(model, features, target, mode):
    filepath = 'images/problem3/{}_confusion_matrix.png'.format(mode)

    prob_pred = model.predict(features)
    prob_pred = np.argmax(prob_pred, axis=1)

    target = np.argmax(target, axis=1)

    # plot confusion matrix
    cm = np.zeros([10, 10])
    for i in range(10):
        for j in range(10):
            for k in range(len(target)):
                if target[k] == i and prob_pred[k] == j:
                    cm[i, j] += 1

    cm = cm / np.sum(cm)
    cm *= 100 # convert to percentage

    plt.clf()
    plt.matshow(cm, cmap=plt.cm.Blues)
    for i in range(10):
        for j in range(10):
                plt.text(i, j, str(round(cm[i, j], 2)) + "%", va='center', ha='center', fontsize=8)
    plt.title('Confusion matrix: {} data'.format(mode))
    
    step_x = int(10 / (10 - 1)) # step between consecutive labels
    x_positions = np.arange(0, 10, step_x) # pixel count at label position
    x_labels = [i for i in range(10)] # labels you want to see
    
    plt.xticks(x_positions, x_labels)
    plt.yticks(x_positions, x_labels)
    plt.ylim((9.5, -0.5))

    plt.savefig(filepath, dpi=300)




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
    plt.clf()
    plt.hist(target_train, bins=np.arange(11) - .5, rwidth=0.8)
    plt.xticks(range(10))
    plt.xlabel('Class')
    plt.ylabel('Training samples per class')
    plt.title('Training set class distribution')
    plt.savefig('images/problem3/training_class_distribution.png', dpi=300)
    
    # load test data
    test_data = np.loadtxt(open(conf.TEST_DATASET_PATH, 'rb'), delimiter=',')
    print('Loaded test data with shape: ', test_data.shape)

    # split test data into features and target
    features_test = test_data[:, :-1]
    target_test = test_data[:, -1]

    # plot test data class distribution
    plt.clf()
    plt.hist(target_test, bins=np.arange(11) - .5, rwidth=0.8)
    plt.xticks(range(10))
    plt.xlabel('Class')
    plt.ylabel('Test samples per class')
    plt.title('Test set class distribution')
    plt.savefig('images/problem3/test_class_distribution.png', dpi=300)


    model = build_model()
    target_train = to_categorical(target_train)
    target_test = to_categorical(target_test)
    history = model.fit(x=features_train,
                        y=target_train,
                        validation_data=(features_test, target_test),
                        epochs=conf.EPOCHS,
                        batch_size=conf.BATCH_SIZE,
                        verbose=1)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot training and validation losses
    plt.clf()
    plt.plot(train_loss, 'r-', label='training loss')
    plt.plot(val_loss, 'b-', label='validation loss')
    plt.xlabel('epochs')
    plt.title("Train and validation loss")
    plt.legend(loc='upper right')
    plt.savefig('images/problem3/model_loss.png', dpi=300)

    filepath_pattern = 'images/problem3/{}_{}_confusion_matrix.png'

    plot_confusion_matrix(model=model,
                          features=features_train,
                          target=target_train,
                          mode='train')

    plot_confusion_matrix(model=model,
                          features=features_test,
                          target=target_test,
                          mode='test')

if __name__ == '__main__':
    main()
