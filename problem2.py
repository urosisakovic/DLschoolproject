import matplotlib.pyplot as plt
import numpy as np

import config
import scipy.io


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

    positive_target = np.zeros((positive_features.shape[0], 1))
    positive_inputs = np.concatenate([positive_features, positive_target], axis=1)
    np.random.shuffle(positive_inputs)

    inputs = np.concatenate([negative_inputs, positive_inputs], axis=0)
    np.random.shuffle(inputs)

    features = inputs[:, :2]
    target = inputs[:, 2]

    return features, target


def main():
    # alias
    conf = config.C.PROBLEM2

    features, target = plot_and_prep_data(conf.DATASET_PATH)
    

if __name__ == '__main__':
    main()
