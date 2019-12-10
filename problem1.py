import matplotlib.pyplot as plt
import numpy as np
import config

from dnn import DNN

def generate_data(A, B, f1, f2):
    x = np.linspace(0, 0.7, 200)
    h = A * np.sin(2 * np.pi * f1 * x) \
        + B * np.sin(2 * np.pi* f2 *x)

    y = h + np.random.normal(scale=0.2 * min(A, B), size=h.shape)

    return x, h, y

def split_dataset(dataset, ratio_train=0.6, ratio_val=0.2, ratio_test=0.2):
    pass

def main():
    # alias
    conf = config.C

    input, true_function, noisy_function = generate_data(conf.A,
                                                         conf.B,
                                                         conf.f1,
                                                         conf.f2)
    
    # TODO: Add legend on graphs
    # save true function graph as an image
    plt.plot(input, true_function, 'b-', linewidth=2, alpha=1)
    plt.savefig('problem1_1.png', dpi=300)
    
    # save true and noisy function graph as an image
    plt.plot(input, noisy_function, 'r-', linewidth=2, alpha=0.8)
    plt.savefig('problem1_2.png', dpi=300)

    quit()

    # ML conventional aliases
    features = input
    target = noisy_function

    # train, validation, test split
    features_train, features_val, features_test = split_dataset(features)
    target_train, target_val, target_test = split_dataset(target)

    # create a regression model
    regressor = DNN()
    regressor.build_model()
    train_loss, val_loss, test_loss = regressor.train(features_train,
                                                      target_train,
                                                      features_val,
                                                      target_val,
                                                      features_test,
                                                      target_test,
                                                      conf.PROBLEM1.steps)

    # plot train, validation and test losses
    plt.clf()
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.plot(test_loss)

    # calculate learned function
    learned_function = regressor.predict(features)

    # plot function learned by the model
    plt.clf()  
    plt.plot(features, target)
    plt.plot(features, learned_function)

if __name__ == '__main__':
    main()