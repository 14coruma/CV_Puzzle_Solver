# Grab the MNIST dataset and nicely format for test/training
import tensorflow as tf

def get_data(n=1000, m=100):
    '''
    Grab MNIST (handwritten digits) from tensorflow module
    Split data, properly format lables, and return as dict
    '''
    # BEGIN: Code from https://towardsdatascience.com/alexnet-8b05c5eb88d4
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images = training_images[:n]
    training_labels = training_labels[:n]
    test_images = test_images[:m]
    test_labels = test_labels[:m]

    training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    # 80-20 Train-Validation split
    num_len_train = int(0.8 * len(training_images))

    ttraining_images = training_images[:num_len_train]
    ttraining_labels = training_labels[:num_len_train]

    valid_images = training_images[num_len_train:]
    valid_labels = training_labels[num_len_train:]

    training_images = ttraining_images
    training_labels = ttraining_labels
    # END: Code from https://towardsdatascience.com/alexnet-8b05c5eb88d4

    return {
        "X_train": training_images,
        "X_valid": valid_images,
        "X_test": test_images,
        "y_train": training_labels,
        "y_valid": valid_labels,
        "y_test": test_labels}