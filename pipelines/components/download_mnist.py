from kfp.v2.dsl import component, Output, Dataset

# # Define component to download MNIST data
@component(base_image='tensorflow/tensorflow')
def download_mnist(output_data: Output[Dataset]):
    import tensorflow as tf
    import numpy as np
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Save the data
    np.savez(output_data.path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
