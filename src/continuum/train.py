import kfp
from kfp import dsl
from kfp.components import create_component_from_func

def train_model(data_path: str, model_path: str):
    import tensorflow as tf
    from tensorflow.keras import layers
    
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=data_path)
    
    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Build model
    model = tf.keras.models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Train model
    model.fit(x_train, y_train, epochs=5)
    
    # Save model
    model.save(model_path)

train_op = create_component_from_func(train_model, base_image='tensorflow/tensorflow:latest')