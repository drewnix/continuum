from kfp.v2.dsl import component, Input, Output, Dataset, Model

# Define component to train MNIST model
@component(base_image='tensorflow/tensorflow')
def train_model(data: Input[Dataset], output_model: Output[Dataset]):
    import numpy as np
    import tensorflow as tf
    
    # Load the data
    with np.load(data.path) as data:
        x_train, y_train = data['x_train'], data['y_train']
    
    # Normalize the data
    x_train = x_train.astype('float32') / 255
    
    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, epochs=5)
    
    # Save the model
    model.save(output_model.path)
