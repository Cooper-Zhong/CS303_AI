import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork:
    def __init__(self, num_classes, hidden_units, learning_rate=0.01, num_iterations=20, random_seed=None):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.hidden_units = hidden_units
        self.model = None

    def build_model(self, input_size):

        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(self.hidden_units, activation='relu'),
            # tf.keras.layers.Dense(32),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(# optimizer='adam',
                      optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        
        input_size = X_train.shape[1]

        self.model = self.build_model(input_size)

        history = self.model.fit(
            X_train, y_train,
            epochs=self.num_iterations,
            batch_size=32,  # You can adjust the batch size
            validation_data=(X_val, y_val) if X_val is not None else None,
            verbose=1  # Set to 1 for progress bar
        )

        return history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy']

    def predict(self, X):
        predicted_probabilities = self.model.predict(X)
        predicted_class = np.argmax(predicted_probabilities, axis=1)
        return predicted_class
