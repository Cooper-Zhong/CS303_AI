import numpy as np
from tqdm import tqdm


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)


def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == y_pred)


class NeuralNetwork:
    def __init__(self, num_classes, hidden_units, learning_rate=0.01, num_iterations=20, random_seed=None, decay_rate=0.99, dropout_prob=0.2):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.dropout_prob = dropout_prob
        self.dropout_mask = None
        self.decay_rate = decay_rate
        self.hidden_units = hidden_units
        self.weights1 = None
        self.biases1 = None
        self.weights2 = None
        self.biases2 = None

    def initialize_weights(self, input_size):
        np.random.seed(self.random_seed)
        self.weights1 = np.random.randn(input_size, self.hidden_units)
        self.biases1 = np.zeros((1, self.hidden_units))
        self.weights2 = np.random.randn(self.hidden_units, self.num_classes)
        self.biases2 = np.zeros((1, self.num_classes))

    def build_model(self, input_size):
        self.initialize_weights(input_size)

    def forward_pass(self, X, training=True):
        # Layer 1
        z1 = np.dot(X, self.weights1) + self.biases1
        a1 = relu(z1)

        # # Dropout layer
        # if training:
        #     self.dropout_mask = np.random.binomial(1, 1 - self.dropout_prob, size=a1.shape)
        #     a1 *= self.dropout_mask / (1 - self.dropout_prob)

        # Layer 2
        z2 = np.dot(a1, self.weights2) + self.biases2
        a2 = softmax(z2)

        return a1, a2

    def backward_pass(self, X, y, a1, a2):
        m = len(X)

        # Output layer
        dz2 = a2 - y
        dw2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer
        dz1 = np.dot(dz2, self.weights2.T) * (a1 > 0)
        # # dropout
        # dz1 *= self.dropout_mask / (1 - self.dropout_prob)

        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.weights2 -= self.learning_rate * dw2
        self.biases2 -= self.learning_rate * db2
        self.weights1 -= self.learning_rate * dw1
        self.biases1 -= self.learning_rate * db1

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        input_size = X_train.shape[1]
        self.build_model(input_size)

        loss_history = []
        val_loss_history = []
        accuracy_history = []
        val_accuracy_history = []

        for epoch in range(self.num_iterations):
            total_loss = 0
            total_accuracy = 0
            # decay learning rate
            self.learning_rate *= self.decay_rate

            # for batch_start in tqdm(range(0, len(X_train), 16), desc=f'Epoch {epoch + 1}', position=None, leave=True):
            for batch_start in range(0, len(X_train), 16):
                batch_end = batch_start + 16
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]

                # Forward pass
                a1, a2 = self.forward_pass(X_batch)

                # Compute loss and accuracy
                loss = categorical_crossentropy(y_batch, a2)
                accuracy_batch = accuracy(y_batch, np.argmax(a2, axis=1))

                total_loss += loss
                total_accuracy += accuracy_batch

                # Backward pass
                self.backward_pass(X_batch, y_batch, a1, a2)

            # Average loss and accuracy for the epoch
            avg_loss = total_loss / (len(X_train) // 16)
            avg_accuracy = total_accuracy / (len(X_train) // 16)

            loss_history.append(avg_loss)
            accuracy_history.append(avg_accuracy)

            # Validation

            _, a2_val = self.forward_pass(X_val)
            val_loss = categorical_crossentropy(y_val, a2_val)
            val_accuracy = accuracy(y_val, np.argmax(a2_val, axis=1))

            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

            if epoch % 10 == 0:
                print(
                    f'Epoch {epoch + 1}: accuracy = {avg_accuracy}, val_accuracy = {val_accuracy}')

        return loss_history, val_loss_history, accuracy_history, val_accuracy_history

    def predict(self, X):
        _, a2 = self.forward_pass(X, training=False)
        predicted_class = np.argmax(a2, axis=1)
        return predicted_class
