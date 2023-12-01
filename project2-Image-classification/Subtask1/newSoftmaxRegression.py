import numpy as np
from tqdm import tqdm


class newSoftmaxRegression:
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=100, random_seed=None):
        """
        Initialize the Multinomial Logistic Regression model.

        Parameters:
        - num_classes: The number of classes for classification.
        - learning_rate: The learning rate for gradient descent (default is 0.01).
        - num_iterations: The number of training iterations (default is 100).
        - random_seed: int, random seed for reproducibility (default: None)
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.weights = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, decay_rate=0.999, regularization_strength=0.01):
        """
        Train the Multinomial Logistic Regression model.

        Parameters:
        - X_train: Training feature data.
        - y_train: Training labels.
        - X_val: Validation feature data (optional).
        - y_val: Validation labels (optional).

        Returns:
        - train_losses: List of training losses during iterations.
        - train_accuracies: List of training accuracies during iterations.
        - val_losses: List of validation losses during iterations (if validation data provided).
        - val_accuracies: List of validation accuracies during iterations (if validation data provided).
        """
        # Add bias term to the training data
        X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

        # Initialize weights with random values
        np.random.seed(self.random_seed)
        self.weights = np.random.randn(X_train_bias.shape[1], self.num_classes)

        # Lists to store training and validation losses during training
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for iteration in tqdm(range(self.num_iterations)):
            # Create mini-batch
            for batch_start in range(0, X_train_bias.shape[0], batch_size):
                batch_end = batch_start + batch_size
                X_batch = X_train_bias[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]

                # Calculate logits and softmax probabilities
                logits = np.dot(X_batch, self.weights)
                exp_logits = np.exp(logits)
                softmax_probs = exp_logits / \
                    np.sum(exp_logits, axis=1, keepdims=True)

                # Compute the cross-entropy loss
                loss = -np.mean(y_batch * np.log(softmax_probs))

                # Compute the gradient and update weights
                gradient = (np.dot(X_batch.T, softmax_probs -
                            y_batch) / X_batch.shape[0])
                self.weights -= self.learning_rate * gradient

                # # Compute the cross-entropy loss with L2 regularization
                # loss = -np.mean(y_batch * np.log(softmax_probs)) + 0.5 * regularization_strength * np.sum(self.weights**2)

                # # Compute the gradient with L2 regularization
                # gradient = (np.dot(X_batch.T, softmax_probs - y_batch) + regularization_strength * self.weights) / X_batch.shape[0]

                # Update weights with learning rate schedule
                # self.learning_rate = self.initial_learning_rate * np.exp(-iteration / 1000)
                self.learning_rate *= decay_rate

            # Calculate training accuracy
            train_pred = np.argmax(softmax_probs, axis=1)
            train_accuracy = np.mean(train_pred == np.argmax(y_batch, axis=1))

            train_accuracies.append(train_accuracy)

            # train_losses.append(loss)
            if iteration % 200 == 0:
                print(f"Iteration: {iteration}, Loss: {loss}")

            if X_val is not None and y_val is not None:
                # Calculate validation loss
                X_val_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
                logits_val = np.dot(X_val_bias, self.weights)
                exp_logits_val = np.exp(logits_val)
                softmax_probs_val = exp_logits_val / \
                    np.sum(exp_logits_val, axis=1, keepdims=True)
                # val_loss = -np.mean(y_val * np.log(softmax_probs_val))

                # Calculate validation accuracy
                val_pred = np.argmax(softmax_probs_val, axis=1)
                val_accuracy = np.mean(val_pred == np.argmax(y_val, axis=1))

                # val_losses.append(val_loss)
                # if iteration % 10 == 0:
                #     val_losses[iteration] = val_loss
                val_accuracies.append(val_accuracy)

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Feature data for prediction.

        Returns:
        - predicted_class: Predicted class labels.
        """
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        logits = np.dot(X_bias, self.weights)
        predicted_class = np.argmax(logits, axis=1)
        return predicted_class
