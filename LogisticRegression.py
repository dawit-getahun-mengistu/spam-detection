import math

# 
def sigmoid(z):
    """
    Sigmoid function to compute the activation.

    Args:
        z (float): Input value.

    Returns:
        float: Sigmoid value.
    """
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10):
        """
        Initialize Logistic Regression.

        Args:
            learning_rate (float, optional): Learning rate for gradient descent. Default is 0.01.
            num_iterations (int, optional): Number of iterations for training. Default is 10.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = {}
        self.bias = {}
        self.labels = [i for i in range(10)]
        
        for label in self.labels:
            self.weights[label] = []
            self.bias[label] = 0

    def fit(self, X, y):
        """
        Fit the Logistic Regression model to the training data.

        Args:
            X (list): Training data features.
            y (list): Training data labels.
        """
        num_features = len(X[0])
        
        for label in self.labels:
            self.weights[label] = [0] * num_features
        
        y_transformed = [self._transform_label(label) for label in y]
        y_classified = list(zip(*y_transformed))
        
        for label in self.labels:
            self._fit_label(X, y_classified[label], label)

    def predict(self, X):
        """
        Predict the labels for the input data.

        Args:
            X (list): Input data features.

        Returns:
            list: Predicted labels.
        """
        predictions = []
        
        for x in X:
            x_prediction = {}
            
            for label in self.labels:
                label_prediction = self._predict_label(label, x)
                x_prediction[label] = label_prediction
            
            prediction = max(x_prediction, key=x_prediction.get)
            predictions.append(prediction)
        
        return predictions

    def _transform_label(self, y):
        """
        Transform the labels into binary representation.

        Args:
            y (int): Label value.

        Returns:
            list: Binary representation of the label.
        """
        transform_vector = []
        
        for label in self.labels:
            if label == y:
                transform_vector.append(1)
            else:
                transform_vector.append(0)
        
        return transform_vector

    def _predict_label(self, label, x):
        """
        Predict the label for a single instance.

        Args:
            label (int): Label value.
            x (list): Input instance features.

        Returns:
            float: Predicted label probability.
        """
        z = sum([self.weights[label][i] * x[i] for i in range(len(x))]) + self.bias[label]
        return sigmoid(z)

    def _fit_label(self, X, y_label, label):
        """
        Fit the Logistic Regression model for a specific label.

        Args:
            X (list): Training data features.
            y_label (list): Binary representation of the label.
            label (int): Label value.
        """
        num_samples = len(y_label)
        num_features = len(X[0])
        
        for _ in range(self.num_iterations):
            for i in range(num_samples):
                predicted_y = self._predict_label(label, X[i])
                error = y_label[i] - predicted_y
                
                for j in range(num_features):
                    self.weights[label][j] += self.learning_rate * error * X[i][j]
                
                self.bias[label] += self.learning_rate * error
