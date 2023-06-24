import math


class NaiveBayes:
    def __init__(self, learning_rate=1.0, alpha=1.0):
        """
        Naive Bayes classifier using Gaussian distribution.

        Args:
            learning_rate (float, optional): Learning rate for probability calculation. Default is 1.0.
            alpha (float, optional): Laplace smoothing parameter. Default is 1.0.
        """
        self.alpha = alpha
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data.

        Args:
            X (list): Training data features.
            y (list): Training data labels.
        """
        n_samples, n_features = len(X), len(X[0])
        self._classes = list(set(y))
        n_classes = len(self._classes)

        self._mean = [[0] * n_features for _ in range(n_classes)]
        self._var = [[0] * n_features for _ in range(n_classes)]
        self._priors = [0] * n_classes

        for idx, c in enumerate(self._classes):
            x_c = [X[i] for i in range(n_samples) if y[i] == c]
            self._mean[idx] = [(sum(feature[i] for feature in x_c) + self.alpha) /
                               (len(x_c) + self.alpha * n_features) for i in range(n_features)]

        for idx, c in enumerate(self._classes):
            x_c = [X[i] for i in range(n_samples) if y[i] == c]
            self._var[idx] = [(sum((feature[i] - self._mean[idx][i]) ** 2 for feature in x_c) + self.alpha) /
                              (len(x_c) + self.alpha * n_features) for i in range(n_features)]

        for idx, c in enumerate(self._classes):
            self._priors[idx] = (sum(1 for label in y if label == c) + self.alpha) / (n_samples + self.alpha * n_classes)

    def predict(self, X):
        """
        Predict the labels for the input data.

        Args:
            X (list): Input data features.

        Returns:
            list: Predicted labels.
        """
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        """
        Predict the label for a single instance.

        Args:
            x (list): Input instance features.

        Returns:
            int: Predicted label.
        """
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = math.log(self._priors[idx])
            probabilities = self._pdf(idx, x)
            posterior = prior
            _posterior = 0
            for p in probabilities:
                if p > 0:
                    _posterior += math.log10(p * math.pow(10, 100))
            posteriors.append(posterior + _posterior / 100)

        return self._classes[posteriors.index(max(posteriors))]

    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function for a class and features.

        Args:
            class_idx (int): Index of the class.
            x (list): Input instance features.

        Returns:
            list: List of probabilities.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        probabilities = []
        for i in range(len(x)):
            exponent = -((x[i] - mean[i]) ** 2) / (2 * max(var[i], 1e-9))
            probability = self.learning_rate * (1.0 / (math.sqrt(2 * math.pi * max(var[i], 1e-9))) *
                                                math.exp(exponent))
            probabilities.append(probability)
        return probabilities


    
    
if __name__ == '__main__':
    pass
    
