import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    p = np.mean(y, axis=0)
    H = -np.sum(p * np.log(p + EPS))
    return H


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    return 1 - np.sum(np.mean(y, axis=0) ** 2)


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """
    return np.var(y)


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        left, right = [], []
        n = X_subset.shape[0]
        for i in range(n):
            if X_subset[i][feature_index] < threshold:
                left.append(i)
            else:
                right.append(i)

        X_left, y_left = X_subset[left], y_subset[left]
        X_right, y_right = X_subset[right], y_subset[right]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        left, right = [], []
        n = X_subset.shape[0]
        for i in range(n):
            if X_subset[i][feature_index] < threshold:
                left.append(i)
            else:
                right.append(i)

        y_left, y_right = y_subset[left], y_subset[right]

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """

        feature_index, threshold = 0, 0
        n, m = X_subset.shape
        G = 0
        for i in range(m):
            thresholds_array = np.sort(np.unique(X_subset[:, i]))[
                               1:-1]  # np.sort(np.unique(np.random.choice(X_subset[:, i], n)))[1:-1]
            for threshold_ in thresholds_array:
                y_left, y_right = self.make_split_only_y(i, threshold_, X_subset, y_subset)
                G_ = self.criterion(y_subset) - y_left.shape[0] / n * self.criterion(y_left) - y_right.shape[
                    0] / n * self.criterion(y_right)
                if G_ > G:
                    feature_index, threshold, G = i, threshold_, G_

        return feature_index, threshold

    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        def value(y):
            if 0 == len(y):
                return 0
            if self.criterion_name in ('gini', 'entropy'):
                return np.argmax(np.sum(y, axis=0))
            if 'variance' == self.criterion_name:
                return np.mean(y)
            return np.median(y)

        def proba(y):
            if 0 == len(y):
                return 0
            if self.criterion_name in ('gini', 'entropy'):
                return np.mean(y, axis=0)
            return np.mean(y)

        self.depth += 1
        min_leaves = X_subset.shape[0]
        if self.depth < self.max_depth and min_leaves >= self.min_samples_split:
            feature_index, threshold = self.choose_best_split(X_subset, y_subset)
            new_node = Node(feature_index, threshold)
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
            new_node.left_child = self.make_tree(X_left, y_left)
            new_node.right_child = self.make_tree(X_right, y_right)
        else:
            new_node = Node(0, 0)
            new_node.value = value(y_subset)
            new_node.proba = proba(y_subset)
        self.depth -= 1

        return new_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def _recursion_predict(self, X, indexes, node, pred, proba=False):
        if not node.left_child:
            for i in indexes:
                pred[i] = node.proba if proba else node.value
        else:
            (X_left, y_left), (X_right, y_right) = self.make_split(node.feature_index, node.value, X, indexes)
            self._recursion_predict(X_left, y_left, node.left_child, pred, proba)
            self._recursion_predict(X_right, y_right, node.right_child, pred, proba)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """

        tree = self.root
        prediction = dict()
        y_predicted = np.zeros(X.shape[0])
        indexes = np.arange(X.shape[0])
        self._recursion_predict(X, indexes, tree, prediction)
        for i, value in prediction.items():
            y_predicted[i] = value
        return y_predicted

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'

        tree = self.root
        prediction = dict()
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes))
        indexes = np.arange(X.shape[0])
        self._recursion_predict(X, indexes, tree, prediction, True)
        for i, proba in prediction.items():
            y_predicted_probs[i] = proba

        return y_predicted_probs