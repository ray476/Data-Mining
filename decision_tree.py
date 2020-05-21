import numpy as np

class Node:
    """A decision tree node."""

    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

# based on https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/tree/_classes.py#L585
# and https://ysu1989.github.io/courses/sp20/cse5243/Classification-BasicConcepts.pdf
class tree_classifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, features, labels):
        self.n_classes_ = len(set(labels))  # labels are assumed to go from 0 to n-1
        self.n_features_ = features.shape[1]
        self.tree_ = self._grow_tree(features, labels)

    def predict(self, features):
        return [self._predict(inputs) for inputs in features]

    def _gini(self, labels):
        return 1.0 - sum((np.sum(labels == labels.size) / labels.size) ** 2 for c in range(self.n_classes_))

    def _best_split(self, features, labels):
        if labels.size <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(labels == c) for c in range(self.n_classes_)]

        # Gini of current node with no split
        best_gini = 1.0 - sum((n / labels.size) ** 2 for n in num_parent)
        best_index, best_threshold = None, None

        # Loop through all features to determine best split
        for index in range(self.n_features_):
            # Sort data along selected feature.  threshold contains the count for the given feature(word).
            # Labels is the label of the document
            thresholds, labels = zip(*sorted(zip(features[:, index], labels)))

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, labels.size):  # try sliding threshold (27-34)
                c = labels[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (labels.size - i)) ** 2 for x in range(self.n_classes_)
                )

                # weighted average as on slide 21
                gini = (i * gini_left + (labels.size - i) * gini_right) / labels.size

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_index = index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_index, best_threshold

    def _grow_tree(self, features, labels, depth=0):
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = [np.sum(labels == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(labels), num_samples=labels.size, num_samples_per_class=num_samples_per_class, predicted_class=predicted_class,)

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            index, thr = self._best_split(features, labels)
            if index is not None:
                indices_left = features[:, index] < thr
                X_left, y_left = features[indices_left], labels[indices_left]
                X_right, y_right = features[~indices_left], labels[~indices_left]
                node.feature_index = index
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
    from sklearn.tree import export_graphviz
    from sklearn.utils import Bunch


    # 1. Load dataset.
    # dataset = load_iris()

        # https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
    df = pd.read_csv("wifi.txt", delimiter="\t")
    data = df.to_numpy()
    dataset = Bunch(
        data=data[:, :-1],
        target=data[:, -1] - 1,
        feature_names=["Wifi {}".format(i) for i in range(1, 8)],
        target_names=["Room {}".format(i) for i in range(1, 5)],
    )
    features, labels = dataset.data, dataset.target

    # 2. Fit decision tree.
    clf = tree_classifier(max_depth=2)
    clf.fit(features, labels)
    clf2 = SklearnDecisionTreeClassifier(max_depth=2)
    clf2.fit(features, labels)

    # 3. Predict.
    # input = [0, 0, 5.0, 1.5]
    input = [-70, 0, 0, 0, -40, 0, 0]
    pred = clf.predict([input])[0]
    print("Input: {}".format(input))
    print("Prediction: " + dataset.target_names[pred])

    pred2 = clf2.predict([input])[0]
    print("Prediction2: " + dataset.target_names[pred])