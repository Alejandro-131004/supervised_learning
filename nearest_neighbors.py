try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from distance import euclidean_distance

import knn
from metrics import accuracy


def classification():
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        random_state=1111,
        class_sep=1.5,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1111)

    clf = knn.KNNClassifier(k=5, distance_func=euclidean_distance)

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("classification accuracy", accuracy(y_test, predictions))


if __name__ == "__main__":
    classification()