import logging

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

from ensemble import RandomForestClassifier, RandomForestRegressor
from metrics import mean_squared_error

logging.basicConfig(level=logging.DEBUG)


def classification():
    # Generate a random binary classification problem.
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=10, random_state=1111, n_classes=2, class_sep=2.5, n_redundant=0
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1111)

    model = RandomForestClassifier(n_estimators=10, max_depth=4)
    model.fit(X_train, y_train)

    predictions_prob = model.predict(X_test)[:, 1]
    predictions = np.argmax(model.predict(X_test), axis=1)
    #print(predictions.shape)
    print("classification, roc auc score: %s" % roc_auc_score(y_test, predictions_prob))
    print("classification, accuracy score: %s" % accuracy_score(y_test, predictions))



if __name__ == "__main__":
    classification()
