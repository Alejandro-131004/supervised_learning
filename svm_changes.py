import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC  # Support Vector Classification
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.DEBUG)

def classification():
    # Generate a random binary classification problem.
    X, y = make_classification(
        n_samples=1200, n_features=10, n_informative=5, random_state=1111, n_classes=2, class_sep=1.75
    )
    # Convert y to {-1, 1}
    y = (y * 2) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

    for kernel_type in ['rbf', 'linear']:
        model = SVC(kernel=kernel_type, C=0.6, gamma=0.1 if kernel_type == 'rbf' else 'scale')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"Classification accuracy ({kernel_type}): {accuracy_score(y_test, predictions)}")

if __name__ == "__main__":
    classification()
