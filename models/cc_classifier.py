import pickle
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class CCClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, order=None):
        self.base_estimator = (
            base_estimator if base_estimator is not None else LogisticRegression()
        )
        self.order = order
        self.model = ClassifierChain(self.base_estimator, order=self.order)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Base estimator does not support predict_proba.")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
