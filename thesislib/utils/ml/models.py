import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.fixes import logsumexp


class ThesisNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    The idea for this class is quite straightforward
    With Naive Bayes, we make a strong conditional independence assumption on the data that we do have
    This means the given a sample X with a set of variables {x0, x1, ...,xn} the probability that the
    sample X belongs to a class c i.e P(c | X) can be expressed as :
    P(c|X) = P(c|x0) * P(c|x1) * ... * P(c|xn)
    This allows for an individual estimation of P(c|x_i)

    However when X is a mixture of types of variables, e.g continous, categorical (with multiple categories), binomial, etc
    we need to combined the results as shown in the product above.

    This classes provides an interface for combinning different naive bayes classifiers operating on different type of variables
    """
    def __init__(self, classifier_map, classes):
        """
        The classifier map is in this format:
        classifier_map = [[clf, [list_of_keys_for_same_variable_type]], ...]

        So assuming the data is made up of two  gaussian variables say age and height, and two categorical variables say gender and race
        the classifier_map would be something like:
        classifier_map = [[clf_gauss, ['age', 'height']], [clf_cat, ['gender', 'race']]]
        where clf_gauss = sklearn.naive_bayes.GaussianNB() and clf_cat = sklearn.naive_bayes.MultinomialNB()
        :param classifier_map:
        """
        self.classifier_map = classifier_map
        self.fitted = False
        self.is_partial = False

        self.labelbin = LabelBinarizer()
        self.labelbin.fit(classes)
        self.classes_ = self.labelbin.classes_

    def _joint_log_likelihood(self, X):
        """
        Computes the join log likelihood for the different classifiers
        :param X:
        :return:
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted.")

        _probs = np.zeros((X.shape[0], self.classes.shape[0]))
        for idx in range(len(self.classifier_map)):
            clf, keys = self.classifier_map[idx]
            _df = X[keys]
            _probs += clf._joint_log_likelihood(_df)

        return _probs

    def fit(self, X, y):
        """
        Fits the different classifiers on the relevant part of the data and stores the
        fitted classifiers for further use
        :param X:
        :param y:
        :return:
        """
        _y = self.labelbin.fit_transform(y)
        for idx in range(len(self.classifier_map)):
            clf, keys = self.classifier_map[idx]
            _df = X[keys]
            clf.fit(_df, y)
            self.classifier_map[idx][0] = clf

        self.fitted = True

    def partial_fit(self, X, y):
        _y = self.labelbin.fit_transform(y)

        for idx in range(len(self.classifier_map)):
            clf, keys = self.classifier_map[idx]
            _df = X[keys]
            clf.partial_fit(_df, y, self.classes_)
            self.classifier_map[idx][0] = clf

        self.fitted = True
        self.is_partial = True

    def predict_log_proba(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted.")

        _probs = self._joint_log_likelihood(X)
        # normalize (see naive_bayes.py in sklearn for explanation!!)
        _log_prob_x = logsumexp(_probs, axis=1)
        return _probs - np.atleast_2d(_log_prob_x).T

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted.")
        jll = self._joint_log_likelihood(X)
        if type(jll) != "numpy.ndarray":
            jll = jll.values
        return self.classes_[np.argmax(jll, axis=1)]

    def serialize(self):
        return {
            "fitted": self.fitted,
            "classes": self.classes_,
            "classifier_map": self.classifier_map,
            "is_partial": self.is_partial
        }

    @staticmethod
    def load(serialized):
        if type(serialized) is not dict:
            raise ValueError("Serialized model has to be a dict")

        fitted = serialized.get('fitted', None)
        classes = serialized.get('classes', None)
        classifier_map = serialized.get('classifier_map', None)
        is_partial = serialized.get('is_partial', None)

        if not fitted or not classifier_map or not classes or not is_partial:
            raise ValueError("Missing required serialization entities")

        clf = ThesisNaiveBayes(classifier_map, classes)
        clf.fitted = fitted
        clf.is_partial = is_partial

        return clf
