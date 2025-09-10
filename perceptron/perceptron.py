"""
Perceptron implementation adapted from:

Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili.
*Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing, 2022.
Source code: https://github.com/rasbt/machine-learning-book

Modifications:
- Added `fit_break` method for early stopping (stop training once convergence is reached)
- Added execution time measurement in `fit` and `fit_break`

Author of modifications: Lucas Campagnaro
"""

__author__ = "Lucas Campagnaro"

import numpy as np
import time

class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        
        self.errors_ = []

        start = time.time()
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        end = time.time()

        print('Perceptron.fit() --> Time elapsed:', (end - start) * 1000, 'milliseconds')
        
        return self

    def fit_break(self, X, y):
        """Like fit() but stopping at convergence"""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        
        self.errors_ = []

        start = time.time()
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)

            self.errors_.append(errors)
            
            # --- BEGIN: early stopping tweak ---
            if self.errors_[-1] == 0:
                break
            # --- END: early stopping tweak ---
        end = time.time()

        print('Perceptron.fit_break() --> Time elapsed:', (end - start) * 1000, 'milliseconds')
        
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)