import numpy as np
import pandas as pd


class LinearRegression:
    """Class for a Regression Analysis

    Attributes
    ----------
    deg : int
            The degree of polynoms used in the model
    """

    def __init__(self, deg=1):
        self.deg = deg

    def fit(self, X, y):
        """Estimates the Regression parameters beta

        Parameters
        ----------
        X : DataFrame
                The input feature matrix without threshold column
        y : DataFrame
                The dependent variable

        Returns
        ----------
        beta : 1-dim DataFrame with Regression parameters

        """
        X.insert(0, "constant", 1)
        X_t = X.transpose()

        beta = np.array(np.linalg.inv(X_t.dot(X)).dot(X_t.dot(y)))

        return beta

    def predict(self):
        pass
