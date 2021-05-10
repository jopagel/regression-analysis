import numpy as np
import pandas as pd


class LinearRegression:
    """Class for a Regression Analysis

    Attributes
    ----------
    deg : int
            The degree of polynoms used in the model
    """
    def __init__(self, deg = 1):
        self.deg = deg


    def fit(self, X, y):
        """Class for a Regression Analysis

            Parameters
            ----------
            X : DataFrame
                    The input feature matrix without threshold column
            y : DataFrame
                    The dependent variable

            Returns
            ----------

            """
        X = X.insert(0,0)
        X_t = X.t

        beta = np.linalg.inv(X_t.dot(X)) * X_t*y

        beta = X.dot(X.T)


    def predict(self):
        pass

    def __repr__(self):
        pass



