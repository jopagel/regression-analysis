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

    def fit(X, y):
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
        nr_columns = len(X.columns)
        if deg > 1:
            X_pol = pd.DataFrame()
            for d in range(deg):
                for i in range(nr_columns):
                    X_pol.insert(len(X_pol.columns), f"f{i + 1}deg{d + 1}", X.iloc[:, i] ** (d + 1))
        else:
            X_pol = X

        X_pol.insert(0, "constant", 1)

        X_t = X_pol.transpose()

        print(X_pol)

        beta = np.array(np.linalg.inv(X_t.dot(X_pol)).dot(X_t.dot(y)))

        return beta

    def predict(self):
        pass
