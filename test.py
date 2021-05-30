import unittest
import pandas as pd
import numpy as np

from regressionmodels.model import LinearRegression

X = pd.DataFrame([[2, 3, 4]]).transpose()
y = pd.DataFrame([4, 6, 8])


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.linearregression = Regression()
        self.polynomialregression = Regression(deg=2)

    def test_linear(self):
        self.assertEqual(self.linearregression.fit(X, y)[1], 2.0)

    def test_polynomial(self):
        self.assertEqual(self.polynomialregression.fit(X, y)[1], 2.0)

    # can be augmented with elemtent of beta different from [1], in order to check the polynomial parameters


if __name__ == "__main__":
    unittest.main()
