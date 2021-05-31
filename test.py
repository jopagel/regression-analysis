import unittest
import pandas as pd

from regressionmodels.model import Regression

X = pd.DataFrame([2, 3, 4])
y = pd.DataFrame([6, 9, 12])
X_test = pd.DataFrame([3,4,5])


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.linearregression = Regression()
        self.polynomialregression = Regression(deg=2)

    def test_linear_fit(self):
        self.assertEqual(self.linearregression.fit(X, y)[1], 3.0)

    def test_polynomial_fit(self):
        self.assertEqual(self.polynomialregression.fit(X, y)[0], 0.0)

if __name__ == "__main__":
    unittest.main()
