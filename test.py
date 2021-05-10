import unittest
import pandas as pd
import numpy as np

from regressionmodels.model import LinearRegression

X = pd.DataFrame([[2, 3, 4]]).transpose()
y = pd.DataFrame([4, 6, 8])


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.linearregression = LinearRegression()

    def test_linear(self):
        self.assertEqual(self.linearregression.fit(X, y)[1], 2.0)


if __name__ == "__main__":
    unittest.main()
