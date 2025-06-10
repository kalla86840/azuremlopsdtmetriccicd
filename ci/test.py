
import unittest
import joblib
import pandas as pd

class TestModel(unittest.TestCase):
    def test_model_file_exists(self):
        try:
            model = joblib.load('model.joblib')
            self.assertIsNotNone(model)
        except FileNotFoundError:
            self.fail("model.joblib file not found.")

    def test_data_integrity(self):
        df = pd.read_csv('../cars.csv')
        self.assertIn('sale_price', df.columns)
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()
