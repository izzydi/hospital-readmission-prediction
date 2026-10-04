import unittest

import numpy as np
import pandas as pd

from src.hospital_readmission_model import find_target, score


class HospitalPipelineSmokeTests(unittest.TestCase):
    def test_find_target_is_case_insensitive(self):
        self.assertEqual(find_target(["Age", "Readmitted"]), "Readmitted")

    def test_score_returns_expected_metrics(self):
        y_true = pd.Series(["yes", "no", "yes", "no"])
        y_pred = np.array(["yes", "no", "yes", "no"])
        metrics = score(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["f1_weighted"], 1.0)


if __name__ == "__main__":
    unittest.main()
