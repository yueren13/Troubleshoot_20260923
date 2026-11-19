"""Synthetic regression checks for the native background rule. Run with python -m unittest test_09f_score_window."""
import unittest

import numpy as np
import pandas as pd

from roi09f import _window


class ScoreWindowTests(unittest.TestCase):
    def setUp(self):
        row, col = np.indices((8, 8)).reshape(2, -1)
        self.positions = pd.DataFrame(dict(array_row=row, array_col=col, x_um=row * 2., y_um=col * 2.,
                                           sr_in_tissue=(col >= 3), cell_index=np.where(col >= 5, 0, -1),
                                           he_tissue_score=np.where(col < 3, .04, .85)))
        self.positions.loc[0, "he_tissue_score"] = np.nan
        self.positions.loc[1, "cell_index"] = 4

    def select(self, strict):
        return _window(self.positions, 8, .1, .25, np.array([7., 7.]), strict)

    def test_unknown_and_segmented_positions_are_never_background(self):
        table, bg, origin, _ = self.select(True)
        self.assertEqual(origin, [0, 0])
        self.assertEqual(bg.sum(), 22)
        self.assertTrue((table.loc[bg, "cell_index"] < 0).all())
        self.assertTrue(table.loc[bg, "he_tissue_score"].notna().all())
        self.assertFalse(table.loc[bg, "sr_in_tissue"].any())

    def test_low_score_inside_sr_is_explicit_fallback(self):
        self.positions["sr_in_tissue"] = True
        with self.assertRaisesRegex(ValueError, "provisional background"):
            self.select(True)
        _, bg, _, mode = self.select(False)
        self.assertEqual(bg.sum(), 22)
        self.assertIn("including_sr_tissue", mode)


if __name__ == "__main__":
    unittest.main()
