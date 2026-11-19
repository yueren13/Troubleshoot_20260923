"""Regression checks for fixed-disk ownership and independent model readiness in 09H."""
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from core09g import _readiness, _tile_plan


class FixedCoreTests(unittest.TestCase):
    def test_overlapping_fit_halos_preserve_disjoint_disk_ownership(self):
        row, col = np.indices((25, 25)).reshape(2, -1)
        spots = pd.DataFrame(dict(array_row=row, array_col=col, x_um=2. * row, y_um=2. * col))
        center = np.array([24., 24.])
        # With stride four and first disk row/column six, a cell centered in
        # row nine can span the ownership boundary between rows nine and ten.
        cell = pd.DataFrame({"09B_x_um": [18.], "09B_y_um": [24.]})
        disk, bin_owner, cell_owner, _, recs, fits = _tile_plan(
            spots, center, 12., cell, side=8, halo=2)
        expected = np.flatnonzero((2. * row - 24.)**2 + (2. * col - 24.)**2 <= 12.**2)
        scored = np.concatenate([win[disk[win] & (bin_owner[win] == i)] for i, win in fits.items()])
        np.testing.assert_array_equal(np.sort(scored), expected)
        self.assertEqual(sum(rec["n_owned_target_cells"] for rec in recs), 1)

        footprint = np.flatnonzero((row >= 8) & (row <= 10) & (col == 12))
        self.assertEqual(len(np.unique(bin_owner[footprint])), 2)
        centroid = np.flatnonzero((row == 9) & (col == 12)).item()
        self.assertEqual(cell_owner[0], bin_owner[centroid])
        self.assertTrue(np.isin(footprint, fits[cell_owner[0]]).all())

    def test_background_percentage_is_diagnostic_and_methods_are_independent(self):
        genes = pd.DataFrame({"gene": ["geneA", "geneB"], "target": [1, 0]})
        counts = sparse.csc_matrix(np.ones((2, 100), dtype=int))
        owner = np.tile([-1, -1, 0, 1], 25)
        background = np.zeros(100, dtype=bool)
        sparkle, spotclean = _readiness(counts, genes, background, owner)
        self.assertTrue(sparkle["ready"])
        self.assertFalse(spotclean["ready"])
        self.assertIn("fewer_than_two_background_positions", spotclean["reasons"])

        background[:2] = True
        _, spotclean = _readiness(counts, genes, background, owner)
        self.assertTrue(spotclean["ready"])
        self.assertTrue(spotclean["below_25pct_background"])
        self.assertEqual(spotclean["spatial_background_umis"], 4)


if __name__ == "__main__":
    unittest.main()
