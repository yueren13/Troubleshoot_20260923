"""Regression tests for disjoint windows and raw-count reconciliation in 09G."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd

from compare09g import _load_model
from regions09g import _pick


class MatchedRegionTests(unittest.TestCase):
    def test_find_two_disjoint_windows_when_the_first_candidate_blocks_both(self):
        center = {"row": 10, "col": 10}
        top = {"row": 0, "col": 0}
        bottom = {"row": 20, "col": 0}
        self.assertEqual(_pick([center, top, bottom], side=18, n=2), [top, bottom])

    def test_cell_and_gene_order_can_change_but_raw_counts_must_match(self):
        genes = pd.Index(["Cd4", "Lyz2"])
        cells = pd.Index(["cell1", "cell2"])
        expected = np.array([[0, 2], [3, 7]])
        hashes = {"counts.mtx": "fixture-matrix", "positions.tsv": "fixture-positions"}
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "fit.json").write_text(json.dumps(
                {"state": "complete", "method": "SpotClean", "input_sha256": "fixture", "input_files_sha256": hashes}))
            pd.DataFrame({"gene": list(genes), "status": ["spatial_em"]*2}).to_csv(
                folder / "gene_fits.csv", index=False)

            def write(raw):
                np.savez(folder / "cell_counts.npz", genes=genes[::-1].to_numpy(dtype=str),
                         cell_ids=cells[::-1].to_numpy(dtype=str),
                         raw=raw[::-1, ::-1], corrected=expected[::-1, ::-1] * .8)

            write(expected)
            estimates, _, _ = _load_model(folder, "spotclean", "fixture", hashes, genes, cells, expected)
            np.testing.assert_allclose(estimates.loc[cells, genes].T, expected * .8)
            mismatched = expected.copy()
            mismatched[0, 0] += 1
            write(mismatched)
            with self.assertRaisesRegex(ValueError, "Raw gene-by-cell counts"):
                _load_model(folder, "spotclean", "fixture", hashes, genes, cells, expected)


if __name__ == "__main__":
    unittest.main()
