"""Focused checks for the SpotClean canary and recovery error handling."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover import should_pause, tile_budget, tile_priority


class RecoveryPolicyTests(unittest.TestCase):
    def test_fresh_paired_informative_tiles_first_with_more_background(self):
        names = ("strong", "weak", "missing_sp", "retry", "no_mac", "modest")
        original = SimpleNamespace(loc={n: {"near_panT_cells": 9, "macrophage_cells": 4}
                                        for n in names})
        original.loc["no_mac"]["macrophage_cells"] = 0
        original.loc["modest"]["macrophage_cells"] = 2
        def item(name, bg):
            return dict(name=name, sparkle=True if name!="missing_sp" else None, tries=int(name=="retry"),
                        coverage={"n_background":bg, "n_complete_cells":100})
        tasks = [item(name, count) for name, count in zip(names, (500, 15, 2000, 1500, 1300, 1100))]
        ordered = sorted(tasks, key=lambda t:tile_priority(t,original))
        self.assertEqual([t["name"] for t in ordered],
                         ["strong", "weak", "modest", "no_mac", "missing_sp", "retry"])
        self.assertEqual(tile_budget("0"), 0)
        self.assertEqual(tile_budget("1"),1)
        self.assertIsNone(tile_budget("all"))

    def test_stop_on_cli_or_repeated_error(self):
        self.assertTrue(should_pause("Error: Usage: spotclean09f.R INPUT_DIR OUTPUT_DIR", None))
        self.assertTrue(should_pause("Error: there is no package called 'SpotClean'", None))
        self.assertFalse(should_pause("No finite estimate", None))
        self.assertTrue(should_pause("No finite estimate", "No finite estimate"))
        self.assertFalse(should_pause("No finite estimate", "Timeout"))


if __name__ == "__main__":
    unittest.main()
