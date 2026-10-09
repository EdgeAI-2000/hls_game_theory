import gzip
import json
import tempfile
import unittest
from pathlib import Path

from analyze_e6_formal import ScoreAccumulator, iter_jsonl_paths, trace_paths


class E4E6AnalysisTest(unittest.TestCase):
    def test_score_accumulator_reports_proper_metrics_and_curve(self):
        acc = ScoreAccumulator(bins=2)
        for label, score in [(0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)]:
            acc.add(label, score)
        summary = acc.summary()
        self.assertEqual(summary["samples"], 4)
        self.assertAlmostEqual(summary["auc"], 1.0)
        self.assertAlmostEqual(summary["brier"], 0.025)
        self.assertEqual(sum(row["count"] for row in summary["reliability_bins"]), 4)

    def test_jsonl_reader_accepts_gzip_formal_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed_1"
            seed.mkdir()
            path = seed / "gate1_steps.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": 1}) + "\n")
                handle.write(json.dumps({"step": 2}) + "\n")
            rows = list(iter_jsonl_paths(trace_paths(root, "gate1_steps.jsonl")))
            self.assertEqual([row["step"] for row in rows], [1, 2])


if __name__ == "__main__":
    unittest.main()
