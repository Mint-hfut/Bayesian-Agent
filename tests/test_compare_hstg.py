"""Tests for the HSTG ablation comparison script."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("compare_hstg", REPO_ROOT / "experiments" / "compare_hstg.py")
compare_hstg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_hstg)


def _write_run_root(root: Path, benchmark: str, flags, patch_counts, w_locals=None):
    runs = []
    for idx, success in enumerate(flags, 1):
        runs.append(
            {
                "task_id": f"task_{idx:02d}",
                "success": bool(success),
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
            }
        )
    (root).mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(
        json.dumps({"results": {benchmark: runs}}), encoding="utf-8"
    )
    for idx, count in enumerate(patch_counts, 1):
        task_dir = root / "skill_evolution" / benchmark / f"task_{idx:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"- failure_mode=mode_{n} observed=2" for n in range(count)]
        (task_dir / "skill_context_before.md").write_text("\n".join(lines), encoding="utf-8")
        belief = {}
        if w_locals is not None:
            belief["hstg_audit"] = {"w_local": w_locals[idx - 1]}
        (task_dir / "belief_before.json").write_text(json.dumps(belief), encoding="utf-8")


class CompareHstgTest(unittest.TestCase):
    def test_report_covers_final_and_cold_start_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_run_root(
                tmp_path / "categorical",
                "realfin_benchmark",
                flags=[0, 0, 1, 1, 1],
                patch_counts=[0, 0, 1, 1, 1],
            )
            _write_run_root(
                tmp_path / "hstg",
                "realfin_benchmark",
                flags=[0, 1, 1, 1, 1],
                patch_counts=[0, 1, 1, 1, 1],
                w_locals=[0.0, 0.2, 0.3, 0.4, 0.5],
            )
            json_out = tmp_path / "report.json"
            md_out = tmp_path / "report.md"
            code = compare_hstg.main(
                [
                    "--run",
                    f"categorical={tmp_path / 'categorical'}",
                    "--run",
                    f"hstg={tmp_path / 'hstg'}",
                    "--first-k",
                    "3",
                    "--out",
                    str(md_out),
                    "--json-out",
                    str(json_out),
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads(json_out.read_text(encoding="utf-8"))
            arms = {arm["label"]: arm for arm in report["benchmarks"]["realfin_benchmark"]}

            self.assertEqual(arms["categorical"]["summary"]["successes"], 3)
            self.assertEqual(arms["hstg"]["summary"]["successes"], 4)
            self.assertEqual(arms["categorical"]["first_k_successes"], 1)
            self.assertEqual(arms["hstg"]["first_k_successes"], 2)
            self.assertEqual(arms["categorical"]["first_patch_activation_position"], 3)
            self.assertEqual(arms["hstg"]["first_patch_activation_position"], 2)
            self.assertEqual(arms["hstg"]["w_local_curve"], [0.0, 0.2, 0.3, 0.4, 0.5])
            self.assertEqual(arms["categorical"]["w_local_curve"], [None] * 5)
            self.assertEqual(arms["hstg"]["cumulative_accuracy"][-1], 0.8)

            markdown = md_out.read_text(encoding="utf-8")
            self.assertIn("Final Metrics", markdown)
            self.assertIn("Cold Start And Patch Behavior", markdown)
            self.assertIn("w_local Trajectory", markdown)

    def test_missing_results_json_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                compare_hstg.main(["--run", f"x={tmp}/nope"])


if __name__ == "__main__":
    unittest.main()
