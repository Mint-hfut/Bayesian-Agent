#!/usr/bin/env python3
"""Compare Skill-evolution ablation runs (e.g. categorical_bayes vs hstg).

Reads two or more benchmark run roots produced by run_benchmarks.py and
reports the metrics where HSTG's contribution should show up:

- final accuracy / token cost / efficiency
- cold-start window accuracy (first K tasks in execution order)
- patch activation timing (first task whose prompt carried failure-mode
  patches) and average active patch modes per prompt
- cumulative success curve per run for plotting
- w_local trajectory from hstg_audit snapshots when present

Usage:

    python experiments/compare_hstg.py \
      --run categorical=results/hstg_ablation/categorical/bayesian_full \
      --run hstg=results/hstg_ablation/hstg/bayesian_full \
      --first-k 10 \
      --out temp/hstg_compare.md \
      --json-out temp/hstg_compare.json

Each --run value is LABEL=PATH where PATH contains results.json and,
for bayesian runs, a skill_evolution/ directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesian_agent.core.repair import normalize_results, summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Skill-evolution ablation runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Labelled run root containing results.json. Repeat for each arm.",
    )
    parser.add_argument("--benchmark", default="", help="Restrict to one benchmark key, e.g. realfin_benchmark.")
    parser.add_argument("--first-k", type=int, default=10, help="Cold-start window size in execution order.")
    parser.add_argument("--out", default="", help="Optional markdown report path.")
    parser.add_argument("--json-out", default="", help="Optional JSON report path with per-position curves.")
    return parser


def parse_run_specs(specs: List[str]) -> List[Dict[str, str]]:
    runs = []
    for spec in specs:
        label, _, path = spec.partition("=")
        if not label or not path:
            raise SystemExit(f"--run expects LABEL=PATH, got: {spec!r}")
        runs.append({"label": label, "path": path})
    return runs


def load_run(label: str, root: str) -> Dict[str, Any]:
    root_path = Path(root)
    results_path = root_path / "results.json"
    if not results_path.exists():
        raise SystemExit(f"[{label}] results.json not found under {root_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "root": root_path,
        "payload": payload,
        "results": normalize_results(payload),
    }


def analyze_benchmark(run: Mapping[str, Any], benchmark: str, first_k: int) -> Dict[str, Any]:
    ordered = list(run["results"].get(benchmark, []))
    summary = summarize({benchmark: ordered}).get(benchmark, {})
    successes = [bool(item.get("success")) for item in ordered]
    task_ids = [str(item.get("task_id") or f"pos_{idx}") for idx, item in enumerate(ordered, 1)]

    window = successes[:first_k]
    cumulative = []
    running = 0
    for idx, hit in enumerate(successes, 1):
        running += int(hit)
        cumulative.append(round(running / idx, 4))

    patches = analyze_patch_activation(run["root"], benchmark, task_ids)
    w_curve = collect_w_local_curve(run["root"], benchmark, task_ids)

    return {
        "benchmark": benchmark,
        "tasks": len(ordered),
        "summary": summary,
        "first_k": first_k,
        "first_k_successes": sum(window),
        "first_k_accuracy": round(sum(window) / len(window), 4) if window else 0.0,
        "cumulative_accuracy": cumulative,
        "task_ids": task_ids,
        "success_flags": [int(hit) for hit in successes],
        **patches,
        "w_local_curve": w_curve,
    }


def analyze_patch_activation(root: Path, benchmark: str, task_ids: List[str]) -> Dict[str, Any]:
    evolution_root = root / "skill_evolution" / benchmark
    active_counts: List[Optional[int]] = []
    first_active_position = 0
    for position, task_id in enumerate(task_ids, 1):
        context_path = evolution_root / task_id / "skill_context_before.md"
        if not context_path.exists():
            active_counts.append(None)
            continue
        text = context_path.read_text(encoding="utf-8")
        count = sum(1 for line in text.splitlines() if line.startswith("- failure_mode="))
        active_counts.append(count)
        if count > 0 and not first_active_position:
            first_active_position = position
    observed = [count for count in active_counts if count is not None]
    return {
        "patch_snapshots": len(observed),
        "first_patch_activation_position": first_active_position,
        "prompts_with_patches": sum(1 for count in observed if count > 0),
        "mean_active_patch_modes": round(sum(observed) / len(observed), 3) if observed else 0.0,
        "active_patch_counts": active_counts,
    }


def collect_w_local_curve(root: Path, benchmark: str, task_ids: List[str]) -> List[Optional[float]]:
    evolution_root = root / "skill_evolution" / benchmark
    curve: List[Optional[float]] = []
    for task_id in task_ids:
        belief_path = evolution_root / task_id / "belief_before.json"
        value: Optional[float] = None
        if belief_path.exists():
            try:
                audit = json.loads(belief_path.read_text(encoding="utf-8")).get("hstg_audit") or {}
                if "w_local" in audit:
                    value = float(audit["w_local"])
            except (json.JSONDecodeError, TypeError, ValueError):
                value = None
        curve.append(value)
    return curve


def selected_benchmarks(runs: List[Dict[str, Any]], benchmark: str) -> List[str]:
    if benchmark:
        return [benchmark]
    keys: List[str] = []
    for run in runs:
        for key in run["results"]:
            if key not in keys:
                keys.append(key)
    return keys


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# HSTG Ablation Comparison", ""]
    for benchmark, arms in report["benchmarks"].items():
        lines.append(f"## {benchmark}")
        lines.append("")
        lines.append("### Final Metrics")
        lines.append("")
        lines.append("| Run | Accuracy | Success | Total Tokens | Efficiency |")
        lines.append("|---|---:|---:|---:|---:|")
        for arm in arms:
            summary = arm["summary"]
            lines.append(
                f"| {arm['label']} | {summary.get('accuracy', 0.0):.1%} "
                f"| {summary.get('successes', 0)}/{summary.get('tasks', 0)} "
                f"| {summary.get('total_tokens', 0):,} "
                f"| {summary.get('efficiency', 0.0)} |"
            )
        lines.append("")
        lines.append("### Cold Start And Patch Behavior")
        lines.append("")
        lines.append(
            "| Run | First-K Accuracy | First Patch Activation (task #) | Prompts With Patches | Mean Patch Modes/Prompt |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for arm in arms:
            first_activation = arm["first_patch_activation_position"] or "-"
            lines.append(
                f"| {arm['label']} | {arm['first_k_successes']}/{min(arm['first_k'], arm['tasks'])} "
                f"({arm['first_k_accuracy']:.1%}) "
                f"| {first_activation} | {arm['prompts_with_patches']}/{arm['patch_snapshots']} "
                f"| {arm['mean_active_patch_modes']} |"
            )
        lines.append("")
        lines.append("### Cumulative Accuracy Curve")
        lines.append("")
        lines.append("| Task # | " + " | ".join(arm["label"] for arm in arms) + " |")
        lines.append("|---:|" + "---:|" * len(arms))
        depth = max((arm["tasks"] for arm in arms), default=0)
        for position in range(depth):
            row = [str(position + 1)]
            for arm in arms:
                curve = arm["cumulative_accuracy"]
                row.append(f"{curve[position]:.3f}" if position < len(curve) else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        if any(any(value is not None for value in arm["w_local_curve"]) for arm in arms):
            lines.append("### w_local Trajectory (HSTG runs)")
            lines.append("")
            lines.append("| Task # | " + " | ".join(arm["label"] for arm in arms) + " |")
            lines.append("|---:|" + "---:|" * len(arms))
            for position in range(depth):
                row = [str(position + 1)]
                for arm in arms:
                    curve = arm["w_local_curve"]
                    value = curve[position] if position < len(curve) else None
                    row.append(f"{value:.3f}" if value is not None else "-")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runs = [load_run(spec["label"], spec["path"]) for spec in parse_run_specs(args.run)]
    benchmarks = selected_benchmarks(runs, args.benchmark)
    if not benchmarks:
        raise SystemExit("No benchmark results found in the provided run roots.")

    report: Dict[str, Any] = {"first_k": args.first_k, "benchmarks": {}}
    for benchmark in benchmarks:
        arms = []
        for run in runs:
            if benchmark not in run["results"]:
                continue
            analysis = analyze_benchmark(run, benchmark, args.first_k)
            analysis["label"] = run["label"]
            analysis["root"] = str(run["root"])
            arms.append(analysis)
        if arms:
            report["benchmarks"][benchmark] = arms

    markdown = render_markdown(report)
    print(markdown)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"[compare] markdown written to {out_path}", file=sys.stderr)
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[compare] json written to {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
