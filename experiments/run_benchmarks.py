#!/usr/bin/env python3
"""Run Bayesian-Agent benchmarks with GenericAgent as the execution harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesian_agent.adapters.generic_agent import GenericAgentAdapter
from bayesian_agent.benchmarks.realfin import run_realfin
from bayesian_agent.benchmarks.sop_lifelong import DEFAULT_DATA_ROOT, run_sop_lifelong


BENCH_CHOICES = ("core", "sop", "lifelong", "realfin")


@dataclass
class ExperimentRun:
    name: str
    mode: str
    out: Path
    baseline_paths: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.baseline_paths = list(self.baseline_paths or [])


@dataclass
class BenchmarkRun:
    bench: str
    out_root: Path


def build_benchmark_runs(bench: str, model: str, out_root: str = "") -> List[BenchmarkRun]:
    selected = expand_benchmarks(bench)
    if out_root:
        root = Path(out_root)
        if len(selected) == 1:
            return [BenchmarkRun(selected[0], root)]
        return [BenchmarkRun(item, root / item) for item in selected]
    return [BenchmarkRun(item, default_out_root(item, model)) for item in selected]


def expand_benchmarks(bench: str) -> List[str]:
    if bench == "core":
        return ["sop", "lifelong"]
    if bench in BENCH_CHOICES:
        return [bench]
    raise ValueError(f"Unsupported benchmark: {bench}")


def build_run_plan(mode: str, out_root: Path, baseline_paths: Sequence[str]) -> List[ExperimentRun]:
    mode = mode.replace("_", "-")
    out_root = Path(out_root)
    supplied_baseline = [str(path) for path in baseline_paths]
    fresh_baseline = str(out_root / "baseline" / "results.json")
    plan: List[ExperimentRun] = []
    if mode in {"all", "baseline"}:
        plan.append(ExperimentRun("baseline", "baseline", out_root / "baseline"))
    if mode in {"all", "bayesian-full"}:
        plan.append(ExperimentRun("bayesian_full", "bayesian-full", out_root / "bayesian_full"))
    if mode in {"all", "bayesian-incremental"}:
        plan.append(
            ExperimentRun(
                "bayesian_incremental",
                "bayesian-incremental",
                out_root / "bayesian_incremental",
                supplied_baseline or [fresh_baseline],
            )
        )
    if not plan:
        raise ValueError(f"Unsupported mode: {mode}")
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Bayesian-Agent benchmarks through GenericAgent.")
    parser.add_argument("--mode", choices=["all", "baseline", "bayesian-full", "bayesian-incremental"], default="all")
    parser.add_argument("--bench", choices=BENCH_CHOICES, default="core")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--genericagent-root", default="", help="Local GenericAgent checkout. Defaults to discovery.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--out-root",
        default="",
        help="Output root. Single benchmarks default to results/<bench>_<model>; multi-benchmark selections treat this as a parent root.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit tasks for smoke tests. 0 means full benchmark.")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--anthropic-base-url", default="https://api.deepseek.com/anthropic")
    parser.add_argument("--host-header", default="", help="Optional Host header for fixed-IP provider routing.")
    parser.add_argument("--protocol", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--disable-ssl-verify", action="store_true", help="Disable Python requests SSL verification for this run.")
    parser.add_argument("--baseline-results", action="append", default=[], help="Baseline results.json for incremental mode.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without calling the model.")
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    args.api_key_env = args.api_key_env or "DEEPSEEK_API_KEY"
    benchmark_runs = build_benchmark_runs(args.bench, args.model, args.out_root)
    adapter = GenericAgentAdapter(
        root=args.genericagent_root or None,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        anthropic_base_url=args.anthropic_base_url,
        protocol=args.protocol,
        verify_ssl=not args.disable_ssl_verify,
        host_header=args.host_header,
    )

    if args.dry_run:
        print_dry_run(adapter, args, benchmark_runs)
        return 0

    for benchmark in benchmark_runs:
        out_root = benchmark.out_root.resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        plan = build_run_plan(args.mode, out_root, args.baseline_results)
        completed = []
        for spec in plan:
            if spec.mode == "bayesian-incremental":
                missing = [path for path in spec.baseline_paths if not Path(path).exists()]
                if missing:
                    raise FileNotFoundError(
                        "Incremental mode needs baseline results. Missing: "
                        + ", ".join(missing)
                        + ". Run --mode all or pass --baseline-results."
                    )
            print(f"[experiment] starting {benchmark.bench}:{spec.name} -> {spec.out}", flush=True)
            started = time.time()
            result = run_selected_benchmark(adapter, args, spec, benchmark.bench)
            completed.append(
                {
                    "name": spec.name,
                    "mode": spec.mode,
                    "bench": benchmark.bench,
                    "out": str(spec.out),
                    "elapsed_seconds": round(time.time() - started, 2),
                    "summaries": result.get("summaries", {}),
                    "combined_summaries": result.get("combined_summaries", {}),
                }
            )
            print(f"[experiment] finished {benchmark.bench}:{spec.name}", flush=True)
        write_experiment_summary(out_root, args.model, benchmark.bench, completed)
    return 0


def run_selected_benchmark(
    adapter: GenericAgentAdapter,
    args: argparse.Namespace,
    spec: ExperimentRun,
    bench: str,
) -> Mapping[str, object]:
    common = {
        "adapter": adapter,
        "out_root": spec.out,
        "model": args.model,
        "data_root": Path(args.data_root),
        "mode": spec.mode,
        "limit": args.limit,
        "max_turns": args.max_turns,
        "baseline_paths": spec.baseline_paths,
    }
    if bench == "realfin":
        return run_realfin(**common)
    return run_sop_lifelong(**common, bench=bench)


def print_dry_run(
    adapter: GenericAgentAdapter,
    args: argparse.Namespace,
    benchmark_runs: Sequence[BenchmarkRun],
) -> None:
    header = {
        "genericagent_root": str(adapter.resolve_root()),
        "data_root": str(Path(args.data_root).resolve()),
        "model": args.model,
        "requested_bench": args.bench,
        "selected_benchmarks": [item.bench for item in benchmark_runs],
    }
    print(json.dumps(header, indent=2))
    for benchmark in benchmark_runs:
        out_root = benchmark.out_root.resolve()
        plan = build_run_plan(args.mode, out_root, args.baseline_results)
        print(f"\n## bench={benchmark.bench} out_root={out_root}")
        for spec in plan:
            print(f"[{spec.name}] mode={spec.mode} out={spec.out}")
            if spec.baseline_paths:
                print("baseline_results=" + ",".join(spec.baseline_paths))


def load_env_file(path: Path = None) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv."""

    path = Path(path or ".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def write_experiment_summary(
    out_root: Path,
    model: str,
    bench: str,
    completed: Sequence[Mapping[str, object]],
) -> None:
    title = benchmark_title(bench)
    lines = [
        f"# {title}: {model}",
        "",
        "This experiment uses GenericAgent as the execution harness and Bayesian-Agent for benchmark orchestration and Skill evolution.",
        "",
    ]
    for item in completed:
        lines.extend([f"## {item['name']}", "", f"- Output: `{item['out']}`", f"- Elapsed: `{item['elapsed_seconds']}s`", ""])
        table_path = Path(str(item["out"])) / "table.md"
        if table_path.exists():
            lines.append(table_path.read_text(encoding="utf-8").strip())
            lines.append("")
    manifest = {"model": model, "bench": bench, "runs": list(completed)}
    (out_root / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def benchmark_title(bench: str) -> str:
    return {
        "core": "SOP-Bench + Lifelong AgentBench",
        "sop": "SOP-Bench",
        "lifelong": "Lifelong AgentBench",
        "realfin": "RealFin-Bench",
    }.get(bench, bench)


def default_out_root(bench: str, model: str) -> Path:
    model_slug = model.replace("-", "_")
    return Path("results") / f"{bench}_{model_slug}"


if __name__ == "__main__":
    raise SystemExit(main())
