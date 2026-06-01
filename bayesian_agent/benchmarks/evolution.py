"""Bayesian-Agent owned benchmark Skill evolution helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from bayesian_agent.core.context import SkillContextBuilder
from bayesian_agent.core.evidence import TrajectoryEvidence
from bayesian_agent.core.registry import BayesianSkillRegistry


def classify_failure(benchmark: str, run: Mapping[str, Any]) -> str:
    """Classify common benchmark failures into reusable evidence labels."""

    if run.get("success"):
        return ""
    if benchmark == "sop_bench":
        got = str(run.get("got") or "")
        expected = str(run.get("expected") or "")
        if "<final_decision>" in got:
            return "wrote_xml_tags_in_csv_expected_output"
        if not got or got.lower() == "none":
            return "left_expected_output_blank"
        if expected and got != expected:
            return "computed_decision_for_wrong_or_unverified_target_row"
    if benchmark == "lifelong_agentbench":
        got = str(run.get("got_sql") or "")
        expected = str(run.get("expected_sql") or "")
        if "payment_id" in got and "payment_id" not in expected:
            return "invented_unrequested_column"
        if "Turn 1" in got or "🛠️" in got:
            return "wrote_transcript_instead_of_sql_after_workspace_confusion"
        if run.get("error"):
            return str(run.get("error"))[:160]
    return str(run.get("error") or "benchmark_failure")[:160]


def evidence_from_run(benchmark: str, run: Mapping[str, Any]) -> TrajectoryEvidence:
    failure_mode = str(run.get("failure_mode") or classify_failure(benchmark, run))
    return TrajectoryEvidence.from_run(
        run,
        skill_id=f"benchmark/{benchmark}",
        context=benchmark,
        failure_mode=failure_mode,
    )


def record_benchmark_run(registry: BayesianSkillRegistry, benchmark: str, run: Mapping[str, Any]) -> None:
    enriched = dict(run)
    enriched["failure_mode"] = str(run.get("failure_mode") or classify_failure(benchmark, run))
    registry.record(evidence_from_run(benchmark, enriched))


def seed_registry_from_results(registry: BayesianSkillRegistry, benchmark_runs: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    for benchmark, runs in benchmark_runs.items():
        for run in runs:
            record_benchmark_run(registry, benchmark, run)


def build_benchmark_skill_context(benchmark: str, registry: BayesianSkillRegistry) -> str:
    """Render posterior Skill context plus benchmark-specific guardrails."""

    posterior = SkillContextBuilder(registry).render(task_context=benchmark, limit=5)
    rules = _stable_rules(benchmark)
    patches = _failure_mode_patch_rules(benchmark, registry)
    if not posterior and not rules and not patches:
        return ""
    lines = []
    if posterior:
        lines.append(posterior)
    if patches:
        lines.extend(["", f"### Bayesian Failure-Mode Patches: {benchmark}"])
        for failure_mode, count, patch_rules in patches:
            lines.append(f"- failure_mode={failure_mode} observed={count}")
            lines.extend(f"  - {rule}" for rule in patch_rules)
    if rules:
        lines.extend(["", f"### Benchmark SOP Guardrails: {benchmark}"])
        lines.extend(f"- {rule}" for rule in rules)
    return "\n".join(line for line in lines if line is not None).strip()


def _stable_rules(benchmark: str):
    if benchmark == "sop_bench":
        return [
            "Read `sop.txt`, `tools.py`, and the target CSV row before acting.",
            "The requested row is one-indexed after the header; update `rows[row_index - 1]` when using `csv.DictReader`.",
            "Before calling tools, verify the target row's `order_id`, `product_id`, `quantity_requested`, `customer_id`, and `order_total`; never reuse inputs from another row.",
            "Compute only the target row and write only its `expected_output` cell.",
            "Use Python's `csv` module for writing; preserve all other rows and columns exactly.",
            "Write the raw category string only, for example `manual_review`; never write XML tags, Markdown, quotes, or explanations into the cell.",
            "Verify the target row's `expected_output` is non-empty before finishing.",
        ]
    if benchmark == "lifelong_agentbench":
        return [
            "Read `task.json` in the current workspace; do not inspect sibling benchmark runs.",
            "Write exactly one SQL statement to `answer.sql`; no Markdown and no explanation.",
            "Use only columns present in `task.json` unless the instruction explicitly asks for a new value in an existing column.",
            "For INSERT statements, do not include id or primary-key columns unless the instruction explicitly provides their values.",
            "For mutation tasks, write executable SQL that reproduces the expected table state.",
            "If SQL ranking is needed, express ranking inside a subquery and keep the final output to one SQL statement.",
        ]
    return []


def _failure_mode_patch_rules(benchmark: str, registry: BayesianSkillRegistry):
    counts = {}
    for belief in registry.beliefs():
        if belief.skill_id != f"benchmark/{benchmark}" and benchmark not in belief.contexts:
            continue
        for failure_mode, count in belief.failure_modes.items():
            if count > 0:
                counts[failure_mode] = counts.get(failure_mode, 0) + int(count)

    patches = []
    mode_rules = _patch_rule_catalog(benchmark)
    for failure_mode, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rules = mode_rules.get(failure_mode)
        if rules:
            patches.append((failure_mode, count, rules))
    return patches


def _patch_rule_catalog(benchmark: str):
    if benchmark == "sop_bench":
        return {
            "left_expected_output_blank": [
                "After writing, re-read `test_set_with_outputs.csv` and confirm the target row's `expected_output` is non-empty.",
                "If the target cell is empty, write the computed raw category string before finishing.",
            ],
            "wrote_xml_tags_in_csv_expected_output": [
                "Write only the raw category string into `expected_output`; strip XML tags, Markdown, quotes, and explanations.",
            ],
            "computed_decision_for_wrong_or_unverified_target_row": [
                "Before tool calls and before writing, verify the requested one-indexed row and its order fields match the target task.",
            ],
        }
    if benchmark == "lifelong_agentbench":
        return {
            "invented_unrequested_column": [
                "Use only columns present in `task.json`; do not invent id, payment_id, or other primary-key values.",
                "For INSERT, omit id or primary-key columns unless the instruction explicitly provides the value.",
            ],
            "wrote_transcript_instead_of_sql_after_workspace_confusion": [
                "Write exactly one executable SQL statement to `answer.sql`; do not write transcript text, tool logs, Markdown, or explanations.",
                "Read only the current task workspace and avoid copying content from sibling benchmark runs.",
            ],
        }
    return {}
