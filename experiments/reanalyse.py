"""从已保存的逐轮记录重算汇总与配对统计，不重新调用 API。

用途：改了统计口径（比如修了配对方向、换了显著性方法）之后，不必重跑实验。

用法::

    python -m experiments.reanalyse
    python -m experiments.reanalyse --runs experiments/results/llm_runs_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Sequence

from . import llm_loop


def load_rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["passed"] = int(row["passed"])
        row["rep"] = int(row["rep"])
        row["round"] = int(row["round"])
        row["prompt_tokens"] = int(row.get("prompt_tokens") or 0)
        row["completion_tokens"] = int(row.get("completion_tokens") or 0)
        row["reasoning_tokens"] = int(row.get("reasoning_tokens") or 0)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重算 LLM 实验的汇总与配对统计")
    parser.add_argument(
        "--runs",
        default=os.path.join(llm_loop.RESULTS_DIR, "llm_runs.csv"),
        help="逐轮记录 CSV",
    )
    parser.add_argument(
        "--drop-api-error",
        action="store_true",
        help="丢弃服务端错误（额度/权限等）造成的尝试，避免把服务故障算成模型失败",
    )
    options = parser.parse_args(argv)

    rows = load_rows(options.runs)
    if options.drop_api_error:
        before = len(rows)
        rows = [row for row in rows if row.get("layer") != "api_error"]
        print(f"[filter] 丢弃 api_error 行 {before - len(rows)} 条，剩余 {len(rows)} 条")
    summary = llm_loop.summarise(rows)
    stats = llm_loop.compare_arms(rows)

    base = os.path.dirname(options.runs)
    suffix = os.path.basename(options.runs)
    if suffix.startswith("llm_runs"):
        suffix = suffix[len("llm_runs") :]
    if suffix.endswith(".csv"):
        suffix = suffix[: -len(".csv")]
    summary_path = llm_loop._write_csv(
        os.path.join(base, f"llm_summary{suffix}.csv"),
        summary,
        ["arm", "task", "reps", "pass_at_1", "final_pass", "mean_rounds", "total_tokens"],
    )
    stats_path = llm_loop._write_csv(
        os.path.join(base, f"llm_arms{suffix}.csv"),
        stats,
        [
            "arm_a",
            "arm_b",
            "pairs",
            "rate_a",
            "rate_b",
            "delta",
            "ci_low",
            "ci_high",
            "b_only",
            "a_only",
            "mcnemar_p",
        ],
    )

    print(f"[ok] {os.path.relpath(summary_path)}")
    print(f"[ok] {os.path.relpath(stats_path)}")
    for arm in ("oneshot", "resample", "feedback"):
        arm_rows = [row for row in summary if row["arm"] == arm]
        if not arm_rows:
            continue
        pass1 = sum(row["pass_at_1"] for row in arm_rows) / len(arm_rows)
        final = sum(row["final_pass"] for row in arm_rows) / len(arm_rows)
        rounds = sum(row["mean_rounds"] for row in arm_rows) / len(arm_rows)
        tokens = sum(row["total_tokens"] for row in arm_rows)
        print(
            f"{arm:9s} pass@1 = {pass1:.3f}   final = {final:.3f}   "
            f"平均轮数 = {rounds:.2f}   总 tokens = {tokens}"
        )
    for row in stats:
        print(
            f"[{row['arm_a']} vs {row['arm_b']}] {row['rate_a']:.3f} → {row['rate_b']:.3f} "
            f"(Δ={row['delta']:+.3f}, 95% CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}], "
            f"McNemar p={row['mcnemar_p']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
