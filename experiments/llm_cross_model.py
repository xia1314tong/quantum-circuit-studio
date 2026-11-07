"""把不同模型的 LLM 实验结果并成一张跨模型对照表。

只读已保存的 CSV，不调用 API::

    python -m experiments.llm_cross_model \
        --model deepseek-flash=results/llm_summary.csv:results/llm_arms.csv \
        --model deepseek-v4-pro=results/llm_summary_v4pro.csv:results/llm_arms_v4pro.csv

`--model` 的格式是 ``名字=summary.csv:arms.csv``（相对 experiments/ 目录）。
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _resolve(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(RESULTS_DIR, path)


def load_rows(path: str) -> List[dict]:
    with open(_resolve(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(summary_rows: Sequence[dict]) -> Dict[str, dict]:
    """按臂汇总：pass@1 与最终成功率按 rep 加权，轮数与 token 取平均/求和。"""
    buckets: Dict[str, List[dict]] = {}
    for row in summary_rows:
        buckets.setdefault(row["arm"], []).append(row)
    table: Dict[str, dict] = {}
    for arm, rows in buckets.items():
        reps = sum(int(row["reps"]) for row in rows)
        if reps == 0:
            continue
        # 每个任务行的 pass_at_1 / final_pass 本身就是比例，按 reps 加权即总体比例
        pass1 = sum(float(row["pass_at_1"]) * int(row["reps"]) for row in rows) / reps
        final = sum(float(row["final_pass"]) * int(row["reps"]) for row in rows) / reps
        rounds = sum(float(row["mean_rounds"]) * int(row["reps"]) for row in rows) / reps
        tokens = sum(int(row["total_tokens"]) for row in rows)
        table[arm] = {
            "tasks": len(rows),
            "pairs": reps,
            "pass_at_1": round(pass1, 3),
            "final_pass": round(final, 3),
            "mean_rounds": round(rounds, 2),
            "total_tokens": tokens,
        }
    return table


def pair_stats(arms_rows: Sequence[dict], arm_a: str, arm_b: str) -> Optional[dict]:
    for row in arms_rows:
        if row.get("arm_a") == arm_a and row.get("arm_b") == arm_b:
            return row
        if row.get("arm_a") == arm_b and row.get("arm_b") == arm_a:
            flipped = dict(row)
            flipped["arm_a"] = row.get("arm_b", arm_b)
            flipped["arm_b"] = row.get("arm_a", arm_a)
            flipped["rate_a"] = row.get("rate_b", "")
            flipped["rate_b"] = row.get("rate_a", "")
            delta = row.get("delta", "")
            if isinstance(delta, str) and delta:
                flipped["delta"] = delta[1:] if delta.startswith("-") else "-" + delta
            return flipped
    return None


def build_table(entries: Sequence[tuple], arm_a: str = "resample", arm_b: str = "feedback") -> List[dict]:
    """entries: [(model_name, summary_rows, arms_rows), ...]"""
    out: List[dict] = []
    for name, summary_rows, arms_rows in entries:
        stats = aggregate(summary_rows)
        pair = pair_stats(arms_rows, arm_a, arm_b) or {}
        for arm in (arm_a, arm_b, "oneshot"):
            if arm not in stats:
                continue
            row = {"model": name, "arm": arm, **stats[arm]}
            row.update(
                {
                    "delta_vs_resample": pair.get("delta", "") if arm == arm_b else "",
                    "ci_low": pair.get("ci_low", "") if arm == arm_b else "",
                    "ci_high": pair.get("ci_high", "") if arm == arm_b else "",
                    "mcnemar_p": pair.get("mcnemar_p", "") if arm == arm_b else "",
                    "b_only": pair.get("b_only", "") if arm == arm_b else "",
                    "a_only": pair.get("a_only", "") if arm == arm_b else "",
                    # 配对口径：只统计两臂都跑完的 (task, rep)，与上面按任务加权的口径不同
                    "matched_pairs": pair.get("pairs", "") if arm == arm_b else "",
                    "rate_resample": pair.get("rate_a", "") if arm == arm_b else "",
                    "rate_feedback": pair.get("rate_b", "") if arm == arm_b else "",
                }
            )
            out.append(row)
    return out


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm–Bonferroni 校正：同一族假设（这里是三个模型的同一比较）一起校正。"""
    m = len(pvalues)
    order = sorted(range(m), key=lambda index: pvalues[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def write_csv(path: str, rows: Sequence[dict]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def parse_entry(text: str) -> tuple:
    name, _, paths = text.partition("=")
    summary, _, arms = paths.partition(":")
    if not name or not summary or not arms:
        raise SystemExit(f"--model 需要 名字=summary.csv:arms.csv，收到：{text}")
    return name.strip(), summary.strip(), arms.strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="跨模型对照表")
    parser.add_argument("--model", action="append", default=[], help="名字=summary.csv:arms.csv")
    parser.add_argument("--out", default="llm_cross_model.csv")
    parser.add_argument("--arm-a", default="resample")
    parser.add_argument("--arm-b", default="feedback")
    options = parser.parse_args(argv)

    entries = []
    for text in options.model:
        name, summary_path, arms_path = parse_entry(text)
        entries.append((name, load_rows(summary_path), load_rows(arms_path)))
    if not entries:
        raise SystemExit("至少给一个 --model")

    table = build_table(entries, options.arm_a, options.arm_b)
    # 三个模型做的是同一个比较，p 值应作族错误率校正后再解读
    family = [row for row in table if row["arm"] == options.arm_b and row.get("mcnemar_p")]
    pvalues = []
    for row in family:
        try:
            pvalues.append(float(row["mcnemar_p"]))
        except (TypeError, ValueError):
            pvalues.append(1.0)
    for row, adjusted in zip(family, holm_adjust(pvalues)):
        row["p_holm"] = f"{adjusted:.6f}"
    path = write_csv(os.path.join(RESULTS_DIR, options.out), table)
    print(f"[ok] {os.path.relpath(path)}")
    for row in table:
        print(
            f"{row['model']:18s} {row['arm']:9s} pass@1={row['pass_at_1']:.3f} "
            f"final={row['final_pass']:.3f} rounds={row['mean_rounds']:.2f} tokens={row['total_tokens']}"
        )
    for row in family:
        print(
            f"[holm] {row['model']:18s} p={row.get('mcnemar_p', '')} "
            f"→ p_holm={row.get('p_holm', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())