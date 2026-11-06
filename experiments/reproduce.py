"""一键复现：测试 → 数据 → 图 → 第三方交叉验证。

用法::

    python -m experiments.reproduce            # 全跑
    python -m experiments.reproduce --skip-tests --skip-qiskit

投稿时的复现声明对应这条命令：从干净环境执行后，``experiments/results/``
与 ``experiments/figures/`` 内容应与仓库中已提交的一致（确定性计算 + 固定种子）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args: list[str], label: str) -> bool:
    print(f"\n=== {label} ===")
    print("$ " + " ".join(args))
    started = time.time()
    process = subprocess.run(args, cwd=ROOT)
    elapsed = time.time() - started
    ok = process.returncode == 0
    print(f"[{'ok' if ok else 'FAIL'}] {label} ({elapsed:.1f}s)")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="复现论文全部结果")
    parser.add_argument("--skip-tests", action="store_true", help="跳过 pytest")
    parser.add_argument("--skip-qiskit", action="store_true", help="跳过 Qiskit 交叉验证")
    options = parser.parse_args(argv)

    steps: list[tuple[str, list[str], bool]] = []
    if not options.skip_tests:
        steps.append(("测试", [sys.executable, "-m", "pytest", "-q"], True))
    steps.append(("生成数据", [sys.executable, "-m", "experiments.run_experiments"], True))
    steps.append(("生成图表", [sys.executable, "-m", "experiments.make_figures"], True))
    if not options.skip_qiskit:
        steps.append(
            ("Qiskit 交叉验证", [sys.executable, "-m", "experiments.qiskit_check"], False)
        )

    failures = []
    for label, args, required in steps:
        if not _run(args, label):
            failures.append(label)
            if required:
                print(f"\n关键步骤失败，中止：{label}")
                return 1

    print("\n=== 汇总 ===")
    for label, _, required in steps:
        state = "FAIL" if label in failures else "ok"
        print(f"  {state:4s} {label}{'' if required else '（可选）'}")

    print("\n产物：experiments/results/*.csv")
    print("      experiments/figures/*.pdf|png")
    return 0 if not [f for f in failures if f != "Qiskit 交叉验证"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
