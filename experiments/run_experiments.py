"""生成论文所需全部数据。

用法::

    python -m experiments.run_experiments

输出 (均写入 ``experiments/results/``):

* ``validation.csv``    解析式 vs 主程序仿真（PennyLane）的逐点对照
* ``noise_law.csv``     全网格 TVD(p, n)
* ``threshold.csv``     教学可演示边界 p*(n)
* ``mitigation.csv``    四种缓解方法的 TVD(p)
* ``robustness.csv``    模型失配下的鲁棒性
* ``ablation.csv``      消融：单纯形约束 / 已知 p vs 拟合 p

所有结果都来自确定性计算（解析式或 shots=0 状态向量），不含采样噪声，
因此不需要随机种子。
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

# 允许直接以脚本方式运行（python experiments/run_experiments.py）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments import mitigation, models
else:
    from . import mitigation, models

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

NUM_WIRES = range(1, 9)
P_GRID = np.round(np.arange(0.0, 0.5001, 0.005), 4)
EPSILONS = (0.05, 0.10)


def _write_csv(name: str, header, rows) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def ghz_gates(num_wires: int):
    gates = [{"name": "H", "wire": 0, "col": 0, "param": math.pi / 2}]
    for index in range(1, num_wires):
        gates.append(
            {"name": "CNOT", "control": 0, "target": index, "col": index}
        )
    return gates


def run_validation() -> str:
    """解析式与主程序仿真引擎逐点对照。"""
    from quantum_circuit_studio import engine

    rows = []
    for num_wires in (2, 3, 4, 5):
        gates = ghz_gates(num_wires)
        for p in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3):
            simulation = engine.simulate_circuit(
                num_wires,
                gates,
                shots=1,
                noise_enabled=True,
                noise_model="bitflip",
                noise_probability=p,
            )
            analytic = models.ghz_tvd(num_wires, p)
            rows.append(
                [
                    num_wires,
                    p,
                    f"{analytic:.12f}",
                    f"{simulation.ideal_noisy_distance:.12f}",
                    f"{abs(analytic - simulation.ideal_noisy_distance):.3e}",
                ]
            )
    return _write_csv(
        "validation.csv",
        ["num_wires", "p", "analytic_tvd", "simulation_tvd", "abs_error"],
        rows,
    )


def run_noise_law() -> str:
    rows = []
    for num_wires in NUM_WIRES:
        for p in P_GRID:
            tvd = models.ghz_tvd(num_wires, float(p))
            rows.append(
                [num_wires, f"{p:.3f}", f"{tvd:.10f}", f"{1.0 - tvd:.10f}"]
            )
    return _write_csv(
        "noise_law.csv",
        ["num_wires", "p", "tvd", "support_mass"],
        rows,
    )


def run_thresholds() -> str:
    rows = []
    for num_wires in NUM_WIRES:
        contacts = max(num_wires - 1, 0)
        for epsilon in EPSILONS:
            threshold = models.teaching_threshold(num_wires, epsilon)
            asymptotic = epsilon / (2.0 * contacts) if contacts else float("inf")
            rows.append(
                [
                    num_wires,
                    epsilon,
                    contacts,
                    f"{threshold:.8f}",
                    f"{asymptotic:.8f}" if contacts else "inf",
                ]
            )
    return _write_csv(
        "threshold.csv",
        ["num_wires", "epsilon", "qubit0_contacts", "p_star", "p_star_asymptotic"],
        rows,
    )


def run_mitigation() -> str:
    """有限次采样（2×10^4 shots，8 个种子）下的缓解效果，给出均值与标准差。"""
    rows = []
    shots = 20000
    seeds = 8
    for num_wires in (2, 4, 6, 8):
        for p in np.round(np.arange(0.0, 0.3001, 0.01), 4):
            collected = {method: [] for method in mitigation.METHODS}
            for seed in range(seeds):
                results = mitigation.evaluate(
                    num_wires, float(p), shots=shots, seed=seed
                )
                for method in mitigation.METHODS:
                    collected[method].append(results[method])
            for method in mitigation.METHODS:
                values = np.asarray(collected[method], dtype=float)
                rows.append(
                    [
                        num_wires,
                        f"{p:.3f}",
                        method,
                        f"{values.mean():.8f}",
                        f"{values.std(ddof=1):.8f}",
                    ]
                )
    return _write_csv(
        "mitigation.csv",
        ["num_wires", "p", "method", "tvd_mean", "tvd_std"],
        rows,
    )


def run_robustness() -> str:
    rows = []
    for num_wires in (4, 6, 8):
        for p in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            for error in (-0.20, -0.10, 0.0, 0.10, 0.20):
                results = mitigation.evaluate(
                    num_wires, float(p), model_error=float(error)
                )
                rows.append(
                    [
                        num_wires,
                        f"{p:.3f}",
                        f"{error:+.2f}",
                        f"{results['none']:.8f}",
                        f"{results['model']:.8f}",
                    ]
                )
    return _write_csv(
        "robustness.csv",
        ["num_wires", "p", "model_error", "tvd_none", "tvd_model"],
        rows,
    )


def run_ablation() -> str:
    """消融：单纯形约束的作用，以及"已知 p"与"从数据拟合 p"的差别。

    在有限次测量下，反演矩阵的逆会放大采样噪声；不加约束时解会出现负分量，
    这正是约束存在的理由。无采样时两者完全一致（矩阵可逆，精确解即真值），
    因此这一节必须用 shots。
    """
    rows = []
    shots, seeds = 20000, 8
    for num_wires in (4, 6, 8):
        for p in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
            constrained, clipped, negative, fitted = [], [], [], []
            for seed in range(seeds):
                results = mitigation.evaluate(
                    num_wires, float(p), shots=shots, seed=seed
                )
                constrained.append(results["model"])
                clipped.append(results["model_clip"])
                negative.append(results["model_clip_negative_mass"])
                fitted.append(results["model_est"])
            rows.append(
                [
                    num_wires,
                    f"{p:.3f}",
                    f"{np.mean(constrained):.8f}",
                    f"{np.mean(clipped):.8f}",
                    f"{np.mean(negative):.8f}",
                    f"{np.mean(fitted):.8f}",
                ]
            )
    return _write_csv(
        "ablation.csv",
        [
            "num_wires",
            "p",
            "tvd_model_projected",
            "tvd_model_clipped",
            "negative_mass_clipped",
            "tvd_model_fitted_p",
        ],
        rows,
    )


def main() -> None:
    for task in (
        run_validation,
        run_noise_law,
        run_thresholds,
        run_mitigation,
        run_robustness,
        run_ablation,
    ):
        path = task()
        print(f"[ok] {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
