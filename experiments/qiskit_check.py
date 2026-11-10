"""用 Qiskit 的线性代数接口独立交叉验证解析模型。

我们在**另一套库**（``qiskit.quantum_info`` 的 DensityMatrix / Operator /
Choi）里逐门重建同一条 GHZ 线路与同一套门级比特翻转信道，得到输出分布，
再与 :mod:`experiments.models` 的闭式结果逐点比较。回答的是审稿人最常问
的那类问题："结论是不是只在你自己的代码里成立。"

用法::

    python -m experiments.qiskit_check

需要 ``qiskit``（仅用于交叉验证，不是运行本项目的依赖）。
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

from . import models

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _embed(num_wires: int, wire: int, operator: np.ndarray) -> np.ndarray:
    """把单比特算子嵌入到 n 比特空间（wire=0 为最高位，与索引约定一致）。"""
    return np.kron(
        np.kron(np.eye(2 ** wire), operator), np.eye(2 ** (num_wires - wire - 1))
    )


def _cnot(num_wires: int, control: int, target: int) -> np.ndarray:
    dim = 2 ** num_wires
    matrix = np.zeros((dim, dim))
    for index in range(dim):
        bits = [(index >> (num_wires - 1 - k)) & 1 for k in range(num_wires)]
        if bits[control] == 1:
            bits[target] ^= 1
        out = 0
        for bit in bits:
            out = (out << 1) | bit
        matrix[out, index] = 1.0
    return matrix


def _channel(num_wires: int, wire: int, p: float):
    """n 比特空间里作用在第 wire 条线路上的比特翻转信道（Qiskit Kraus）。"""
    from qiskit.quantum_info import Kraus

    identity = np.eye(2, dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    return Kraus(
        [
            math.sqrt(1.0 - p) * _embed(num_wires, wire, identity),
            math.sqrt(p) * _embed(num_wires, wire, pauli_x),
        ]
    )


def qiskit_distribution(num_wires: int, p: float) -> np.ndarray:
    from qiskit.quantum_info import DensityMatrix, Operator

    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    state = DensityMatrix.from_label("0" * num_wires)

    state = state.evolve(Operator(_embed(num_wires, 0, hadamard)))
    state = state.evolve(_channel(num_wires, 0, p))

    for target in range(1, num_wires):
        state = state.evolve(Operator(_cnot(num_wires, 0, target)))
        state = state.evolve(_channel(num_wires, 0, p))
        state = state.evolve(_channel(num_wires, target, p))

    return np.real(np.diag(state.data))


def run_crosscheck() -> str:
    rows = []
    for num_wires in (2, 3, 4):
        for p in (0.02, 0.05, 0.1, 0.2):
            reference = qiskit_distribution(num_wires, p)
            analytic = models.ghz_distribution(num_wires, p)
            rows.append(
                [
                    num_wires,
                    p,
                    f"{models.tvd(reference, analytic):.3e}",
                    f"{models.tvd(reference, models.ideal_ghz_distribution(num_wires)):.10f}",
                    f"{models.ghz_tvd(num_wires, p):.10f}",
                ]
            )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "qiskit_crosscheck.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["num_wires", "p", "distribution_gap", "qiskit_tvd", "analytic_tvd"]
        )
        writer.writerows(rows)
    return path


def main() -> None:
    path = run_crosscheck()
    print(f"[ok] {os.path.relpath(path)}")
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            print(
                f"  n={row['num_wires']} p={row['p']}: "
                f"gap={row['distribution_gap']} "
                f"tvd={row['qiskit_tvd']} vs {row['analytic_tvd']}"
            )


if __name__ == "__main__":
    main()
