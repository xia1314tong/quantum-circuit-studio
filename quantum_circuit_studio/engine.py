"""量子电路仿真核心。

这里只放与界面无关的纯计算逻辑：门的应用顺序、噪声信道、
采样、以及概率分布指标。界面层（mixins/simulation.py）只负责
收集参数、展示结果与绘制图表。

把它单独抽出来的目的有三个：

1. 可测试：不依赖 Qt，可以直接用 pytest 验证数值正确性；
2. 可复用：批量实验脚本（experiments/）与主程序共用同一套门语义，
   避免"论文里的实验"和"软件里的结果"是两套实现；
3. 可替换：以后换仿真后端时，只需要改这一个文件。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pennylane as qml

NOISE_MODELS = ("bitflip", "phaseflip", "depolarizing")

#: 噪声信道构造函数映射：名称 -> (构造器, 是否支持 default.mixed)
_NOISE_OPERATIONS = {
    "bitflip": qml.BitFlip,
    "phaseflip": qml.PhaseFlip,
    "depolarizing": qml.DepolarizingChannel,
}

Gate = Dict[str, object]


def circuit_depth(gates: Sequence[Gate]) -> int:
    """按列号计算电路深度；空电路深度为 0。"""
    return max((int(gate.get("col", 0)) for gate in gates), default=-1) + 1


def tv_distance(prob_a: Iterable[float], prob_b: Iterable[float]) -> float:
    """总变差距离（Total Variation Distance），范围 0~1，越小越接近。"""
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    return float(0.5 * np.sum(np.abs(a - b)))


def shannon_entropy(probabilities: Iterable[float]) -> float:
    """香农熵（bit）；忽略小于 1e-12 的概率分量。"""
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 1e-12]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log2(values)))


def sample_counts(
    sampled: np.ndarray,
    labels: Sequence[str],
    num_wires: int,
) -> Tuple[Dict[str, int], np.ndarray, int]:
    """把 PennyLane 的采样结果转换成按计算基态统计的计数与频率。"""
    sampled = np.asarray(sampled, dtype=int)
    # shots=1 时 PennyLane 返回的是标量/0 维数组，先补成 1 维
    sampled = np.atleast_1d(sampled)
    if sampled.ndim == 1:
        sampled = sampled.reshape(-1, num_wires)
    if sampled.shape[0] == 0:
        raise ValueError("没有得到有效的测量结果。")

    counts = {label: 0 for label in labels}
    for row in sampled:
        bitstring = "".join(str(int(bit)) for bit in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1

    total = int(sampled.shape[0])
    frequencies = np.array(
        [counts[label] / total for label in labels],
        dtype=float,
    )
    return counts, frequencies, total


def apply_gates(
    gates: Sequence[Gate],
    noise_enabled: bool = False,
    noise_model: str = "bitflip",
    noise_probability: float = 0.0,
) -> None:
    """在 QNode 内部按列顺序施加量子门，可选地在每个门后加噪声。

    该函数必须在 ``@qml.qnode`` 装饰的函数体内调用。
    """
    ordered = sorted(
        enumerate(gates),
        key=lambda item: (int(item[1].get("col", 0)), item[0]),
    )

    for _, gate in ordered:
        name = gate["name"]
        wires_used: List[int] = []

        if name == "H":
            wire = int(gate["wire"])
            qml.Hadamard(wires=wire)
            wires_used = [wire]
        elif name == "X":
            wire = int(gate["wire"])
            qml.PauliX(wires=wire)
            wires_used = [wire]
        elif name == "Y":
            wire = int(gate["wire"])
            qml.PauliY(wires=wire)
            wires_used = [wire]
        elif name == "Z":
            wire = int(gate["wire"])
            qml.PauliZ(wires=wire)
            wires_used = [wire]
        elif name == "RX":
            wire = int(gate["wire"])
            qml.RX(float(gate.get("param", np.pi / 2)), wires=wire)
            wires_used = [wire]
        elif name == "RY":
            wire = int(gate["wire"])
            qml.RY(float(gate.get("param", np.pi / 2)), wires=wire)
            wires_used = [wire]
        elif name == "RZ":
            wire = int(gate["wire"])
            qml.RZ(float(gate.get("param", np.pi / 2)), wires=wire)
            wires_used = [wire]
        elif name == "CNOT":
            control = int(gate["control"])
            target = int(gate["target"])
            qml.CNOT(wires=[control, target])
            wires_used = [control, target]
        else:
            raise ValueError(f"未知的量子门：{name}")

        if noise_enabled and noise_probability > 0:
            operation = _NOISE_OPERATIONS.get(noise_model)
            if operation is None:
                raise ValueError(f"未知的噪声模型：{noise_model}")
            for wire in wires_used:
                operation(noise_probability, wires=wire)


@dataclass
class SimulationResult:
    """一次仿真的完整数值结果。"""

    num_wires: int
    gates: List[Gate]
    shots: int
    labels: List[str]
    state: np.ndarray
    ideal_probs: np.ndarray
    ideal_counts: Dict[str, int]
    ideal_frequencies: np.ndarray
    ideal_actual_shots: int
    noise_enabled: bool
    noise_model: str
    noise_probability: float
    depth: int
    ideal_entropy: float
    ideal_sample_distance: float
    noisy_probs: np.ndarray | None = None
    noisy_counts: Dict[str, int] | None = None
    noisy_frequencies: np.ndarray | None = None
    noisy_actual_shots: int | None = None

    # -- 派生指标 ---------------------------------------------------------
    @property
    def ideal_noisy_distance(self) -> float | None:
        if self.noisy_probs is None:
            return None
        return tv_distance(self.ideal_probs, self.noisy_probs)

    @property
    def noisy_entropy(self) -> float | None:
        if self.noisy_probs is None:
            return None
        return shannon_entropy(self.noisy_probs)

    @property
    def noisy_sample_distance(self) -> float | None:
        if self.noisy_probs is None or self.noisy_frequencies is None:
            return None
        return tv_distance(self.noisy_probs, self.noisy_frequencies)

    @property
    def distribution_fidelity(self) -> float:
        """计算基概率分布的保真度（Bhattacharyya 系数的平方）；未启用噪声时为 1。

        注意：这不是量子态保真度。它只比较 |振幅|² 构成的分布，因此**对相位
        错误完全不敏感**——例如相位翻转后真实态保真度约 0.5，这里仍返回 1.0。
        要判断相位或纠缠是否正确，需要比较密度矩阵，本模块不做这件事。
        """
        if self.noisy_probs is None:
            return 1.0
        overlap = float(np.sum(np.sqrt(self.ideal_probs * self.noisy_probs)))
        return overlap ** 2

    @property
    def fidelity(self) -> float:
        """向后兼容别名，等价于 ``distribution_fidelity``。"""
        return self.distribution_fidelity


def simulate_circuit(
    num_wires: int,
    gates: Sequence[Gate],
    shots: int = 1000,
    noise_enabled: bool = False,
    noise_model: str = "bitflip",
    noise_probability: float = 0.0,
) -> SimulationResult:
    """对给定电路做一次完整仿真：理想态、采样、可选的含噪声对比。"""
    if num_wires < 1:
        raise ValueError("量子比特数必须大于 0")

    gates = [dict(gate) for gate in gates]
    labels = [format(index, f"0{num_wires}b") for index in range(2 ** num_wires)]

    # 1. 理想电路：精确状态向量与理论概率
    ideal_device = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(ideal_device)
    def ideal_circuit():
        apply_gates(gates, noise_enabled=False)
        return qml.state(), qml.probs(wires=range(num_wires))

    state, ideal_probs = ideal_circuit()
    state = np.asarray(state, dtype=complex)
    ideal_probs = np.clip(np.asarray(ideal_probs, dtype=float), 0.0, None)

    norm = float(np.sum(ideal_probs))
    if norm <= 0:
        raise ValueError("理想电路计算得到的总概率无效。")
    ideal_probs = ideal_probs / norm

    # 2. 理想电路：有限次测量
    ideal_sample_device = qml.device("default.qubit", wires=num_wires, shots=shots)

    @qml.qnode(ideal_sample_device)
    def ideal_sample_circuit():
        apply_gates(gates, noise_enabled=False)
        return qml.sample(wires=range(num_wires))

    ideal_counts, ideal_frequencies, ideal_actual = sample_counts(
        ideal_sample_circuit(), labels, num_wires
    )

    # 3. 可选的含噪声仿真：default.qubit 不支持信道，必须用 default.mixed
    noisy_probs = noisy_frequencies = None
    noisy_counts = None
    noisy_actual = None

    if noise_enabled:
        noisy_device = qml.device("default.mixed", wires=num_wires)

        @qml.qnode(noisy_device)
        def noisy_circuit():
            apply_gates(gates, True, noise_model, noise_probability)
            return qml.probs(wires=range(num_wires))

        noisy_probs = np.clip(np.asarray(noisy_circuit(), dtype=float), 0.0, None)
        noisy_norm = float(np.sum(noisy_probs))
        if noisy_norm <= 0:
            raise ValueError("含噪声电路计算得到的总概率无效。")
        noisy_probs = noisy_probs / noisy_norm

        noisy_sample_device = qml.device(
            "default.mixed", wires=num_wires, shots=shots
        )

        @qml.qnode(noisy_sample_device)
        def noisy_sample_circuit():
            apply_gates(gates, True, noise_model, noise_probability)
            return qml.sample(wires=range(num_wires))

        noisy_counts, noisy_frequencies, noisy_actual = sample_counts(
            noisy_sample_circuit(), labels, num_wires
        )

    return SimulationResult(
        num_wires=num_wires,
        gates=gates,
        shots=shots,
        labels=labels,
        state=state,
        ideal_probs=ideal_probs,
        ideal_counts=ideal_counts,
        ideal_frequencies=ideal_frequencies,
        ideal_actual_shots=ideal_actual,
        noise_enabled=bool(noise_enabled),
        noise_model=noise_model,
        noise_probability=float(noise_probability),
        depth=circuit_depth(gates),
        ideal_entropy=shannon_entropy(ideal_probs),
        ideal_sample_distance=tv_distance(ideal_probs, ideal_frequencies),
        noisy_probs=noisy_probs,
        noisy_counts=noisy_counts,
        noisy_frequencies=noisy_frequencies,
        noisy_actual_shots=noisy_actual,
    )
