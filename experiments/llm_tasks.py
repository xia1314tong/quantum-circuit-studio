"""LLM 生成电路任务集：题目 + 期望态 + 三层校验。

每个任务给出：

* ``prompt``     —— 给模型的自然语言需求（中文，和产品里的 AI 助手一致）；
* ``num_wires``  —— 期望的比特数；
* ``expected``   —— 期望的末态（复数向量，按 q0 为最高位的顺序）。

期望态可以用参考电路（由本项目自己的引擎算出）给出，也可以直接数学构造
（W 态、QFT 这类用解析式更可靠）。判定时比较 ``|<psi|phi>|^2``，
因此对全局相位不敏感，但能区分 |+> 与 |-> 这种"概率相同、相位不同"的错误。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from quantum_circuit_studio.engine import simulate_circuit


def gate(name: str, wire: int, col: int, param: float = 0.0) -> dict:
    return {"name": name, "wire": wire, "col": col, "param": param}


def cnot(control: int, target: int, col: int) -> dict:
    return {"name": "CNOT", "control": control, "target": target, "col": col}


def _ghz_gates(num_wires: int) -> List[dict]:
    gates = [gate("H", 0, 0)]
    for index in range(1, num_wires):
        gates.append(cnot(0, index, index))
    return gates


def _layout_ghz_columns() -> Callable[[List[dict]], List[str]]:
    def check(gates: List[dict]) -> List[str]:
        errors: List[str] = []
        hadamard_columns = [
            int(g.get("col", 0))
            for g in gates
            if g["name"] == "H" and int(g.get("wire", -1)) == 0
        ]
        if not hadamard_columns:
            errors.append("缺少作用在 q0 上的 H 门")
        elif hadamard_columns[0] != 0:
            errors.append(f"q0 上的 H 门必须放在第 0 列，实际在第 {hadamard_columns[0]} 列")
        return errors

    return check


def _w_state(num_wires: int = 3) -> np.ndarray:
    state = np.zeros(2 ** num_wires, dtype=complex)
    for index in range(num_wires):
        # q0 是最高位：单激发态 |0..010..0>
        state[2 ** (num_wires - 1 - index)] = 1.0 / np.sqrt(num_wires)
    return state


def _qft_on(num_wires: int, input_index: int) -> np.ndarray:
    size = 2 ** num_wires
    omega = np.exp(2j * np.pi / size)
    column = np.array(
        [omega ** ((j * input_index) % size) for j in range(size)], dtype=complex
    )
    return column / np.sqrt(size)


@dataclass
class Task:
    key: str
    prompt: str
    num_wires: int
    gates: Optional[List[dict]] = None
    state: Optional[np.ndarray] = None
    layout: Optional[Callable[[List[dict]], List[str]]] = None
    tags: List[str] = field(default_factory=list)

    def expected_state(self) -> np.ndarray:
        if self.state is not None:
            return np.asarray(self.state, dtype=complex)
        assert self.gates is not None
        result = simulate_circuit(self.num_wires, self.gates, shots=1000)
        return np.asarray(result.state, dtype=complex)

    def expected_probabilities(self) -> np.ndarray:
        return np.abs(self.expected_state()) ** 2


TASKS: List[Task] = [
    Task(
        key="bell",
        prompt="用 2 个量子比特构造 Bell 态 (|00> + |11>)/sqrt(2)。",
        num_wires=2,
        gates=[gate("H", 0, 0), cnot(0, 1, 1)],
        tags=["基础", "纠缠"],
    ),
    Task(
        key="ghz3",
        prompt="用 3 个量子比特构造 GHZ 态 (|000> + |111>)/sqrt(2)。",
        num_wires=3,
        gates=_ghz_gates(3),
        tags=["基础", "纠缠"],
    ),
    Task(
        key="ghz4",
        prompt="用 4 个量子比特构造 GHZ 态 (|0000> + |1111>)/sqrt(2)。",
        num_wires=4,
        gates=_ghz_gates(4),
        tags=["基础", "纠缠"],
    ),
    Task(
        key="ghz5",
        prompt="用 5 个量子比特构造 GHZ 态 (|00000> + |11111>)/sqrt(2)。",
        num_wires=5,
        gates=_ghz_gates(5),
        tags=["规模", "纠缠"],
    ),
    Task(
        key="ghz5_layout",
        prompt=(
            "用 5 个量子比特构造 GHZ 态。硬性要求：作用在 q0 上的 H 门必须放在第 0 列，"
            "然后依次在第 1、2、3、4 列用 q0 控制 q1、q2、q3、q4 做 CNOT。"
        ),
        num_wires=5,
        gates=_ghz_gates(5),
        layout=_layout_ghz_columns(),
        tags=["规模", "约束"],
    ),
    Task(
        key="plus2",
        prompt="用 2 个量子比特制备态 (|00> + |10>)/sqrt(2)，即 q0 处于 |+>、q1 处于 |0>。",
        num_wires=2,
        gates=[gate("H", 0, 0)],
        tags=["基础", "单比特"],
    ),
    Task(
        key="uniform3",
        prompt="用 3 个量子比特制备所有 8 个计算基态的均匀叠加。",
        num_wires=3,
        gates=[gate("H", 0, 0), gate("H", 1, 0), gate("H", 2, 0)],
        tags=["基础", "叠加"],
    ),
    Task(
        key="w3",
        prompt="用 3 个量子比特制备 W 态 (|001> + |010> + |100>)/sqrt(3)。",
        num_wires=3,
        state=_w_state(3),
        tags=["困难", "多比特"],
    ),
    Task(
        key="bitflip1",
        prompt="用 1 个量子比特制备 |1> 态。",
        num_wires=1,
        gates=[gate("X", 0, 0)],
        tags=["基础", "单比特"],
    ),
    Task(
        key="ry_quarter",
        prompt="用 1 个量子比特制备态 cos(pi/8)|0> + sin(pi/8)|1>。",
        num_wires=1,
        gates=[gate("RY", 0, 0, np.pi / 4)],
        tags=["基础", "旋转门"],
    ),
    Task(
        key="qft2",
        prompt=(
            "对 2 个量子比特做 2 比特量子傅里叶变换，输入态是 |10>"
            "（即先在 q0 上加 X 门，再执行 2 比特 QFT）。"
        ),
        num_wires=2,
        state=_qft_on(2, 2),
        tags=["困难", "算法"],
    ),
    Task(
        key="minus1",
        prompt="用 1 个量子比特制备 (|0> - |1>)/sqrt(2)。",
        num_wires=1,
        gates=[gate("X", 0, 0), gate("H", 0, 1)],
        tags=["基础", "相位"],
    ),
    # 以下三个任务刻意提高难度／规模：它们能逼出"语法完全合法、但态错了"的结果，
    # 用来检验数值层（而不是只检验格式）才是这套闭环的真正价值所在。
    Task(
        key="w4",
        prompt="用 4 个量子比特制备单激发均匀叠加态 (|0001> + |0010> + |0100> + |1000>)/2。",
        num_wires=4,
        state=_w_state(4),
        tags=["困难", "多比特"],
    ),
    Task(
        key="qft3",
        prompt=(
            "对 3 个量子比特做 3 比特量子傅里叶变换，输入态是 |101>"
            "（即先在 q0、q2 上加 X 门，再执行 3 比特 QFT）。"
        ),
        num_wires=3,
        state=_qft_on(3, 5),
        tags=["困难", "算法"],
    ),
    Task(
        key="ghz6",
        prompt="用 6 个量子比特制备 GHZ 态 (|000000> + |111111>)/sqrt(2)。",
        num_wires=6,
        gates=_ghz_gates(6),
        tags=["规模", "多比特"],
    ),
    # 第二批：扩大任务多样性，覆盖相位、布局约束、4 比特算法与更大规模。
    Task(
        key="ghz8",
        prompt="用 8 个量子比特制备 GHZ 态 (|00000000> + |11111111>)/sqrt(2)。",
        num_wires=8,
        gates=_ghz_gates(8),
        tags=["规模", "多比特"],
    ),
    Task(
        key="uniform4",
        prompt="用 4 个量子比特制备全部 16 个计算基态的均匀叠加。",
        num_wires=4,
        gates=[gate("H", wire, 0) for wire in range(4)],
        tags=["基础", "叠加"],
    ),
    Task(
        key="plusi",
        prompt="用 1 个量子比特制备 (|0> + i|1>)/sqrt(2)。",
        num_wires=1,
        gates=[gate("H", 0, 0), gate("RZ", 0, 1, np.pi / 2)],
        tags=["基础", "相位"],
    ),
    Task(
        key="bell_minus",
        prompt="用 2 个量子比特制备 (|00> - |11>)/sqrt(2)。",
        num_wires=2,
        gates=[gate("H", 0, 0), cnot(0, 1, 1), gate("Z", 0, 2)],
        tags=["基础", "相位"],
    ),
    Task(
        key="bell_psi",
        prompt="用 2 个量子比特制备 (|01> - |10>)/sqrt(2)。",
        num_wires=2,
        gates=[gate("X", 1, 0), gate("H", 0, 1), cnot(0, 1, 2), gate("Z", 0, 3)],
        tags=["困难", "相位"],
    ),
    Task(
        key="partial_ent",
        prompt=(
            "用 2 个量子比特制备 cos(pi/8)|00> + sin(pi/8)|11>"
            "（即两比特部分纠缠态，Schmidt 角为 pi/8）。"
        ),
        num_wires=2,
        gates=[gate("RY", 0, 0, np.pi / 4), cnot(0, 1, 1)],
        tags=["困难", "旋转门"],
    ),
    Task(
        key="cluster3",
        prompt=(
            "用 3 个量子比特制备一维簇态：先在每个比特上加 H，"
            "再对相邻的 (q0,q1) 与 (q1,q2) 各施加一个 CZ 门。"
            "本工具没有 CZ 门，请用 H 与 CNOT 组合实现。"
        ),
        num_wires=3,
        gates=[
            gate("H", 0, 0),
            gate("H", 1, 0),
            gate("H", 2, 0),
            gate("H", 1, 1),
            cnot(0, 1, 2),
            gate("H", 1, 3),
            gate("H", 2, 4),
            cnot(1, 2, 5),
            gate("H", 2, 6),
        ],
        tags=["困难", "多比特"],
    ),
    Task(
        key="ghz4_layout",
        prompt=(
            "用 4 个量子比特制备 GHZ 态，并且必须满足布局约束："
            "H 门放在第 0 列，CNOT 依次放在第 1、2、3 列，且控制位依次为 q0、q1、q2。"
        ),
        num_wires=4,
        gates=_ghz_gates(4),
        layout=_layout_ghz_columns(),
        tags=["语义", "布局约束"],
    ),
    Task(
        key="qft4",
        prompt=(
            "对 4 个量子比特做 4 比特量子傅里叶变换，输入态是 |1000>"
            "（即先在 q0 上加 X 门，再执行 4 比特 QFT）。"
        ),
        num_wires=4,
        state=_qft_on(4, 8),
        tags=["困难", "算法"],
    ),
]
