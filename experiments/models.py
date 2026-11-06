"""解析模型：GHZ 教学电路在门级比特翻转噪声下的**精确**输出分布。

模型与主程序仿真器保持一致：电路为 ``H(q0)`` + ``CNOT(q0, qi)`` (i=1..n-1)，
每个门之后对它所作用的每一条线路施加一次比特翻转信道 ``B(p)``。

两个关键观察让我们可以完全绕开 2^n 维密度矩阵：

1. **H 之后的那次信道对布居没有贡献。** 此时 q0 处于 |+⟩，而
   ``B(p)|+⟩⟨+|`` 是 |+⟩ 与 |−⟩ 的混合，即 Z 方向退相干：布居仍是
   1/2、1/2，只有相干项乘 (1-2p)。测量在计算基下进行，因此它对输出
   分布没有影响。CNOT 只把 Z 误差传播成 Z⊗Z，同样不改布居。
2. **剩下的 X 型误差是线性的。** 控制位上的 X 误差会沿后续 CNOT
   传播成"控制位 + 之后所有目标位"的联合翻转，所以最终翻转图案是
   误差变量的 GF(2) 线性函数。于是"翻转图案的分布"可以用一个
   **两状态转移矩阵**（前缀校验位 c ∈ {0,1}）在 O(n) 时间内精确算出。

第 2 点正是本项目踩到的坑：如果天真地假设每条线路独立翻转、把接触
次数直接代入 ``compose_bitflip``，n≤3 时还能对上，n≥4 就会系统性
偏差（例如 n=4、p=0.1 时 TVD 算成 0.4486，真值是 0.45964）。
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def compose_bitflip(p: float, times: int) -> float:
    """连续施加 ``times`` 次比特翻转信道后的等效翻转概率。

    单次比特翻转满足 ``B(p1) ∘ B(p2) = B(p1 + p2 - 2 p1 p2)``，
    因此 m 次复合为 ``(1 - (1 - 2p)^m) / 2``。
    """
    if times <= 0:
        return 0.0
    return (1.0 - (1.0 - 2.0 * p) ** times) / 2.0


def tvd(prob_a: Iterable[float], prob_b: Iterable[float]) -> float:
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    return float(0.5 * np.sum(np.abs(a - b)))


# --------------------------------------------------------------------------
# 翻转图案分布：两状态转移
# --------------------------------------------------------------------------
def _contact_counts(num_wires: int) -> int:
    return max(num_wires - 1, 0)


def flip_pattern_probability(num_wires: int, p: float, pattern: int) -> float:
    """给定 n 位翻转图案 ``pattern``（整数编码，MSB 为 q0）的概率。

    记第 i 条 CNOT 之后控制位上的误差为 ``e_i``、目标位误差为 ``t_i``
    （均独立服从 Bern(p)），前缀校验位 ``c_k = e_1 ⊕ … ⊕ e_k``。则

    * 线路 0 的翻转 = ``c_{n-1}``；
    * 线路 i (i≥1) 的翻转 = ``t_i ⊕ c_{i-1}``。

    对给定图案做一次 O(n) 的 DP：``dp[c]`` 表示走到当前前缀校验位 c 时
    的加权概率。
    """
    wires = list(range(num_wires - 1, -1, -1))
    desired = [(pattern >> shift) & 1 for shift in wires]
    if num_wires == 1:
        return 1.0 if desired[0] == 0 else 0.0

    dp = np.array([1.0, 0.0])
    for index in range(1, num_wires):
        # 由 c_{index-1} 决定 t_index 取什么值时满足约束：t = v_index ⊕ c
        weight = np.array([1.0 - p, p])  # weight[c] = P(t = v_index ⊕ c)
        # 依据 c 选择对应权重
        w0 = 1.0 - p if desired[index] == 0 else p
        w1 = p if desired[index] == 0 else 1.0 - p
        dp = dp * np.array([w0, w1])
        # 再乘上这一步的控制位误差 e_index，把校验位推到 c_index
        dp = dp @ np.array([[1.0 - p, p], [p, 1.0 - p]])

    return float(dp[desired[0]])


def flip_pattern_distribution(num_wires: int, p: float) -> np.ndarray:
    """全部 2^n 个翻转图案的概率，索引与 ``format(i, '0nb')`` 一致。"""
    size = 2 ** num_wires
    return np.array(
        [flip_pattern_probability(num_wires, p, index) for index in range(size)],
        dtype=float,
    )


# --------------------------------------------------------------------------
# 输出分布
# --------------------------------------------------------------------------
def ghz_distribution(num_wires: int, p: float) -> np.ndarray:
    """GHZ 电路在门级比特翻转噪声下的精确输出分布（长度 2^n）。"""
    size = 2 ** num_wires
    flips = flip_pattern_distribution(num_wires, p)
    indices = np.arange(size)
    mirrored = indices ^ (size - 1)  # 与 |1..1> 分支的异或偏移
    return 0.5 * (flips + flips[mirrored])


def ideal_ghz_distribution(num_wires: int) -> np.ndarray:
    size = 2 ** num_wires
    distribution = np.zeros(size, dtype=float)
    distribution[0] = 0.5
    distribution[-1] = 0.5
    return distribution


def support_mass(num_wires: int, p: float) -> float:
    """理想支撑集 {0..0, 1..1} 上的概率质量，等于 1 - TVD。"""
    noisy = ghz_distribution(num_wires, p)
    return float(noisy[0] + noisy[-1])


def ghz_tvd(num_wires: int, p: float) -> float:
    """与理想分布的总变差距离（精确闭式，O(n) 求值）。"""
    return 1.0 - support_mass(num_wires, p)


def ghz_transfer_matrix(num_wires: int, p: float) -> np.ndarray:
    """噪声信道在输出分布上的作用矩阵：``A[s, b] = P(s | 理想输出 b)``。

    由于翻转图案是加性的，``A[s, b] = z[s ⊕ b]``，即一个循环矩阵。
    """
    flips = flip_pattern_distribution(num_wires, p)
    size = 2 ** num_wires
    indices = np.arange(size)
    return flips[indices[:, None] ^ indices[None, :]]


# --------------------------------------------------------------------------
# 判据
# --------------------------------------------------------------------------
def solve_threshold(
    func: Callable[[float], float],
    target: float,
    low: float = 0.0,
    high: float = 0.5,
    tolerance: float = 1e-12,
) -> float:
    """二分求 ``func(p) = target`` 的根（假定 func 在区间内单调递增）。"""
    if func(low) > target:
        return low
    if func(high) < target:
        return high
    for _ in range(200):
        mid = 0.5 * (low + high)
        if func(mid) < target:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    return 0.5 * (low + high)


def teaching_threshold(num_wires: int, epsilon: float = 0.1) -> float:
    """教学可演示边界 p*(n)：TVD 首次超过 epsilon 的噪声概率。"""
    return solve_threshold(lambda p: ghz_tvd(num_wires, p), epsilon)


def small_noise_slope(num_wires: int) -> float:
    """小噪声斜率：TVD ≈ slope · p，其中 slope = 2(n-1)。

    推导：低阶只保留"恰好一次 X 误差、且未被抵消"的项，
    支撑质量 ≈ 1 - 2(n-1)p，故 TVD ≈ 2(n-1)p。
    """
    return 2.0 * _contact_counts(num_wires)


def naive_independent_tvd(num_wires: int, p: float) -> float:
    """反例模型：错误地假设各线路独立翻转时的 TVD。

    保留它是为了在论文里量化"忽略传播相关性"带来的系统误差。
    """
    contacts = np.ones(num_wires, dtype=int)
    contacts[0] = _contact_counts(num_wires)
    q = np.array([compose_bitflip(p, int(m)) for m in contacts])

    size = 2 ** num_wires
    indices = np.arange(size)
    bits = (indices[:, None] >> np.arange(num_wires - 1, -1, -1)) & 1
    branch_zero = np.prod(np.where(bits == 1, q, 1.0 - q), axis=1)
    branch_one = np.prod(np.where(bits == 0, q, 1.0 - q), axis=1)
    distribution = 0.5 * (branch_zero + branch_one)
    return 1.0 - float(distribution[0] + distribution[-1])
