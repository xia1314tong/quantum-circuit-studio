"""GHZ 教学电路在门级比特翻转噪声下的判定工具。

平台用它回答界面上的一个具体问题：**当前比特数与噪声强度下，这次演示还能不能信？**

模型约定与引擎一致：每执行一个门，就在它涉及的每条线路上紧跟一个比特翻转信道。
对 ``H(q0) + CNOT(q0, qi)`` 构成的 GHZ 族，有两点使闭式求解成为可能：

1. ``H`` 之后的那次信道作用在 ``|+⟩`` 上，等价于相位阻尼，不改变计算基布居，
   因此不计入"可见"误差；
2. 其余 X 型误差沿后续 CNOT 线性传播，最终翻转图案是各误差变量的 GF(2) 线性函数。

``flip_pattern_probability`` 用两状态前缀 DP 在 O(n) 时间给出任意翻转图案的概率，
``ghz_tvd`` 与 ``teaching_threshold`` 在其上给出总变差距离与教学容限阈值。
"""

from __future__ import annotations


def compose_bitflip(p: float, times: int) -> float:
    """连续施加 ``times`` 次比特翻转后的等效翻转概率。

    单次信道满足 ``B(p1) ∘ B(p2) = B(p1 + p2 - 2 p1 p2)``，
    因此 m 次复合为 ``(1 - (1 - 2p)^m) / 2``。
    """
    if times <= 0:
        return 0.0
    return (1.0 - (1.0 - 2.0 * p) ** times) / 2.0


def flip_pattern_probability(num_wires: int, p: float, pattern: int) -> float:
    """给定 n 位翻转图案 ``pattern``（整数编码，MSB 为 q0）的概率。

    记第 i 条 CNOT 之后控制位上的误差为 ``e_i``、目标位为 ``t_i``（独立服从 Bern(p)），
    前缀校验位 ``c_k = e_1 ⊕ … ⊕ e_k``。则线路 0 的翻转是 ``c_{n-1}``，
    线路 i (i≥1) 的翻转是 ``t_i ⊕ c_{i-1}``。
    """
    desired = [(pattern >> shift) & 1 for shift in range(num_wires - 1, -1, -1)]
    if num_wires <= 1:
        return 1.0 if desired[0] == 0 else 0.0

    keep, flip = 1.0 - p, p
    dp = [1.0, 0.0]  # dp[c]：前缀校验位为 c 的加权概率
    for index in range(1, num_wires):
        want = desired[index]
        w0 = keep if want == 0 else flip
        w1 = flip if want == 0 else keep
        dp = [dp[0] * w0, dp[1] * w1]
        # 再乘上这一步的控制位误差，把校验位推到 c_index
        dp = [dp[0] * keep + dp[1] * flip, dp[0] * flip + dp[1] * keep]

    return dp[desired[0]]


def support_mass(num_wires: int, p: float) -> float:
    """理想分布两个非零振幅（全 0 与全 1）在含噪分布下的总质量。"""
    if num_wires <= 0:
        return 1.0
    full = (1 << num_wires) - 1
    return flip_pattern_probability(num_wires, p, 0) + flip_pattern_probability(num_wires, p, full)


def ghz_tvd(num_wires: int, p: float) -> float:
    """GHZ 电路含噪分布与理想分布的总变差距离。"""
    return max(0.0, 1.0 - support_mass(num_wires, p))


def teaching_threshold(num_wires: int, epsilon: float = 0.1) -> float:
    """使 ``ghz_tvd(n, p)`` 首次超过 ``epsilon`` 的噪声强度 p\\*（二分，容限 1e-12）。

    ``num_wires <= 1`` 时没有 CNOT，TVD 恒为 0，返回可达上界 0.5。
    """
    if num_wires <= 1:
        return 0.5
    low, high = 0.0, 0.5
    if ghz_tvd(num_wires, high) <= epsilon:
        return high
    for _ in range(200):
        mid = 0.5 * (low + high)
        if ghz_tvd(num_wires, mid) <= epsilon:
            low = mid
        else:
            high = mid
        if high - low < 1e-12:
            break
    return 0.5 * (low + high)
