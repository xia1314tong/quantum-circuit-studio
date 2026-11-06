"""误差缓解：已知信道反演 vs 标准基线。

在同一套噪声模型（门级比特翻转，见 :mod:`experiments.models`）下比较：

* ``none``    不做缓解；
* ``readout`` 标准 CompleteMeasFitter 式方法——制备计算基态、统计混淆
  矩阵、反演。本模型中测量本身无噪声，标定矩阵是单位阵，因此它捕获不到
  任何门层噪声（这本身就是一条可写的结论）；
* ``zne``     零噪声外推（噪声放大 1×/2×/3× + 线性外推），模型无关基线；
* ``model``   本文方法：用已知信道构造转移矩阵 ``A``，解带单纯形约束的
  最小二乘 ``min ||A q - p_meas||``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .models import (
    flip_pattern_distribution,
    ghz_distribution,
    ghz_transfer_matrix,
    ideal_ghz_distribution,
)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """把向量投影到概率单纯形上（Duchi et al., 2008）。"""
    n = vector.size
    u = np.sort(vector)[::-1]
    cumulative = np.cumsum(u) - 1.0
    indices = np.arange(1, n + 1)
    valid = u - cumulative / indices > 0
    rho = np.nonzero(valid)[0][-1]
    theta = cumulative[rho] / (rho + 1.0)
    return np.maximum(vector - theta, 0.0)


def readout_confusion_matrix(num_wires: int) -> np.ndarray:
    """CompleteMeasFitter 式标定矩阵：本模型中测量无噪声，故为单位阵。"""
    return np.eye(2 ** num_wires)


def channel_matrix(num_wires: int, p: float) -> np.ndarray:
    """噪声信道的分布层转移矩阵（精确，循环结构）。"""
    return ghz_transfer_matrix(num_wires, p)


# --------------------------------------------------------------------------
# 缓解方法
# --------------------------------------------------------------------------
def mitigate(measured: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """已知信道反演 + 单纯形投影。"""
    solution, *_ = np.linalg.lstsq(matrix, measured, rcond=None)
    return project_to_simplex(solution)


def mitigate_raw(measured: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """消融对照：只做最小二乘反演，不加单纯形约束。

    噪声较大时解会出现负分量、也不再归一，这正说明约束不是"美化"而是必需的。
    """
    solution, *_ = np.linalg.lstsq(matrix, measured, rcond=None)
    return np.asarray(solution, dtype=float)


def zne_support_mass(
    num_wires: int,
    p: float,
    scale_factors: Sequence[float] = (1.0, 2.0, 3.0),
) -> float:
    """零噪声外推：放大噪声后对支撑质量做线性外推到 c=0。"""
    masses = []
    for scale in scale_factors:
        scaled_p = min(p * scale, 0.5)
        distribution = ghz_distribution(num_wires, scaled_p)
        masses.append(float(distribution[0] + distribution[-1]))
    coefficients = np.polyfit(
        np.asarray(scale_factors, dtype=float), np.asarray(masses), deg=1
    )
    return float(np.clip(np.polyval(coefficients, 0.0), 0.0, 1.0))


@dataclass
class MitigationRow:
    num_wires: int
    p: float
    method: str
    tvd: float


METHODS = ("none", "readout", "zne", "model", "model_est")


def estimate_noise(num_wires: int, measured: np.ndarray) -> float:
    """从实测分布反推噪声强度 p（匹配支撑质量，二分求解）。

    这一步让 ``model_est`` 方法不需要"被告知"真实 p——它只从数据里
    估计参数，再反演，因此比 ``model`` 更接近真实实验流程。
    """
    from .models import support_mass

    observed = float(measured[0] + measured[-1])
    low, high = 0.0, 0.5
    if observed >= support_mass(num_wires, 0.0):
        return 0.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if support_mass(num_wires, mid) > observed:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def evaluate(
    num_wires: int,
    p: float,
    model_error: float = 0.0,
    shots: int | None = None,
    seed: int | None = None,
) -> Dict[str, float]:
    """在单点 (n, p) 上给出各方法缓解后的 TVD。

    ``shots``  给定时按多项分布对"实测分布"做有限次采样（更接近真实实验），
                否则使用精确分布。
    ``model_error`` 是"已知模型"的参数估计相对误差（0.1 表示误以为噪声是
                1.1p），用于鲁棒性分析。
    """
    ideal = ideal_ghz_distribution(num_wires)
    exact = ghz_distribution(num_wires, p)
    measured = exact
    if shots is not None:
        rng = np.random.default_rng(seed)
        measured = rng.multinomial(shots, exact) / float(shots)

    results: Dict[str, float] = {}
    results["none"] = float(np.sum(np.abs(ideal - measured)) / 2)

    readout = mitigate(measured, readout_confusion_matrix(num_wires))
    results["readout"] = float(np.sum(np.abs(ideal - readout)) / 2)

    results["zne"] = float(np.clip(1.0 - zne_support_mass(num_wires, p), 0.0, 1.0))

    estimated_p = float(np.clip(p * (1.0 + model_error), 0.0, 0.5))
    assumed = mitigate(measured, channel_matrix(num_wires, estimated_p))
    results["model"] = float(np.sum(np.abs(ideal - assumed)) / 2)

    fitted_p = estimate_noise(num_wires, measured)
    fitted = mitigate(measured, channel_matrix(num_wires, fitted_p))
    results["model_est"] = float(np.sum(np.abs(ideal - fitted)) / 2)

    # 消融：同样反演，但不加单纯形约束（负分量裁零后重新归一）
    raw = mitigate_raw(measured, channel_matrix(num_wires, estimated_p))
    clipped = np.clip(raw, 0.0, None)
    total = float(clipped.sum())
    normalised = clipped / total if total > 0 else clipped
    results["model_raw"] = float(np.sum(np.abs(ideal - normalised)) / 2)
    results["model_raw_negative_mass"] = float(-np.clip(raw, None, 0.0).sum())

    return results
