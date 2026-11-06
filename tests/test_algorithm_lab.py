"""量子算法实验室的数值验证。

四个算法都是纯 numpy 实现（不经过 PennyLane），因此可以直接
和解析解、DFT 矩阵对照，作为"软件里展示的数值是对的"的证据。
"""
import math

import numpy as np
import pytest

from quantum_circuit_studio.algorithm_lab import AlgorithmLabDialog as Lab


# --------------------------------------------------------------------------
# Grover
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4])
def test_grover_recommended_rounds(n):
    theta = math.asin(math.sqrt(1 / 2 ** n))
    expected = max(0, int(round(math.pi / (4 * theta) - 0.5)))

    _, _, recommended, _ = Lab._grover(n, format(2 ** n - 1, f"0{n}b"), 0)

    assert recommended == expected


@pytest.mark.parametrize("n", [2, 3, 4])
def test_grover_amplifies_target_to_closed_form(n):
    target = format(1, f"0{n}b")
    rounds = 2
    state, history, _, idx = Lab._grover(n, target, rounds)

    theta = math.asin(math.sqrt(1 / 2 ** n))
    expected = math.sin((2 * rounds + 1) * theta) ** 2

    assert np.isclose(abs(state[idx]) ** 2, expected, atol=1e-9)
    # 归一化在每一步都应保持
    for step in history:
        assert np.isclose(float(np.vdot(step, step).real), 1.0, atol=1e-9)


def test_grover_success_probability_beats_classical_baseline():
    n = 4
    state, _, _, idx = Lab._grover(n, format(9, f"0{n}b"), 0)
    assert abs(state[idx]) ** 2 > 1 / 2 ** n * 5


# --------------------------------------------------------------------------
# Deutsch-Jozsa
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4])
def test_deutsch_jozsa_constant_is_all_zero(n):
    state, values = Lab._deutsch_jozsa(n, balanced=False)
    assert np.all(values == 0)
    assert np.isclose(abs(state[0]) ** 2, 1.0, atol=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_deutsch_jozsa_balanced_concentrates_on_last_bit(n):
    state, values = Lab._deutsch_jozsa(n, balanced=True)
    assert set(np.unique(values)) == {0, 1}
    assert np.isclose(abs(state[1]) ** 2, 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# QFT
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4])
def test_qft_matches_dft_matrix(n):
    rng = np.random.default_rng(42)
    vector = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    vector /= np.linalg.norm(vector)

    transformed, matrix = Lab._qft(vector)

    assert np.allclose(transformed, matrix @ vector)
    # DFT 矩阵应为酉矩阵
    assert np.allclose(matrix.conj().T @ matrix, np.eye(2 ** n), atol=1e-9)
    assert np.isclose(np.linalg.norm(transformed), 1.0, atol=1e-9)


def test_qft_of_zero_state_is_uniform():
    n = 3
    vector = np.zeros(2 ** n, dtype=complex)
    vector[0] = 1.0

    transformed, _ = Lab._qft(vector)

    assert np.allclose(np.abs(transformed) ** 2, 1 / 2 ** n, atol=1e-9)


# --------------------------------------------------------------------------
# 量子隐形传态
# --------------------------------------------------------------------------
@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (math.pi / 2, 0.0), (1.1, 0.4), (2.3, 5.0)])
def test_teleportation_restores_input_state(theta, phi):
    input_state = np.array(
        [math.cos(theta / 2), np.exp(1j * phi) * math.sin(theta / 2)],
        dtype=complex,
    )

    branches = Lab._teleport_branches(input_state)

    assert len(branches) == 4
    # 四个分支的概率之和必须是 1
    assert math.isclose(sum(b[1] for b in branches), 1.0, abs_tol=1e-9)
    # 每个分支修正后的 Bob 态都必须等于输入态（允许整体相位）
    for _, _, bob, fidelity in branches:
        assert math.isclose(fidelity, 1.0, abs_tol=1e-9)
        overlap = abs(np.vdot(input_state, bob))
        assert math.isclose(overlap, 1.0, abs_tol=1e-9)
