"""仿真核心（engine.py）的数值正确性测试。

这些断言不依赖界面，直接对照解析解，用来证明"软件算得对"，
而不是"软件跑起来了"。
"""
import math

import numpy as np
import pytest

from quantum_circuit_studio import engine


def bell_gates():
    return [
        {"name": "H", "wire": 0, "col": 0, "param": math.pi / 2},
        {"name": "CNOT", "control": 0, "target": 1, "col": 1},
    ]


def ghZ_gates(n):
    """n 比特 GHZ 态：H(q0) 后依次 CNOT(0, i)。"""
    gates = [{"name": "H", "wire": 0, "col": 0, "param": math.pi / 2}]
    for i in range(1, n):
        gates.append({"name": "CNOT", "control": 0, "target": i, "col": i})
    return gates


# --------------------------------------------------------------------------
# 1. 理想态：与解析解逐元素比对
# --------------------------------------------------------------------------
def test_bell_state_matches_analytic():
    result = engine.simulate_circuit(2, bell_gates(), shots=100)

    assert np.allclose(result.ideal_probs, [0.5, 0.0, 0.0, 0.5], atol=1e-9)
    assert np.isclose(result.ideal_probs.sum(), 1.0, atol=1e-12)
    # |Φ+> = (|00> + |11>)/√2
    assert np.allclose(result.state, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], atol=1e-9)
    assert np.isclose(result.ideal_entropy, 1.0, atol=1e-9)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_ghz_state_has_two_nonzero_amplitudes(n):
    result = engine.simulate_circuit(n, ghZ_gates(n), shots=50)

    nonzero = np.flatnonzero(result.ideal_probs > 1e-8)
    labels = {result.labels[i] for i in nonzero}

    assert labels == {"0" * n, "1" * n}
    assert np.allclose(result.ideal_probs[nonzero], [0.5, 0.5], atol=1e-9)


def test_x_gate_flips_single_qubit():
    result = engine.simulate_circuit(
        1, [{"name": "X", "wire": 0, "col": 0}], shots=10
    )
    assert np.allclose(result.ideal_probs, [0.0, 1.0], atol=1e-12)


def test_rotation_gate_matches_formula():
    theta = 0.7
    result = engine.simulate_circuit(
        1, [{"name": "RY", "wire": 0, "col": 0, "param": theta}], shots=10
    )
    expected = [math.cos(theta / 2) ** 2, math.sin(theta / 2) ** 2]
    assert np.allclose(result.ideal_probs, expected, atol=1e-12)


def test_gate_order_follows_column_not_list_order():
    # 列表顺序故意打乱：X 在 col=0，H 在 col=1，正确顺序应为 X 之后 H
    gates = [
        {"name": "H", "wire": 0, "col": 1, "param": math.pi / 2},
        {"name": "X", "wire": 0, "col": 0},
    ]
    result = engine.simulate_circuit(1, gates, shots=10)
    # H|1> = (|0> - |1>)/√2，概率仍各半，但相位为负
    assert np.allclose(result.ideal_probs, [0.5, 0.5], atol=1e-12)
    assert result.state[1].real < 0


# --------------------------------------------------------------------------
# 2. 噪声：物理约束
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", engine.NOISE_MODELS)
def test_noise_keeps_distribution_normalized(model):
    result = engine.simulate_circuit(
        2,
        bell_gates(),
        shots=100,
        noise_enabled=True,
        noise_model=model,
        noise_probability=0.2,
    )
    assert np.isclose(float(result.noisy_probs.sum()), 1.0, atol=1e-9)


@pytest.mark.parametrize("model", engine.NOISE_MODELS)
def test_noise_monotonically_degrades_fidelity(model):
    """噪声概率增大时，含噪分布与理想分布的差异应单调（不降）增大。"""
    distances = []
    for p in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]:
        result = engine.simulate_circuit(
            2,
            bell_gates(),
            shots=100,
            noise_enabled=True,
            noise_model=model,
            noise_probability=p,
        )
        distances.append(result.ideal_noisy_distance)

    assert distances[0] == pytest.approx(0.0, abs=1e-12)
    assert all(b >= a - 1e-9 for a, b in zip(distances, distances[1:])), distances


def test_bitflip_at_half_probability_on_bell_state():
    """p=0.5 的比特翻转把 |Φ+> 完全抹平为四个基态各 1/4。"""
    result = engine.simulate_circuit(
        2,
        bell_gates(),
        shots=200,
        noise_enabled=True,
        noise_model="bitflip",
        noise_probability=0.5,
    )
    assert np.allclose(result.noisy_probs, [0.25] * 4, atol=1e-9)
    assert result.ideal_noisy_distance == pytest.approx(0.5, abs=1e-9)


def test_bitflip_at_quarter_probability_matches_closed_form():
    """p=0.25 时（每个门对其涉及的每条线路各加一次信道）分布为
    [0.3125, 0.1875, 0.1875, 0.3125]，与 TVD=0.375 对应。"""
    result = engine.simulate_circuit(
        2,
        bell_gates(),
        shots=200,
        noise_enabled=True,
        noise_model="bitflip",
        noise_probability=0.25,
    )
    assert np.allclose(result.noisy_probs, [0.3125, 0.1875, 0.1875, 0.3125], atol=1e-9)
    assert result.ideal_noisy_distance == pytest.approx(0.375, abs=1e-9)


def test_noise_disabled_skips_mixed_device():
    result = engine.simulate_circuit(2, bell_gates(), shots=10)
    assert result.noisy_probs is None
    assert result.ideal_noisy_distance is None
    assert result.fidelity == 1.0


# --------------------------------------------------------------------------
# 3. 采样与指标
# --------------------------------------------------------------------------
def test_sampling_counts_are_consistent():
    result = engine.simulate_circuit(2, bell_gates(), shots=500)
    assert sum(result.ideal_counts.values()) == 500
    assert np.isclose(result.ideal_frequencies.sum(), 1.0, atol=1e-12)
    # Bell 态只可能测到 00 / 11
    assert result.ideal_counts["01"] == 0
    assert result.ideal_counts["10"] == 0


def test_sampling_converges_to_theory():
    """shots 增大时，采样频率与理论概率的总变差距离应下降。"""
    np.random.seed(20261104)  # 采样是随机的，固定种子避免偶发失败
    coarse = engine.simulate_circuit(2, bell_gates(), shots=200)
    fine = engine.simulate_circuit(2, bell_gates(), shots=20000)
    assert fine.ideal_sample_distance < 0.02
    assert fine.ideal_sample_distance < coarse.ideal_sample_distance


def test_tv_distance_and_entropy_bounds():
    assert engine.tv_distance([1, 0], [1, 0]) == 0.0
    assert engine.tv_distance([1, 0], [0, 1]) == 1.0
    assert engine.shannon_entropy([1.0]) == 0.0
    assert engine.shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert engine.shannon_entropy([0.25] * 4) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# 4. 输入校验
# --------------------------------------------------------------------------
def test_unknown_gate_raises():
    with pytest.raises(ValueError):
        engine.simulate_circuit(1, [{"name": "FOO", "wire": 0, "col": 0}], shots=10)


def test_unknown_noise_model_raises():
    with pytest.raises(ValueError):
        engine.simulate_circuit(
            1,
            [{"name": "H", "wire": 0, "col": 0}],
            shots=10,
            noise_enabled=True,
            noise_model="nope",
            noise_probability=0.1,
        )


def test_invalid_qubit_count_raises():
    with pytest.raises(ValueError):
        engine.simulate_circuit(0, [], shots=10)
