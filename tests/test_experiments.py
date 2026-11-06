"""实验层测试：解析模型、判据与误差缓解。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiments import mitigation, models
from quantum_circuit_studio import engine


def ghz_gates(num_wires: int):
    gates = [{"name": "H", "wire": 0, "col": 0, "param": math.pi / 2}]
    for index in range(1, num_wires):
        gates.append({"name": "CNOT", "control": 0, "target": index, "col": index})
    return gates


# --------------------------------------------------------------------------
# 解析模型
# --------------------------------------------------------------------------
@pytest.mark.parametrize("num_wires", [2, 3, 4])
@pytest.mark.parametrize("p", [0.02, 0.1, 0.3])
def test_analytic_matches_simulator(num_wires, p):
    """解析闭式必须与主程序仿真逐点一致（本项目的核心可复现性主张）。"""
    simulation = engine.simulate_circuit(
        num_wires,
        ghz_gates(num_wires),
        shots=1,
        noise_enabled=True,
        noise_model="bitflip",
        noise_probability=p,
    )
    assert simulation.ideal_noisy_distance == pytest.approx(
        models.ghz_tvd(num_wires, p), abs=1e-10
    )


@pytest.mark.parametrize("num_wires", [1, 2, 5, 8])
@pytest.mark.parametrize("p", [0.0, 0.05, 0.25, 0.5])
def test_distribution_is_valid(num_wires, p):
    distribution = models.ghz_distribution(num_wires, p)
    assert distribution.shape == (2 ** num_wires,)
    assert np.all(distribution >= -1e-15)
    assert distribution.sum() == pytest.approx(1.0, abs=1e-12)


def test_no_noise_gives_zero_distance():
    for num_wires in (1, 2, 4):
        assert models.ghz_tvd(num_wires, 0.0) == pytest.approx(0.0, abs=1e-15)


def test_single_wire_is_immune_to_bitflip():
    """n=1 时电路只有 H，比特翻转等价于相位阻尼，分布不变。"""
    assert models.ghz_tvd(1, 0.4) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("num_wires", [2, 4, 6])
def test_tvd_monotone_in_p(num_wires):
    values = [models.ghz_tvd(num_wires, p) for p in (0.0, 0.05, 0.1, 0.2, 0.3)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_transfer_matrix_matches_distribution():
    num_wires, p = 4, 0.05
    matrix = models.ghz_transfer_matrix(num_wires, p)
    assert matrix.shape == (16, 16)
    assert np.allclose(matrix.sum(axis=0), 1.0, atol=1e-12)
    represented = matrix @ models.ideal_ghz_distribution(num_wires)
    assert np.allclose(represented, models.ghz_distribution(num_wires, p), atol=1e-12)


# --------------------------------------------------------------------------
# "坑"的回归测试：独立翻转近似在 n>=4 时失效
# --------------------------------------------------------------------------
def test_independent_flip_approximation_fails_from_four_qubits():
    exact = models.ghz_tvd(4, 0.1)
    naive = models.naive_independent_tvd(4, 0.1)
    assert exact == pytest.approx(0.45964, abs=1e-6)
    assert abs(exact - naive) > 1e-3


def test_independent_flip_approximation_coincides_up_to_three_qubits():
    for num_wires in (2, 3):
        assert models.naive_independent_tvd(
            num_wires, 0.1
        ) == pytest.approx(models.ghz_tvd(num_wires, 0.1), abs=1e-12)


# --------------------------------------------------------------------------
# 判据
# --------------------------------------------------------------------------
def test_threshold_definition():
    num_wires, epsilon = 5, 0.1
    threshold = models.teaching_threshold(num_wires, epsilon)
    assert models.ghz_tvd(num_wires, threshold) == pytest.approx(epsilon, abs=1e-9)


def test_threshold_decreases_with_qubits():
    thresholds = [models.teaching_threshold(n, 0.1) for n in range(2, 9)]
    assert all(b < a for a, b in zip(thresholds, thresholds[1:]))


def test_asymptotic_threshold_is_accurate_for_large_n():
    for num_wires in (6, 7, 8):
        exact = models.teaching_threshold(num_wires, 0.1)
        asymptotic = 0.1 / (2.0 * (num_wires - 1))
        assert exact == pytest.approx(asymptotic, rel=0.1)


def test_small_noise_slope():
    for num_wires in (2, 4, 8):
        slope = models.small_noise_slope(num_wires)
        p = 1e-4
        assert models.ghz_tvd(num_wires, p) / p == pytest.approx(slope, rel=1e-2)


# --------------------------------------------------------------------------
# 误差缓解
# --------------------------------------------------------------------------
def test_mitigation_recovers_distribution():
    num_wires, p = 6, 0.05
    ideal = models.ideal_ghz_distribution(num_wires)
    noisy = models.ghz_distribution(num_wires, p)
    recovered = mitigation.mitigate(noisy, models.ghz_transfer_matrix(num_wires, p))
    assert models.tvd(ideal, recovered) < 1e-8


def test_mitigation_beats_baselines_with_finite_shots():
    num_wires, p = 6, 0.05
    results = mitigation.evaluate(num_wires, p, shots=20000, seed=0)
    assert results["none"] > 0.3
    assert results["readout"] == pytest.approx(results["none"], abs=1e-12)
    assert results["zne"] < results["none"]
    assert results["model_est"] < 0.02
    assert results["model_est"] <= results["zne"]


def test_zne_improves_support_mass():
    num_wires, p = 4, 0.1
    plain = models.support_mass(num_wires, p)
    extrapolated = mitigation.zne_support_mass(num_wires, p)
    assert extrapolated > plain
    assert extrapolated <= 1.0


def test_estimate_noise_recovers_parameter():
    num_wires, p = 5, 0.08
    rng = np.random.default_rng(7)
    measured = rng.multinomial(20000, models.ghz_distribution(num_wires, p)) / 20000
    assert mitigation.estimate_noise(num_wires, measured) == pytest.approx(p, rel=0.05)


def test_simplex_projection():
    vector = np.array([0.6, 0.5, -0.1])
    projected = mitigation.project_to_simplex(vector)
    assert projected.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(projected >= 0.0)

def test_qiskit_crosscheck_agrees():
    """独立实现（Qiskit quantum_info）与解析模型逐点一致。"""
    pytest.importorskip("qiskit")
    from experiments.models import ghz_distribution
    from experiments.qiskit_check import qiskit_distribution

    for n in (2, 3):
        for p in (0.02, 0.1):
            ours = ghz_distribution(n, p)
            theirs = qiskit_distribution(n, p)
            assert np.max(np.abs(ours - theirs)) < 1e-12

def test_simplex_constraint_matters_under_sampling_noise():
    """有限次测量下，不加单纯形约束会让反演出现负概率、误差显著变大。"""
    results = mitigation.evaluate(8, 0.10, shots=20000, seed=1)
    assert results["model_raw_negative_mass"] > 0.05
    assert results["model_raw"] > 3 * results["model"]
    assert results["model"] < 0.05

def test_llm_validator_layers():
    """三层校验各自能拦下对应类型的错误。"""
    from experiments import llm_loop, llm_tasks

    task = [item for item in llm_tasks.TASKS if item.key == "bell"][0]
    good = {
        "num_wires": 2,
        "gates": [
            {"name": "H", "wire": 0, "col": 0},
            {"name": "CNOT", "control": 0, "target": 1, "col": 1},
        ],
    }
    assert llm_loop.validate(good, task)["passed"]

    bad_cnot = {"num_wires": 2, "gates": [{"name": "CNOT", "control": 1, "target": 1, "col": 0}]}
    assert llm_loop.failure_layer(llm_loop.validate(bad_cnot, task)) == "syntax"

    nested = {
        "num_wires": 2,
        "gates": [
            {"name": "H", "wire": 0, "col": 0},
            {"name": "X", "wire": 0, "col": 0},
        ],
    }
    assert llm_loop.failure_layer(llm_loop.validate(nested, task)) == "syntax"

    wrong_wires = {"num_wires": 3, "gates": [{"name": "H", "wire": 0, "col": 0}]}
    assert llm_loop.failure_layer(llm_loop.validate(wrong_wires, task)) == "semantic"

    wrong_state = {"num_wires": 2, "gates": [{"name": "H", "wire": 0, "col": 0}]}
    assert llm_loop.failure_layer(llm_loop.validate(wrong_state, task)) == "numeric"


def test_llm_parse_tolerates_code_fence():
    from experiments.llm_loop import parse_json

    payload, errors = parse_json('```json\n{"num_wires": 1, "gates": []}\n```')
    assert not errors
    assert payload["num_wires"] == 1

    assert parse_json("我不知道")[0] is None


def test_llm_task_expected_states_are_normalised():
    import numpy as np

    from experiments import llm_tasks

    for task in llm_tasks.TASKS:
        state = task.expected_state()
        assert state.shape == (2 ** task.num_wires,)
        assert np.isclose(np.vdot(state, state).real, 1.0, atol=1e-9)


def test_llm_arms_and_statistics():
    from experiments.llm_loop import compare_arms, mcnemar_exact

    # 配对统计：同一 (task, rep) 才成对，oneshot 不参与两两比较
    rows = [
        {"arm": "resample", "task": "a", "rep": 0, "round": 1, "passed": 0},
        {"arm": "resample", "task": "a", "rep": 1, "round": 1, "passed": 1},
        {"arm": "resample", "task": "b", "rep": 0, "round": 1, "passed": 1},
        {"arm": "feedback", "task": "a", "rep": 0, "round": 1, "passed": 0},
        {"arm": "feedback", "task": "a", "rep": 0, "round": 2, "passed": 1},
        {"arm": "feedback", "task": "a", "rep": 1, "round": 1, "passed": 1},
        {"arm": "feedback", "task": "b", "rep": 0, "round": 1, "passed": 1},
    ]
    stats = compare_arms(rows)
    assert len(stats) == 1
    entry = stats[0]
    assert entry["pairs"] == 3
    assert entry["rate_a"] == pytest.approx(2 / 3, abs=1e-4)
    assert entry["rate_b"] == 1.0
    assert entry["b_only"] == 1
    assert entry["a_only"] == 0

    # McNemar 精确检验：全部不一致格都在同一侧时，双侧 p 约为 2 * (1/2)^n
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(2 * 0.5 ** 5, abs=1e-9)


def test_engine_single_shot_sampling():
    from quantum_circuit_studio.engine import simulate_circuit

    result = simulate_circuit(2, [{"name": "H", "wire": 0, "col": 0, "param": 0.0}], shots=1)
    assert sum(result.ideal_counts.values()) == 1
    assert result.ideal_actual_shots == 1
def test_cross_model_aggregates_weights_by_reps():
    from experiments.llm_cross_model import aggregate, pair_stats

    rows = [
        {"arm": "resample", "reps": "3", "pass_at_1": "0.5", "final_pass": "0.5",
         "mean_rounds": "2.0", "total_tokens": "100"},
        {"arm": "resample", "reps": "1", "pass_at_1": "1.0", "final_pass": "1.0",
         "mean_rounds": "1.0", "total_tokens": "50"},
    ]
    stats = aggregate(rows)["resample"]
    assert stats["pairs"] == 4
    assert stats["pass_at_1"] == 0.625  # (0.5*3 + 1.0*1) / 4
    assert stats["total_tokens"] == 150

    arms = [{"arm_a": "resample", "arm_b": "feedback", "delta": "0.042",
             "ci_low": "0.0", "ci_high": "0.097", "mcnemar_p": "0.25",
             "b_only": "3", "a_only": "0"}]
    forward = pair_stats(arms, "resample", "feedback")
    assert forward["delta"] == "0.042"
    backward = pair_stats(arms, "feedback", "resample")
    assert backward["arm_a"] == "feedback"
    assert backward["delta"] == "-0.042"
    assert backward["rate_a"] == ""  # 原始行没有 rate 列时不应炸
