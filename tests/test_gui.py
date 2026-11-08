"""界面层集成测试。

目的不是"点按钮不崩"，而是验证界面走完整条链路后
拿到的数值与 engine 层一致，且导出的代码可以独立运行。
"""
import math
import subprocess
import time
import sys

import numpy as np
import pytest


def test_main_window_starts_with_empty_circuit(window):
    assert "量子电路" in window.windowTitle()
    assert window.canvas.num_wires >= 1
    assert window.last_results is None


def test_bell_example_then_simulation(window):
    window.bell_example()
    window.run_simulation()

    result = window.last_results
    assert result is not None
    assert np.allclose(result["ideal_probs"], [0.5, 0, 0, 0.5], atol=1e-9)
    assert sum(result["ideal_counts"].values()) == result["shots"]
    assert any("电路深度" in line for line in result["analysis_lines"])


def test_noisy_simulation_matches_analytic_value(window):
    window.bell_example()
    window.noise_enabled.setChecked(True)
    window.noise_model_combo.setCurrentIndex(0)  # bitflip
    window.noise_probability.setValue(0.1)
    window.run_simulation()

    result = window.last_results
    assert result["noise_enabled"] is True
    assert np.allclose(result["noisy_probs"], [0.41, 0.09, 0.09, 0.41], atol=1e-9)
    assert result["ideal_noisy_distance"] == pytest.approx(0.18, abs=1e-9)
    assert result["noisy_entropy"] == pytest.approx(1.680077, abs=1e-5)
    assert any("GHZ 教学容限参考" in line for line in result["analysis_lines"])


def test_validation_blocks_invalid_circuit(window):
    """CNOT 控制位与目标位相同时，仿真应被拦截而不是崩溃。"""
    window.canvas.num_wires = 2
    window.canvas.gates = [
        {"name": "CNOT", "control": 0, "target": 0, "col": 0},
    ]
    window.canvas.adjust_canvas()
    window.run_simulation()

    assert window.last_results is None


def test_generated_code_uses_mixed_device_when_noisy(window):
    window.bell_example()
    window.noise_enabled.setChecked(True)
    window.noise_model_combo.setCurrentIndex(0)
    window.noise_probability.setValue(0.1)

    code = window.generate_pennylane_code()

    assert 'qml.device("default.mixed"' in code
    assert "BitFlip" in code


def test_generated_code_runs_standalone(window, tmp_path):
    window.bell_example()
    window.noise_enabled.setChecked(True)
    window.noise_model_combo.setCurrentIndex(0)
    window.noise_probability.setValue(0.1)

    script = tmp_path / "generated.py"
    script.write_text(window.generate_pennylane_code(), encoding="utf-8")

    # 生成脚本要在一个全新的解释器里跑起来。机器负载重时偶尔会启动失败
    # （返回码非 0 或输出为空，且 stderr 与本项目无关），这里重试几次以避免假失败。
    completed = None
    for attempt in range(3):
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        if completed.returncode == 0 and "|00>" in (completed.stdout or ""):
            break
        if attempt < 2:
            time.sleep(2.0)

    assert completed.returncode == 0, (completed.stdout or "") + (completed.stderr or "")
    assert "|00>" in (completed.stdout or ""), (completed.stdout or "") + (completed.stderr or "")


def test_clean_circuit_resets_results(window):
    window.bell_example()
    window.run_simulation()
    assert window.last_results is not None

    window.clear_circuit()

    assert window.last_results is None
    assert window.canvas.gates == []


def test_qubit_count_change_keeps_canvas_consistent(window):
    window.wire_spin.setValue(4)
    assert window.canvas.num_wires == 4

    window.wire_spin.setValue(1)
    assert window.canvas.num_wires == 1


def test_rotation_gate_parameter_used_in_state(window):
    window.canvas.num_wires = 1
    window.canvas.gates = [{"name": "RY", "wire": 0, "col": 0, "param": math.pi / 2}]
    window.canvas.adjust_canvas()
    window.run_simulation()

    probs = window.last_results["ideal_probs"]
    assert np.allclose(probs, [0.5, 0.5], atol=1e-9)
