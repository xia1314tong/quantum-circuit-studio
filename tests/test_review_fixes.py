# -*- coding: utf-8 -*-
"""2026-11-08 代码审查 8 条问题的回归测试。"""
from __future__ import annotations

import io
import contextlib

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from quantum_circuit_studio.algorithm_lab import (
    deutsch_jozsa_code_body,
    grover_code_body,
)
from quantum_circuit_studio.widgets import CircuitCanvas

import experiments.llm_cross_model as cross
import experiments.llm_loop as loop
import experiments.llm_tasks as tasks
import experiments.mitigation as mitigation


HEADER = "import pennylane as qml\nimport numpy as np\n"


def make_canvas(num_wires: int = 4) -> CircuitCanvas:
    canvas = CircuitCanvas(lambda *a, **k: None, lambda *a, **k: None, lambda *a, **k: None)
    canvas.set_wire_count(num_wires)
    return canvas


def mouse_event(kind, x, y, buttons, button=Qt.MouseButton.LeftButton):
    return QMouseEvent(
        kind,
        QPointF(float(x), float(y)),
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def drag_gate(canvas, gate_index, start, end):
    """模拟一次完整拖动：按下 → 移动 → 松开。"""
    gate = canvas.gates[gate_index]
    canvas.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, *start,
                                      Qt.MouseButton.LeftButton))
    # 按下后确认抓住了正确的一端
    canvas.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, *end,
                                      Qt.MouseButton.LeftButton))
    canvas.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, *end,
                                         Qt.MouseButton.NoButton))
    return gate


def run_body(body: str) -> str:
    """把导出的脚本片段跑起来，返回标准输出。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(HEADER + body, {"__name__": "__review__"})  # noqa: S102 - 测试执行导出代码
    return buffer.getvalue()


# ---------------------------------------------------------------- 1. CNOT 拖动
def test_canvas_really_has_requested_wires():
    canvas = make_canvas(4)
    assert canvas.num_wires == 4


def test_cnot_drag_from_control_keeps_connection():
    canvas = make_canvas(4)
    canvas.gates = [{"name": "CNOT", "control": 0, "target": 3, "col": 2}]
    start = (canvas.col_x(2), canvas.wire_y(0))
    end = (canvas.col_x(3), canvas.wire_y(0))
    gate = drag_gate(canvas, 0, start, end)
    assert (gate["control"], gate["target"]) == (0, 3)
    assert gate["col"] == 3


def test_cnot_drag_from_target_keeps_connection():
    canvas = make_canvas(4)
    canvas.gates = [{"name": "CNOT", "control": 1, "target": 2, "col": 1}]
    start = (canvas.col_x(1), canvas.wire_y(2))
    end = (canvas.col_x(2), canvas.wire_y(2))
    gate = drag_gate(canvas, 0, start, end)
    assert (gate["control"], gate["target"]) == (1, 2), "从目标端横拖不应改变连接"
    assert gate["col"] == 2


def test_cnot_drag_refuses_out_of_range_vertical_move():
    canvas = make_canvas(4)
    canvas.gates = [{"name": "CNOT", "control": 0, "target": 3, "col": 2}]
    start = (canvas.col_x(2), canvas.wire_y(0))
    end = (canvas.col_x(2), canvas.wire_y(1))
    gate = drag_gate(canvas, 0, start, end)
    assert (gate["control"], gate["target"]) == (0, 3)

    neighbour = {"name": "CNOT", "control": 1, "target": 2, "col": 1}
    assert canvas.cnot_drag_endpoints(neighbour, 0) == (0, 1)


# ------------------------------------------------------- 2. Grover 导出代码
def test_grover_export_marks_phase_with_multicontrolled_z():
    output = run_body(grover_code_body(2, "11", 1, 4000))
    frequency = float(output.split("target frequency:")[1].split()[0])
    assert frequency > 0.9, output


# --------------------------------------------------- 3. Deutsch–Jozsa 导出代码
def test_deutsch_jozsa_export_honours_constant_option():
    output = run_body(deutsch_jozsa_code_body(3, balanced=False, shots=400))
    assert "function type: constant" in output
    counts = output.split("measurement counts:")[1].split("\n")[0]
    assert counts.strip() == "Counter({'000': 400})", output


def test_deutsch_jozsa_export_honours_balanced_option():
    output = run_body(deutsch_jozsa_code_body(3, balanced=True, shots=400))
    assert "function type: balanced" in output
    counts = output.split("measurement counts:")[1].split("\n")[0]
    assert "'000'" not in counts, output


# ------------------------------------------------ 4. 断点续跑：'0' 不是真
def test_as_bool_handles_csv_strings():
    assert loop.as_bool("0") is False
    assert loop.as_bool("false") is False
    assert loop.as_bool("") is False
    assert loop.as_bool("1") is True
    assert loop.as_bool(True) is True


def test_summarise_does_not_count_failed_attempt_as_success():
    rows = [
        {"arm": "feedback", "task": "bell", "rep": "0", "round": "1",
         "passed": "0", "layer": "syntax", "prompt_tokens": "10", "completion_tokens": "5"},
        {"arm": "feedback", "task": "bell", "rep": "0", "round": "2",
         "passed": "0", "layer": "syntax", "prompt_tokens": "10", "completion_tokens": "5"},
    ]
    summary = loop.summarise(rows)
    assert summary[0]["final_pass"] == 0.0
    assert summary[0]["pass_at_1"] == 0.0


# ------------------------------------------------------- 5. 布局校验补 CNOT
def ghz_gates(controls: list[int]) -> list[dict]:
    gates = [{"name": "H", "wire": 0, "col": 0}]
    for index, control in enumerate(controls, start=1):
        gates.append({"name": "CNOT", "control": control, "target": index, "col": index})
    return gates


def test_layout_check_accepts_correct_columns():
    assert tasks._layout_ghz_columns(4)(ghz_gates([0, 0, 0])) == []
    chained = tasks._layout_ghz_columns(4, controls=[0, 1, 2])
    assert chained(ghz_gates([0, 1, 2])) == []
    # 控制位换成别的组合就不该通过
    assert chained(ghz_gates([0, 0, 0]))


def test_layout_check_rejects_wrong_control_and_column():
    check = tasks._layout_ghz_columns(4)

    wrong_control = ghz_gates([0, 0, 0])
    wrong_control[1]["control"] = 3
    assert check(wrong_control), "控制位错误的电路不应通过布局校验"

    wrong_column = ghz_gates([0, 0, 0])
    wrong_column[1]["col"] = 2
    assert check(wrong_column), "CNOT 放错列不应通过布局校验"


# --------------------------------------------- 6. 反向比较：CI 与计数要翻转
def test_pair_stats_reversal_flips_interval_and_counts():
    rows = [
        {"arm_a": "resample", "arm_b": "feedback", "pairs": "72",
         "rate_a": "0.9000", "rate_b": "0.9500", "delta": "0.0500",
         "ci_low": "0.0100", "ci_high": "0.1000",
         "b_only": "4", "a_only": "1", "mcnemar_p": "0.2"},
    ]
    flipped = cross.pair_stats(rows, "feedback", "resample")
    assert flipped["delta"] == "-0.0500"
    assert flipped["ci_low"] == "-0.1000"
    assert flipped["ci_high"] == "-0.0100"
    assert flipped["b_only"] == "1"
    assert flipped["a_only"] == "4"


def test_negate_keeps_scientific_notation():
    assert cross._negate("1e-4") == "-1e-4"
    assert cross._negate("-1e-4") == "1e-4"
    assert cross._negate("2.5E-3") == "-2.5E-3"
    assert cross._negate("0.0500") == "-0.0500"


# ------------------------------------------------------ 7. ZNE 不静默截断
def test_zne_refuses_scale_that_exceeds_unit_probability():
    with pytest.raises(ValueError):
        mitigation.zne_support_mass(2, 0.4, shots=1000, seed=1)


def test_zne_uses_actual_noise_strength():
    value = mitigation.zne_support_mass(2, 0.1, shots=20000, seed=7)
    assert 0.0 <= value <= 1.0


# ------------------------------------------------ 8. 图 6 用同一套配对样本
def test_fig6_can_render_from_saved_csv(tmp_path, monkeypatch):
    from experiments import make_figures

    monkeypatch.setattr(make_figures, "FIGURES", tmp_path)
    make_figures.fig6_cross_model()
    assert (tmp_path / "fig6_cross_model.png").stat().st_size > 0
