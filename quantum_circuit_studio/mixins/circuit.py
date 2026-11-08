"""电路校验与线路操作。"""
from ..common import (
    GATE_NAMES, QMessageBox, ROTATION_GATES, copy, np
)


class CircuitMixin:
    def collect_validation_errors(self):

        errors = []

        count = self.canvas.num_wires
        gates = self.canvas.gates

        if not 1 <= count <= 8:
            errors.append("量子比特数量必须在 1 到 8 之间。")

        # 每列每个量子比特最多由一个门占用
        occupied = {}

        for index, gate in enumerate(gates, start=1):

            name = gate.get("name")

            if name not in GATE_NAMES:
                errors.append(
                    f"第 {index} 个门：不支持的量子门 {name}。"
                )
                continue

            col = gate.get("col", 0)

            if (
                    not isinstance(col, (int, np.integer))
                    or col < 0
            ):
                errors.append(
                    f"第 {index} 个门：列编号必须是非负整数。"
                )
                continue

            if name == "CNOT":

                if count < 2:
                    errors.append(
                        f"第 {index} 个 CNOT 门："
                        "量子比特数量不足。"
                    )
                    continue

                control = gate.get("control")
                target = gate.get("target")

                if not isinstance(control, (int, np.integer)):
                    errors.append(
                        f"第 {index} 个 CNOT 门：控制端编号无效。"
                    )
                    continue

                if not isinstance(target, (int, np.integer)):
                    errors.append(
                        f"第 {index} 个 CNOT 门：目标端编号无效。"
                    )
                    continue

                if (
                        not 0 <= control < count
                        or not 0 <= target < count
                ):
                    errors.append(
                        f"第 {index} 个 CNOT 门："
                        "控制端或目标端超出量子比特范围。"
                    )
                    continue

                if control == target:
                    errors.append(
                        f"第 {index} 个 CNOT 门："
                        "控制端不能与目标端相同。"
                    )
                    continue

                wires = {control, target}

            else:

                wire = gate.get("wire")

                if not isinstance(wire, (int, np.integer)):
                    errors.append(
                        f"第 {index} 个 {name} 门："
                        "量子比特编号无效。"
                    )
                    continue

                if not 0 <= wire < count:
                    errors.append(
                        f"第 {index} 个 {name} 门："
                        "量子比特编号超出范围。"
                    )
                    continue

                wires = {wire}

            # 检查旋转参数
            if name in ROTATION_GATES:

                param = gate.get("param", np.pi / 2)

                try:
                    param = float(param)

                    if not np.isfinite(param):
                        raise ValueError

                except (TypeError, ValueError):
                    errors.append(
                        f"第 {index} 个 {name} 门："
                        "旋转角度必须是有限数值。"
                    )
                    continue

            # 检查同一列是否有门占用相同量子比特
            for wire in wires:

                key = (int(col), int(wire))

                if key in occupied:

                    other_index = occupied[key]

                    errors.append(
                        f"第 {index} 个 {name} 门与"
                        f"第 {other_index} 个门冲突："
                        f"第 {col} 列的 q{wire} 被重复占用。"
                    )

                else:

                    occupied[key] = index

        return errors

    def check_circuit(self):

        errors = self.collect_validation_errors()

        if errors:
            message = (
                    "发现以下问题，请修改后再运行：\n\n"
                    + "\n".join(
                f"{i}. {msg}"
                for i, msg in enumerate(errors, 1)
            )
            )

            QMessageBox.warning(
                self,
                "电路检查未通过",
                message
            )

            self.set_status(
                f"电路检查未通过，共发现 {len(errors)} 个问题"
            )

            return False

        QMessageBox.information(
            self,
            "电路检查",
            "电路检查通过！\n\n"
            f"量子比特数：{self.canvas.num_wires}\n"
            f"量子门数量：{len(self.canvas.gates)}\n\n"
            "当前电路可以提交仿真。"
        )

        self.set_status("电路检查通过")

        return True

    def clear_circuit(self):

        if not self.canvas.gates:
            self.set_status("当前线路已经为空")
            return

        self.record_history()

        self.canvas.gates = []
        self.canvas.select_gate(None)

        self.canvas.adjust_canvas()
        self.canvas.update()

        self.result_label.setText("线路已清空。")
        self.analysis_label.setText("运行仿真后，这里会显示电路深度、熵和分布差异等分析指标。")
        self.last_results = None

        self.figure.clear()
        self.chart.draw()

        self.set_status("线路已清空")

    def bell_example(self):

        target_gates = [
            {
                "name": "H",
                "wire": 0,
                "col": 0,
                "param": float(np.pi / 2)
            },
            {
                "name": "CNOT",
                "control": 0,
                "target": 1,
                "col": 1
            }
        ]

        # 整个示例作为一次操作
        current = self.make_snapshot()

        if (
                current["num_wires"] == 2
                and current["gates"] == target_gates
        ):
            self.set_status("当前已经是 Bell 态示例")
            return

        self.record_history(current)

        self.wire_spin.blockSignals(True)
        self.wire_spin.setValue(2)
        self.wire_spin.blockSignals(False)

        self.canvas.num_wires = 2
        self.canvas.gates = copy.deepcopy(target_gates)

        self.canvas.select_gate(None)
        self.canvas.adjust_canvas()
        self.canvas.update()

        self.set_status("已载入 Bell 态示例")
