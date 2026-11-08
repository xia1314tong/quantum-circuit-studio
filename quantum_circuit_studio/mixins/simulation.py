"""PennyLane 量子电路仿真。"""
from ..common import (
    QMessageBox, QDialog, QPlainTextEdit, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QApplication, QLabel, copy, datetime, np, qml, traceback
)
from ..engine import simulate_circuit
from ..thresholds import teaching_threshold


class SimulationMixin:
    def run_simulation(self):

        # 新一轮仿真开始时，先使旧结果失效，避免失败后误导出旧数据
        self.last_results = None
        self.refresh_result_status()

        # 仿真前执行电路合法性检查
        errors = self.collect_validation_errors()

        if errors:
            message = (
                    "电路存在问题，无法运行仿真：\n\n"
                    + "\n".join(
                f"{i}. {msg}"
                for i, msg in enumerate(errors, 1)
            )
            )
            QMessageBox.warning(
                self,
                "仿真前检查失败",
                message
            )
            self.set_status("仿真已取消，请先修正电路")
            return

        n = self.canvas.num_wires
        gates = copy.deepcopy(self.canvas.gates)
        shots = self.shots_spin.value()

        noise_enabled = self.noise_enabled.isChecked()
        noise_model = self.noise_model_combo.currentData()
        noise_probability = float(self.noise_probability.value())

        noise_names = {
            "bitflip": "比特翻转（BitFlip）",
            "phaseflip": "相位翻转（PhaseFlip）",
            "depolarizing": "退极化（Depolarizing）"
        }
        noise_name = noise_names.get(noise_model, str(noise_model))

        try:
            result = simulate_circuit(
                num_wires=n,
                gates=gates,
                shots=shots,
                noise_enabled=noise_enabled,
                noise_model=noise_model,
                noise_probability=noise_probability,
            )

            state = result.state
            labels = result.labels
            ideal_probs = result.ideal_probs
            ideal_counts = result.ideal_counts
            ideal_frequencies = result.ideal_frequencies
            ideal_actual_shots = result.ideal_actual_shots
            noisy_probs = result.noisy_probs
            noisy_counts = result.noisy_counts
            noisy_frequencies = result.noisy_frequencies
            noisy_actual_shots = result.noisy_actual_shots
            depth = result.depth
            ideal_sample_distance = result.ideal_sample_distance
            ideal_entropy = result.ideal_entropy

            analysis_lines = [
                f"量子比特数：{n}",
                f"量子门数量：{len(gates)}",
                f"电路深度（按列计算）：{depth}",
                f"理论概率总和：{np.sum(ideal_probs):.6f}",
                f"理想理论分布熵：{ideal_entropy:.4f} bit",
                f"理想理论与采样分布的总变差距离：{ideal_sample_distance:.6f}",
                "说明：总变差距离越接近 0，两个概率分布越接近。"
            ]

            ideal_noisy_distance = result.ideal_noisy_distance
            noisy_sample_distance = result.noisy_sample_distance
            noisy_entropy = result.noisy_entropy

            if noise_enabled:
                analysis_lines.extend([
                    f"噪声模型：{noise_name}",
                    f"噪声概率：{noise_probability:.3f}",
                    f"理想分布与含噪声理论分布的总变差距离：{ideal_noisy_distance:.6f}",
                    f"含噪声理论分布熵：{noisy_entropy:.4f} bit",
                    f"含噪声理论与采样分布的总变差距离：{noisy_sample_distance:.6f}"
                ])
                p_star = teaching_threshold(n)
                verdict = "已超出" if noise_probability > p_star else "仍在"
                analysis_lines.append(
                    f"GHZ 教学容限参考：n={n} 时 p*≈{p_star:.4f}，当前 p={noise_probability:.3f}"
                    f"（{verdict} 10% 偏差边界；该阈值只对 GHZ 族成立）"
                )
            else:
                analysis_lines.append("噪声模拟：未启用")

            # 5. 整理结果文本
            probability_rows = [
                f"|{label}⟩：{prob:.6f}（{prob * 100:.2f}%）"
                for label, prob in zip(labels, ideal_probs)
                if prob > 1e-8
            ]

            ideal_sample_rows = [
                f"|{label}⟩：{ideal_counts[label]} 次"
                f"（频率 {ideal_counts[label] / ideal_actual_shots * 100:.2f}%）"
                for label in labels
                if ideal_counts[label] > 0
            ]

            vector_rows = [
                f"|{label}⟩：{amplitude.real:+.6f}{amplitude.imag:+.6f}i"
                f"；理论概率 {abs(amplitude) ** 2:.6f}"
                for label, amplitude in zip(labels, state)
                if abs(amplitude) > 1e-8
            ]

            result_parts = [
                f"测量次数：{shots}",
                "【理想电路理论概率】",
                "\n".join(probability_rows) or "无非零概率状态",
                "【理想电路有限次测量】",
                "\n".join(ideal_sample_rows) or "无测量结果",
                "【理想电路量子态向量】",
                "\n".join(vector_rows) or "无非零振幅",
            ]

            if noise_enabled:
                noisy_probability_rows = [
                    f"|{label}⟩：{prob:.6f}（{prob * 100:.2f}%）"
                    for label, prob in zip(labels, noisy_probs)
                    if prob > 1e-8
                ]
                noisy_sample_rows = [
                    f"|{label}⟩：{noisy_counts[label]} 次"
                    f"（频率 {noisy_counts[label] / noisy_actual_shots * 100:.2f}%）"
                    for label in labels
                    if noisy_counts[label] > 0
                ]
                result_parts.extend([
                    f"【含噪声电路】{noise_name}，概率={noise_probability:.3f}",
                    "【含噪声理论概率】",
                    "\n".join(noisy_probability_rows) or "无非零概率状态",
                    "【含噪声有限次测量】",
                    "\n".join(noisy_sample_rows) or "无测量结果"
                ])

            # 结果明细与实验分析分开显示，实验分析有独立面板。
            self.result_label.setText("\n\n".join(result_parts))
            self.analysis_label.setText("\n".join(analysis_lines))

            # 6. 绘制理想与含噪声结果
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            x = np.arange(len(labels))

            if noise_enabled:
                series = [
                    ("理想理论概率", ideal_probs, "#3B82F6"),
                    ("理想采样频率", ideal_frequencies, "#93C5FD"),
                    ("含噪声理论概率", noisy_probs, "#F59E0B"),
                    ("含噪声采样频率", noisy_frequencies, "#FCD34D"),
                ]
            else:
                series = [
                    ("理论概率", ideal_probs, "#3B82F6"),
                    ("采样频率", ideal_frequencies, "#F59E0B"),
                ]

            width = min(0.8 / len(series), 0.32)
            offsets = (
                              np.arange(len(series)) - (len(series) - 1) / 2
                      ) * width

            for offset, (label, values, color) in zip(offsets, series):
                ax.bar(
                    x + offset,
                    values,
                    width,
                    label=label,
                    color=color
                )

            tick_step = max(1, int(np.ceil(len(labels) / 16)))
            ax.set_xticks(x[::tick_step])
            ax.set_xticklabels(labels[::tick_step])
            ax.set_xlabel("计算基态")
            ax.set_ylabel("概率 / 频率")
            ax.set_ylim(0, 1.05)

            title = (
                f"理想与含噪声分布对比（{noise_name}，p={noise_probability:.3f}）"
                if noise_enabled
                else "理想理论概率与测量采样频率"
            )
            ax.set_title(title, fontsize=10, pad=10)
            ax.legend(fontsize=7, frameon=False, loc="upper right")
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            # 刻度较少时横排，避免倾斜标签在窄画布中被裁掉
            rotate = 45 if len(labels) > 8 else 0
            ax.tick_params(axis="x", labelrotation=rotate, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.margins(x=0.02)
            self.figure.tight_layout()
            self.chart.draw()

            # 7. 保存本次仿真数据，供后续导出
            self.last_results = {
                "num_wires": n,
                "gates": copy.deepcopy(gates),
                "shots": shots,
                "labels": labels,
                "state": state.copy(),
                "ideal_probs": ideal_probs.copy(),
                "ideal_counts": dict(ideal_counts),
                "ideal_frequencies": ideal_frequencies.copy(),
                "noise_enabled": noise_enabled,
                "noise_model": noise_model,
                "noise_name": noise_name,
                "noise_probability": noise_probability,
                "noisy_probs": None if noisy_probs is None else noisy_probs.copy(),
                "noisy_counts": None if noisy_counts is None else dict(noisy_counts),
                "noisy_frequencies": (
                    None if noisy_frequencies is None
                    else noisy_frequencies.copy()
                ),
                "analysis_lines": list(analysis_lines),
                "ideal_sample_distance": ideal_sample_distance,
                "ideal_noisy_distance": ideal_noisy_distance,
                "noisy_sample_distance": noisy_sample_distance,
                "ideal_entropy": ideal_entropy,
                "noisy_entropy": noisy_entropy,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self.set_status(
                f"仿真完成：{shots} 次测量"
                + ("，已完成含噪声对比" if noise_enabled else "")
            )
            # 仿真结束后，自动将下方独立滚动区域定位到结果图附近。
            if hasattr(self, "bottom_scroll") and hasattr(self, "result_box"):
                self.bottom_scroll.ensureWidgetVisible(self.result_box, 0, 0)

        except Exception as error:
            error_text = traceback.format_exc()
            print(error_text)
            QMessageBox.critical(
                self,
                "仿真失败",
                error_text
            )
            self.set_status("仿真失败")

    def generate_pennylane_code(self):
        """根据当前画布与仿真参数生成可独立运行的 PennyLane Python 脚本。"""
        n = int(self.canvas.num_wires)
        shots = int(self.shots_spin.value())
        gates = sorted(copy.deepcopy(self.canvas.gates), key=lambda g: g.get("col", 0))
        noise_enabled = self.noise_enabled.isChecked()
        noise_model = self.noise_model_combo.currentData()
        p = float(self.noise_probability.value())
        lines = [
            '"""由 Quantum Circuit Studio 自动生成。"""',
            'import pennylane as qml',
            'from pennylane import numpy as np',
            '', f'n_qubits = {n}', f'shots = {shots}',
            f'noise_enabled = {noise_enabled}', f'noise_probability = {p!r}',
            f'noise_model = {noise_model!r}', '',
        ]
        # default.qubit 不支持信道，含噪声电路必须切换到 default.mixed
        noise_active = bool(noise_enabled and p > 0)
        device_name = "default.mixed" if noise_active else "default.qubit"
        lines += [
            f'dev = qml.device("{device_name}", wires=n_qubits, shots=shots)', '',
            '@qml.qnode(dev)', 'def circuit():',
            '    # 按照画布中的量子门顺序执行'
        ]
        if not gates:
            lines.append('    # 当前电路为空，直接测量初始态')
        for g in gates:
            name = g.get('name')
            wire = int(g.get('wire', 0))
            if name in ('H','X','Y','Z'):
                op = {'H':'Hadamard','X':'PauliX','Y':'PauliY','Z':'PauliZ'}[name]
                lines.append(f'    qml.{op}(wires={wire})')
                wires = [wire]
            elif name in ('RX','RY','RZ'):
                angle = float(g.get('param', np.pi/2))
                lines.append(f'    qml.{name}({angle!r}, wires={wire})')
                wires = [wire]
            elif name == 'CNOT':
                control, target = int(g.get('control', 0)), int(g.get('target', 1))
                lines.append(f'    qml.CNOT(wires=[{control}, {target}])')
                wires = [control, target]
            else:
                continue
            if noise_active:
                noise_op = {'bitflip':'BitFlip', 'phaseflip':'PhaseFlip', 'depolarizing':'DepolarizingChannel'}.get(noise_model)
                if noise_op:
                    for w in wires:
                        lines.append(f'    qml.{noise_op}(noise_probability, wires={w})')
        lines += [
            '    return qml.probs(wires=range(n_qubits))', '',
            'if __name__ == "__main__":',
            '    probabilities = circuit()',
            '    print("测量概率：")',
            '    for index, probability in enumerate(probabilities):',
            '        bitstring = format(index, f"0{n_qubits}b")',
            '        print(f"|{bitstring}>: {probability:.6f}")', ''
        ]
        return '\n'.join(lines)

    def show_generated_code(self):
        """弹出代码查看器，支持复制和另存为 .py。"""
        code = self.generate_pennylane_code()
        dialog = QDialog(self)
        dialog.setWindowTitle("当前量子电路的 PennyLane 代码")
        dialog.resize(760, 620)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("以下代码根据当前画布和参数生成，可复制或保存为 .py 文件。"))
        editor = QPlainTextEdit()
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setPlainText(code)
        layout.addWidget(editor, 1)
        buttons = QHBoxLayout()
        copy_button = QPushButton("复制代码")
        save_button = QPushButton("保存为 .py")
        close_button = QPushButton("关闭")
        buttons.addWidget(copy_button)
        buttons.addWidget(save_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        copy_button.clicked.connect(lambda: (QApplication.clipboard().setText(editor.toPlainText()), self.set_status("Python 代码已复制到剪贴板")))
        def save_code():
            path, _ = QFileDialog.getSaveFileName(dialog, "保存 Python 代码", "quantum_circuit.py", "Python 文件 (*.py)")
            if path:
                if not path.lower().endswith('.py'):
                    path += '.py'
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(editor.toPlainText())
                    self.set_status(f"代码已保存：{path}")
                except OSError as exc:
                    QMessageBox.critical(dialog, "保存失败", str(exc))
        save_button.clicked.connect(save_code)
        close_button.clicked.connect(dialog.accept)
        dialog.exec()

