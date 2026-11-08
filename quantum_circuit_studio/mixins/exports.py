"""实验配置、线路与结果导入导出。"""
from ..common import (
    GATE_NAMES, Path, QFileDialog, QMessageBox, ROTATION_GATES, copy,
    csv, datetime, json, np
)


class ExportMixin:
    def get_exportable_results(self):
        """导出前确认结果与当前电路、测量和噪声设置一致。"""
        data = self.last_results
        if data is None:
            QMessageBox.information(
                self,
                "没有仿真结果",
                "请先点击“运行仿真”，生成结果后再导出。"
            )
            return None

        is_stale = self.result_is_stale()
        self.refresh_result_status()

        if is_stale:
            QMessageBox.warning(
                self,
                "结果已过期",
                "电路或仿真参数在上次运行后发生了变化。\n"
                "请先重新运行仿真，再导出结果。"
            )
            return None

        return data

    def export_csv(self):
        """导出每个计算基态的理论概率和测量统计。"""
        data = self.get_exportable_results()
        if data is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出仿真数据",
            "quantum_simulation_results.csv",
            "CSV 文件 (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            fields = [
                "计算基态",
                "理想理论概率",
                "理想测量次数",
                "理想采样频率",
                "含噪声理论概率",
                "含噪声测量次数",
                "含噪声采样频率"
            ]

            with open(path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()

                for index, label in enumerate(data["labels"]):
                    noisy_probs = data["noisy_probs"]
                    noisy_counts = data["noisy_counts"]
                    noisy_frequencies = data["noisy_frequencies"]

                    writer.writerow({
                        "计算基态": f"|{label}⟩",
                        "理想理论概率": f'{data["ideal_probs"][index]:.10f}',
                        "理想测量次数": data["ideal_counts"][label],
                        "理想采样频率": f'{data["ideal_frequencies"][index]:.10f}',
                        "含噪声理论概率": (
                            "" if noisy_probs is None
                            else f"{noisy_probs[index]:.10f}"
                        ),
                        "含噪声测量次数": (
                            "" if noisy_counts is None
                            else noisy_counts[label]
                        ),
                        "含噪声采样频率": (
                            "" if noisy_frequencies is None
                            else f"{noisy_frequencies[index]:.10f}"
                        )
                    })

            self.set_status(f"CSV 已导出：{path}")
            QMessageBox.information(self, "导出成功", f"数据已保存到：\n{path}")

        except (OSError, csv.Error) as error:
            QMessageBox.critical(self, "CSV 导出失败", str(error))

    def export_chart(self):
        """将当前概率对比图保存为高清 PNG。"""
        data = self.get_exportable_results()
        if data is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存概率图",
            "quantum_probability_comparison.png",
            "PNG 图片 (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            self.figure.savefig(
                path,
                format="png",
                dpi=220,
                bbox_inches="tight"
            )
            self.set_status(f"概率图已保存：{path}")
            QMessageBox.information(self, "导出成功", f"概率图已保存到：\n{path}")

        except OSError as error:
            QMessageBox.critical(self, "图片导出失败", str(error))

    def export_report(self):
        """导出包含线路信息、设置、概率和分析指标的 Markdown 实验报告。"""
        data = self.get_exportable_results()
        if data is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出实验报告",
            "quantum_simulation_report.md",
            "Markdown 报告 (*.md);;文本文件 (*.txt)"
        )
        if not path:
            return
        if Path(path).suffix.lower() not in (".md", ".txt"):
            path += ".md"

        try:
            gate_lines = []
            for index, gate in enumerate(data["gates"], 1):
                name = gate["name"]
                if name == "CNOT":
                    description = (
                        f"CNOT：控制 q{gate['control']}，目标 q{gate['target']}，"
                        f"第 {gate['col']} 列"
                    )
                else:
                    description = (
                        f"{name}：作用于 q{gate['wire']}，第 {gate['col']} 列"
                    )
                    if name in ROTATION_GATES:
                        angle_deg = float(np.degrees(gate.get("param", np.pi / 2)))
                        description += f"，角度 {angle_deg:.2f}°"
                gate_lines.append(f"{index}. {description}")

            if data["noise_enabled"]:
                noise_summary = (
                    f"已启用，模型为 {data['noise_name']}，"
                    f"噪声概率 p={data['noise_probability']:.3f}"
                )
            else:
                noise_summary = "未启用"

            content = [
                "# 量子电路仿真实验报告",
                "",
                f"- 生成时间：{data['timestamp']}",
                f"- 量子比特数：{data['num_wires']}",
                f"- 量子门数量：{len(data['gates'])}",
                f"- 电路深度（按列计算）：{max((int(g.get('col', 0)) for g in data['gates']), default=-1) + 1}",
                f"- 测量次数：{data['shots']}",
                f"- 噪声设置：{noise_summary}",
                "",
                "## 一、量子门序列",
                "",
                *(gate_lines if gate_lines else ["电路为空（未添加量子门）。"]),
                "",
                "## 二、理想电路测量结果",
                "",
                "| 计算基态 | 理论概率 | 测量次数 | 采样频率 |",
                "|---|---:|---:|---:|"
            ]

            for index, label in enumerate(data["labels"]):
                content.append(
                    f"| `|{label}⟩` | {data['ideal_probs'][index]:.8f} "
                    f"| {data['ideal_counts'][label]} "
                    f"| {data['ideal_frequencies'][index]:.8f} |"
                )

            if data["noise_enabled"]:
                content.extend([
                    "",
                    "## 三、含噪声电路测量结果",
                    "",
                    "| 计算基态 | 含噪声理论概率 | 测量次数 | 采样频率 |",
                    "|---|---:|---:|---:|"
                ])
                for index, label in enumerate(data["labels"]):
                    content.append(
                        f"| `|{label}⟩` | {data['noisy_probs'][index]:.8f} "
                        f"| {data['noisy_counts'][label]} "
                        f"| {data['noisy_frequencies'][index]:.8f} |"
                    )

            content.extend([
                "",
                "## 四、实验分析",
                ""
            ])
            content.extend(
                f"- {line}" for line in data["analysis_lines"]
            )

            content.extend([
                "",
                "## 五、量子态向量（理想电路）",
                "",
                "| 计算基态 | 复振幅 | 理论概率 |",
                "|---|---:|---:|"
            ])
            for label, amplitude in zip(data["labels"], data["state"]):
                if abs(amplitude) > 1e-8:
                    complex_text = (
                        f"{amplitude.real:+.8f}"
                        f"{amplitude.imag:+.8f}i"
                    )
                    content.append(
                        f"| `|{label}⟩` | {complex_text} "
                        f"| {abs(amplitude) ** 2:.8f} |"
                    )

            with open(path, "w", encoding="utf-8") as file:
                file.write("\n".join(content) + "\n")

            self.set_status(f"实验报告已导出：{path}")
            QMessageBox.information(self, "导出成功", f"实验报告已保存到：\n{path}")

        except OSError as error:
            QMessageBox.critical(self, "报告导出失败", str(error))

    def save_experiment_config(self):
        """保存电路、测量次数和噪声参数，不包含上次仿真结果。"""
        validation_errors = self.collect_validation_errors()
        if validation_errors:
            QMessageBox.warning(
                self,
                "无法保存实验配置",
                "当前电路尚未通过合法性检查，请先修正以下问题：\n\n"
                + "\n".join(
                    f"{index}. {message}"
                    for index, message in enumerate(validation_errors, 1)
                )
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存完整实验配置",
            "quantum_experiment_config.json",
            "JSON 配置文件 (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        snapshot = self.experiment_settings_snapshot()
        payload = {
            "application": "Quantum Circuit Studio",
            "file_type": "quantum_circuit_studio_experiment_config",
            "version": 1,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": snapshot
        }

        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            self.set_status(f"完整实验配置已保存：{path}")
            QMessageBox.information(
                self,
                "配置保存成功",
                "已保存量子比特数、量子门序列、测量次数及噪声设置。\n\n"
                f"文件位置：\n{path}"
            )
        except OSError as error:
            QMessageBox.critical(self, "配置保存失败", str(error))

    def load_experiment_config(self):
        """加载完整实验配置；验证完成后再统一应用，避免半加载状态。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "加载完整实验配置",
            "",
            "JSON 配置文件 (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            if not isinstance(payload, dict):
                raise ValueError("配置文件格式无效：根节点必须是 JSON 对象。")
            if payload.get("file_type") != "quantum_circuit_studio_experiment_config":
                raise ValueError(
                    "这不是完整实验配置文件。\n"
                    "请使用“加载实验配置”打开由“保存实验配置”生成的 JSON；"
                    "普通电路文件请使用“加载电路”。"
                )
            if payload.get("version") != 1:
                raise ValueError("不支持的实验配置版本。")

            config = payload.get("configuration", {})
            if not isinstance(config, dict):
                raise ValueError("配置内容格式无效。")

            # 验证量子比特数与测量次数
            count = int(config.get("num_wires", 2))
            if not 1 <= count <= 8:
                raise ValueError("量子比特数必须在 1 到 8 之间。")

            shots = int(config.get("shots", 1000))
            if not 100 <= shots <= 100000:
                raise ValueError("测量次数必须在 100 到 100000 之间。")

            # 验证噪声配置
            noise_enabled = bool(config.get("noise_enabled", False))
            noise_model = config.get("noise_model", "bitflip")
            valid_noise_models = {"bitflip", "phaseflip", "depolarizing"}
            if noise_model not in valid_noise_models:
                raise ValueError(f"不支持的噪声模型：{noise_model}")

            noise_probability = float(config.get("noise_probability", 0.05))
            if not np.isfinite(noise_probability) or not 0.0 <= noise_probability <= 1.0:
                raise ValueError("噪声概率必须是 0 到 1 之间的有限数值。")

            # 验证量子门；只有全部检查通过后才修改当前程序状态。
            raw_gates = config.get("gates", [])
            if not isinstance(raw_gates, list):
                raise ValueError("量子门序列必须是列表。")

            gates = []
            occupied = {}
            for index, raw in enumerate(raw_gates, start=1):
                if not isinstance(raw, dict):
                    raise ValueError(f"第 {index} 个量子门的配置格式无效。")

                name = raw.get("name")
                if name not in GATE_NAMES:
                    raise ValueError(f"第 {index} 个量子门不受支持：{name}")

                col = int(raw.get("col", 0))
                if col < 0:
                    raise ValueError(f"第 {index} 个量子门的列编号不能为负数。")

                if name == "CNOT":
                    if count < 2:
                        raise ValueError("CNOT 门至少需要两个量子比特。")
                    control = int(raw.get("control", 0))
                    target = int(raw.get("target", 1))
                    if not (0 <= control < count and 0 <= target < count):
                        raise ValueError(f"第 {index} 个 CNOT 门的线路编号超出范围。")
                    if control == target:
                        raise ValueError(f"第 {index} 个 CNOT 门的控制端和目标端不能相同。")
                    gate = {
                        "name": "CNOT",
                        "control": control,
                        "target": target,
                        "col": col
                    }
                    used_wires = {control, target}
                else:
                    wire = int(raw.get("wire", 0))
                    if not 0 <= wire < count:
                        raise ValueError(f"第 {index} 个 {name} 门的线路编号超出范围。")
                    param = float(raw.get("param", np.pi / 2))
                    if not np.isfinite(param):
                        raise ValueError(f"第 {index} 个 {name} 门的参数必须是有限数值。")
                    gate = {
                        "name": name,
                        "wire": wire,
                        "col": col,
                        "param": param
                    }
                    used_wires = {wire}

                # 同一列不能有两个门占用同一量子比特
                for wire in used_wires:
                    key = (col, wire)
                    if key in occupied:
                        raise ValueError(
                            f"第 {index} 个门与第 {occupied[key]} 个门冲突："
                            f"第 {col} 列的 q{wire} 被重复占用。"
                        )
                    occupied[key] = index

                gates.append(gate)

            # 应用配置时阻止信号触发中间状态更新。
            self.wire_spin.blockSignals(True)
            self.shots_spin.blockSignals(True)
            self.noise_enabled.blockSignals(True)
            self.noise_model_combo.blockSignals(True)
            self.noise_probability.blockSignals(True)
            try:
                self.wire_spin.setValue(count)
                self.shots_spin.setValue(shots)
                self.noise_enabled.setChecked(noise_enabled)
                model_index = self.noise_model_combo.findData(noise_model)
                if model_index < 0:
                    raise ValueError(f"找不到噪声模型：{noise_model}")
                self.noise_model_combo.setCurrentIndex(model_index)
                self.noise_probability.setValue(noise_probability)
                self.canvas.num_wires = count
                self.canvas.gates = copy.deepcopy(gates)
            finally:
                self.wire_spin.blockSignals(False)
                self.shots_spin.blockSignals(False)
                self.noise_enabled.blockSignals(False)
                self.noise_model_combo.blockSignals(False)
                self.noise_probability.blockSignals(False)

            self.update_noise_controls()
            self.canvas.select_gate(None)
            self.canvas.adjust_canvas()
            self.canvas.update()

            # 完整配置已更换，清理旧历史和旧结果，防止旧结果被误导出。
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.update_history_buttons()
            self.last_results = None
            self.result_label.setText("实验配置已加载，请运行仿真生成新结果。")
            self.analysis_label.setText(
                "配置已加载。运行仿真后，这里会显示电路深度、熵和分布差异等分析指标。"
            )
            self.figure.clear()
            self.chart.draw()
            self.refresh_result_status()

            self.set_status(f"实验配置加载成功：{path}。请重新运行仿真。")
            QMessageBox.information(
                self,
                "配置加载成功",
                "已恢复量子线路、量子比特数、测量次数和噪声设置。\n"
                "由于配置文件保存的是实验设置而不是仿真结果，请重新运行仿真。"
            )

        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "配置加载失败", str(error))

    def save_circuit(self):

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存电路",
            "quantum_circuit.json",
            "JSON 文件 (*.json)"
        )

        if not path:
            return

        data = self.make_snapshot()

        try:

            with open(path, "w", encoding="utf-8") as file:

                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2
                )

            self.set_status("电路保存成功")

        except Exception as error:

            QMessageBox.critical(
                self,
                "保存失败",
                str(error)
            )

    def load_circuit(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "加载电路",
            "",
            "JSON 文件 (*.json)"
        )

        if not path:
            return

        try:

            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            count = max(
                1,
                min(
                    8,
                    int(data.get("num_wires", 2))
                )
            )

            valid = []

            for gate in data.get("gates", []):

                name = gate.get("name")

                if name not in GATE_NAMES:
                    continue

                col = max(
                    0,
                    int(gate.get("col", 0))
                )

                if name == "CNOT":

                    control = int(
                        gate.get("control", 0)
                    )

                    target = int(
                        gate.get("target", 1)
                    )

                    if (
                            count >= 2
                            and 0 <= control < count
                            and 0 <= target < count
                            and control != target
                    ):
                        valid.append({
                            "name": name,
                            "control": control,
                            "target": target,
                            "col": col
                        })

                else:

                    wire = int(
                        gate.get("wire", 0)
                    )

                    if 0 <= wire < count:
                        param = float(
                            gate.get("param", np.pi / 2)
                        )

                        valid.append({
                            "name": name,
                            "wire": wire,
                            "col": col,
                            "param": param
                        })

            # 加载成功前保存当前状态，方便撤销
            new_state = {
                "num_wires": count,
                "gates": valid
            }

            self.record_history()

            self.wire_spin.blockSignals(True)
            self.wire_spin.setValue(count)
            self.wire_spin.blockSignals(False)

            self.canvas.num_wires = count
            self.canvas.gates = copy.deepcopy(valid)

            self.canvas.select_gate(None)
            self.canvas.adjust_canvas()
            self.canvas.update()

            self.set_status("电路加载成功")

        except Exception as error:

            QMessageBox.critical(
                self,
                "加载失败",
                str(error)
            )
