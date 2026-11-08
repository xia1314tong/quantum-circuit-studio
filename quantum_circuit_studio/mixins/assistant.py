"""AI 量子助手与 DeepSeek API。"""
import sys

from ..common import (
    GATE_NAMES, Path, QApplication, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QMessageBox, QPlainTextEdit, QPushButton, QTimer, QVBoxLayout,
    ROTATION_GATES, copy, json, np, traceback
)
from ..widgets import DeepSeekResultReceiver, DeepSeekWorker, SafeAssistantDialog
from ..quantum_demo import QuantumDemoDialog


class AssistantMixin:
    def _save_deepseek_config(self):
        """保存当前模型选择及本机密钥配置；密钥不显示在界面中。"""
        try:
            self.deepseek_config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.deepseek_config_path.open("w", encoding="utf-8") as config_file:
                json.dump(
                    {"api_key": self.deepseek_api_key, "model": self.deepseek_model},
                    config_file, ensure_ascii=False, indent=2,
                )
            return True
        except Exception as exc:
            print(f"保存 DeepSeek 配置失败：{exc}")
            return False

    def _call_deepseek(self, system_prompt, user_prompt, json_mode=False, max_tokens=5000, thinking=None):
        """调用 DeepSeek OpenAI 兼容接口。API Key 不写入本地文件。

        thinking=None 时沿用 API 默认设置；thinking=False 时关闭思考模式，
        避免解释类请求把 max_tokens 消耗在内部推理阶段而没有最终回答。
        """
        if not self.deepseek_api_key:
            raise ValueError(
                "尚未配置 DeepSeek API Key。请在程序同目录的 deepseek_config.json 中填写 api_key，或设置环境变量 DEEPSEEK_API_KEY。"
            )
        if not self.deepseek_api_key.isascii() or any(ch.isspace() for ch in self.deepseek_api_key):
            raise ValueError(
                "deepseek_config.json 中的 api_key 含有非 ASCII 字符、空格或换行，请重新复制完整密钥。"
            )
        if not self.deepseek_model or not self.deepseek_model.isascii():
            raise ValueError("模型名称无效，请在下拉框选择英文模型名称。")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "当前 Python 环境没有安装 openai SDK。请在当前解释器下运行：\n"
                f'"{sys.executable}" -m pip install openai'
            ) from exc

        client = OpenAI(
            api_key=self.deepseek_api_key,
            base_url="https://api.deepseek.com",
            timeout=90.0,
            max_retries=1,
        )

        request_args = {
            "model": self.deepseek_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            request_args["response_format"] = {"type": "json_object"}
        if thinking is not None:
            # DeepSeek 官方 Chat Completions 接口通过 extra_body 控制思考模式。
            request_args["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            }

        response = client.chat.completions.create(**request_args)
        if not getattr(response, "choices", None):
            raise RuntimeError("DeepSeek 返回结果中没有 choices。请稍后重试。")

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if not isinstance(content, str) or not content.strip():
            # 给出服务端结束原因和 token 用量，便于区分输出耗尽、资源中断等情况。
            finish_reason = getattr(choice, "finish_reason", None) or "未知"
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            details = [f"finish_reason={finish_reason}"]
            if prompt_tokens is not None:
                details.append(f"prompt_tokens={prompt_tokens}")
            if completion_tokens is not None:
                details.append(f"completion_tokens={completion_tokens}")
            if finish_reason == "length":
                hint = "模型输出达到 token 上限；可缩短解释请求或关闭思考模式后重试。"
            elif finish_reason == "insufficient_system_resource":
                hint = "DeepSeek 服务端推理资源不足，建议稍后重试。"
            elif finish_reason == "content_filter":
                hint = "服务端内容过滤未返回正文。"
            else:
                hint = "请稍后重试；如果持续发生，请检查模型状态。"
            raise RuntimeError(
                "DeepSeek 返回了空内容（" + ", ".join(details) + "）。" + hint
            )
        return content.strip()

    def _ai_current_context(self, include_results=True):
        """组装当前电路与实验设置，供 AI 解释。只传必要数据，不传本机路径或 API Key。"""
        settings = self.experiment_settings_snapshot()
        context = {
            "num_wires": settings["num_wires"],
            "gates": settings["gates"],
            "shots": settings["shots"],
            "noise": {
                "enabled": settings["noise_enabled"],
                "model": settings["noise_model"],
                "probability": settings["noise_probability"],
            },
        }

        if not include_results:
            return context

        data = self.last_results
        if data is None:
            context["latest_result"] = {
                "status": "未运行仿真",
                "note": "当前没有可以引用的实际仿真数据。不要编造概率或测量次数。",
            }
            return context

        if self.result_is_stale():
            context["latest_result"] = {
                "status": "已过期",
                "note": "上次结果对应的电路或参数已发生变化，不得把旧结果当作当前设置的仿真结果。请提示用户重新运行仿真。",
            }
            return context

        rows = []
        for index, label in enumerate(data["labels"]):
            row = {
                "basis": f"|{label}⟩",
                "ideal_probability": float(data["ideal_probs"][index]),
                "ideal_count": int(data["ideal_counts"][label]),
                "ideal_frequency": float(data["ideal_frequencies"][index]),
            }
            if data.get("noisy_probs") is not None:
                row.update({
                    "noisy_probability": float(data["noisy_probs"][index]),
                    "noisy_count": int(data["noisy_counts"][label]),
                    "noisy_frequency": float(data["noisy_frequencies"][index]),
                })
            rows.append(row)

        state_rows = []
        for label, amplitude in zip(data["labels"], data["state"]):
            if abs(amplitude) > 1e-8:
                state_rows.append({
                    "basis": f"|{label}⟩",
                    "real": float(np.real(amplitude)),
                    "imaginary": float(np.imag(amplitude)),
                    "probability": float(abs(amplitude) ** 2),
                })

        context["latest_result"] = {
            "status": "最新",
            "timestamp": data.get("timestamp", ""),
            "noise_enabled": bool(data["noise_enabled"]),
            "noise_model_name": data.get("noise_name"),
            "noise_probability": float(data["noise_probability"]),
            "probability_rows": rows,
            "nonzero_state_amplitudes": state_rows,
            "analysis_metrics": list(data.get("analysis_lines", [])),
        }
        return context

    def _validate_ai_circuit(self, payload):
        """验证 AI 生成的结构化电路，不能直接信任模型输出。"""
        if not isinstance(payload, dict):
            raise ValueError("模型返回的 JSON 顶层必须是对象。")

        n = payload.get("num_wires")
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 8:
            raise ValueError("生成方案的 num_wires 必须是 1 到 8 之间的整数。")

        raw_gates = payload.get("gates")
        if not isinstance(raw_gates, list):
            raise ValueError("生成方案中缺少 gates 列表。")
        if len(raw_gates) > 128:
            raise ValueError("一次最多允许生成 128 个量子门，请缩小电路需求。")

        allowed = set(GATE_NAMES)
        normalized = []
        occupied = {}

        for index, raw in enumerate(raw_gates, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"第 {index} 个量子门的格式不是 JSON 对象。")

            name = raw.get("name")
            if name not in allowed:
                raise ValueError(f"第 {index} 个量子门不受支持：{name!r}。")

            col = raw.get("col", index - 1)
            if isinstance(col, bool) or not isinstance(col, int) or col < 0:
                raise ValueError(f"第 {index} 个门的 col 必须是非负整数。")

            if name == "CNOT":
                control = raw.get("control")
                target = raw.get("target")
                for label, value in (("控制位", control), ("目标位", target)):
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError(f"第 {index} 个 CNOT 门的{label}必须是整数。")
                    if not 0 <= value < n:
                        raise ValueError(f"第 {index} 个 CNOT 门的{label}超出量子比特范围。")
                if control == target:
                    raise ValueError(f"第 {index} 个 CNOT 门的控制位和目标位不能相同。")
                gate = {"name": "CNOT", "control": control, "target": target, "col": col}
                wires = [control, target]
            else:
                wire = raw.get("wire")
                if isinstance(wire, bool) or not isinstance(wire, int):
                    raise ValueError(f"第 {index} 个 {name} 门的 wire 必须是整数。")
                if not 0 <= wire < n:
                    raise ValueError(f"第 {index} 个 {name} 门的线路编号超出范围。")
                gate = {"name": name, "wire": wire, "col": col}
                wires = [wire]
                if name in ROTATION_GATES:
                    try:
                        param = float(raw.get("param", np.pi / 2))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"第 {index} 个 {name} 门的 param 必须是弧度数值。") from exc
                    if not np.isfinite(param):
                        raise ValueError(f"第 {index} 个 {name} 门的 param 必须是有限数值。")
                    if abs(param) > 20 * np.pi:
                        raise ValueError(f"第 {index} 个 {name} 门的角度超出界面允许范围（±3600°）。")
                    gate["param"] = param
                else:
                    gate["param"] = float(np.pi / 2)

            for wire in wires:
                key = (col, wire)
                if key in occupied:
                    raise ValueError(
                        f"电路存在列冲突：第 {col} 列的 q{wire} 被第 {occupied[key]} 个门和第 {index} 个门同时占用。"
                    )
                occupied[key] = index

            normalized.append(gate)

        return {
            "num_wires": n,
            "gates": normalized,
            "explanation": str(payload.get("explanation", "模型未提供方案说明。")),
        }

    def open_quantum_demo(self):
        """打开 1-4 比特动态演示窗口，并复用当前 DeepSeek 讲解接口。"""
        count = int(self.canvas.num_wires)
        if not 1 <= count <= 4:
            QMessageBox.information(
                self, "量子演化演示",
                "动态演示模块目前支持 1～4 个量子比特。\n请先把工具栏中的量子比特数调整到 4 或以下。"
            )
            return
        errors = self.collect_validation_errors()
        if errors:
            QMessageBox.warning(
                self, "电路校验未通过",
                "请先修正当前电路：\n\n" + "\n".join(f"{i}. {msg}" for i, msg in enumerate(errors, 1))
            )
            return
        dialog = QuantumDemoDialog(
            count,
            copy.deepcopy(self.canvas.gates),
            ai_callback=self._call_deepseek,
            parent=self,
        )
        dialog.exec()

    def open_ai_assistant(self):
        """打开 AI 量子助手。网络请求放到后台线程，期间展示持续旋转的等待提示。"""
        dialog = SafeAssistantDialog(self)
        # 与主窗口使用同一个 app_icon.ico 图标。
        dialog.setWindowIcon(self.windowIcon())
        dialog.setWindowTitle("AI量子助手")
        dialog.resize(900, 760)
        dialog.setMinimumSize(760, 620)
        root = QVBoxLayout(dialog)

        intro = QLabel(
            "自然语言生成电路 · 解释当前实验 · 生成实验报告\n"
            "AI 负责方案和文字说明，电路有效性与仿真数值仍由本地程序检查和计算。"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型："))
        model_combo = QComboBox()
        model_combo.setEditable(False)
        model_combo.addItems(["deepseek-flash", "deepseek-v4-pro"])
        if model_combo.findText(self.deepseek_model) < 0:
            model_combo.addItem(self.deepseek_model)
        model_combo.setCurrentText(self.deepseek_model)
        model_combo.setMinimumWidth(190)
        model_row.addWidget(model_combo)
        model_save_label = QLabel("模型设置会自动保存")
        model_save_label.setStyleSheet("color:#667085; font-size:9pt;")
        model_row.addWidget(model_save_label)
        model_row.addStretch(1)
        root.addLayout(model_row)

        privacy_note = QLabel(
            "API Key 不在窗口中输入；程序从同目录 deepseek_config.json 读取本机配置，"
            "也支持环境变量 DEEPSEEK_API_KEY。配置文件含密钥，请勿发送给他人或公开上传。\n"
            "发送请求时，当前电路/实验数据和输入内容会发送至 DeepSeek API。"
        )
        privacy_note.setWordWrap(True)
        privacy_note.setStyleSheet("color:#667085; font-size:9pt;")
        root.addWidget(privacy_note)

        root.addWidget(QLabel("自然语言需求或补充说明："))
        prompt_edit = QPlainTextEdit()
        prompt_edit.setPlaceholderText(
            "生成电路示例：帮我生成三量子比特 GHZ 态。\n"
            "解释实验示例：重点解释噪声概率从 0.01 增加到 0.1 时的差异。\n"
            "生成报告时可以填写报告用途或希望重点讨论的内容。"
        )
        prompt_edit.setMaximumHeight(115)
        root.addWidget(prompt_edit)

        action_row = QHBoxLayout()
        generate_button = QPushButton("① 自然语言生成电路")
        explain_button = QPushButton("② 解释当前电路/实验")
        report_button = QPushButton("③ 生成 AI 实验报告")
        apply_button = QPushButton("应用生成的电路")
        apply_button.setEnabled(False)
        action_row.addWidget(generate_button)
        action_row.addWidget(explain_button)
        action_row.addWidget(report_button)
        action_row.addWidget(apply_button)
        root.addLayout(action_row)

        # 旋转符号由 QTimer 驱动；API 在后台线程请求，因此动画不会因网络等待而卡住。
        busy_row = QHBoxLayout()
        spinner_label = QLabel("")
        spinner_label.setStyleSheet("color:#2563eb; font-size:12pt; font-weight:600;")
        busy_row.addWidget(spinner_label)
        busy_row.addStretch(1)
        root.addLayout(busy_row)
        spinner_frames = ["◐", "◓", "◑", "◒"]
        spinner_state = {"index": 0, "busy": False, "worker": None, "receiver": None}
        dialog.busy_check = lambda: spinner_state["busy"]
        spinner_timer = QTimer(dialog)
        spinner_timer.setInterval(120)

        def animate_spinner():
            i = spinner_state["index"]
            spinner_label.setText(f"{spinner_frames[i % len(spinner_frames)]} 请稍等…")
            spinner_state["index"] = i + 1

        spinner_timer.timeout.connect(animate_spinner)

        root.addWidget(QLabel("AI 输出："))
        output_edit = QPlainTextEdit()
        output_edit.setReadOnly(True)
        output_edit.setPlaceholderText("点击上方任一功能按钮后，结果会显示在这里。")
        root.addWidget(output_edit, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        close_row.addWidget(close_button)
        root.addLayout(close_row)

        pending = {"circuit": None}
        action_buttons = [generate_button, explain_button, report_button]
        busy_controls = action_buttons + [apply_button, model_combo, close_button]

        def persist_model(model_name):
            model_name = str(model_name).strip()
            if not model_name:
                return
            self.deepseek_model = model_name
            if self._save_deepseek_config():
                model_save_label.setText("模型已保存")
                model_save_label.setStyleSheet("color:#15803d; font-size:9pt;")
            else:
                model_save_label.setText("保存失败：请检查程序目录权限")
                model_save_label.setStyleSheet("color:#b42318; font-size:9pt;")

        model_combo.currentTextChanged.connect(persist_model)

        def enable_controls():
            for item in busy_controls:
                item.setEnabled(True)
            apply_button.setEnabled(pending["circuit"] is not None)

        def finish_busy():
            spinner_timer.stop()
            spinner_label.setText("")
            spinner_state["busy"] = False
            enable_controls()

        def run_action(button, task, on_success):
            if spinner_state["busy"]:
                return
            if not self.deepseek_api_key:
                QMessageBox.warning(
                    dialog,
                    "尚未配置 API Key",
                    "界面已取消密钥输入框。请在程序 main.py 同目录的 deepseek_config.json 中填写 api_key，"
                    "然后重启程序。也可以设置系统环境变量 DEEPSEEK_API_KEY。\n\n"
                    f"配置文件位置：\n{self.deepseek_config_path}\n\n"
                    "请勿把含有真实密钥的配置文件发给他人或上传到公开仓库。"
                )
                return

            spinner_state["busy"] = True
            for item in busy_controls:
                item.setEnabled(False)
            output_edit.setPlainText("请稍等…")
            spinner_state["index"] = 0
            animate_spinner()
            spinner_timer.start()
            QApplication.processEvents()

            worker = DeepSeekWorker(task, dialog)
            spinner_state["worker"] = worker

            def handle_success(result):
                finish_busy()
                try:
                    on_success(result)
                except Exception as exc:
                    output_edit.setPlainText(
                        f"处理 AI 返回结果时出错：{exc}\n\n{traceback.format_exc()}"
                    )
                    self.set_status("AI 返回结果处理失败")

            def handle_error(error_message):
                finish_busy()
                output_edit.setPlainText(
                    f"DeepSeek 请求失败：{error_message}\n\n"
                    "请检查 API Key、网络连接、账户额度和模型名称。"
                )
                self.set_status("AI 请求失败")

            receiver = DeepSeekResultReceiver(handle_success, handle_error, dialog)
            spinner_state["receiver"] = receiver
            worker.result_ready.connect(receiver.receive_result)
            worker.failed.connect(receiver.receive_error)
            worker.finished.connect(worker.deleteLater)
            worker.start()

        def start_generate_circuit():
            request_text = prompt_edit.toPlainText().strip()
            if not request_text:
                QMessageBox.information(dialog, "需要输入需求", "请先输入电路需求，例如：生成两量子比特 Bell 态。")
                return
            system_prompt = """
你是量子电路设计助手。根据用户的自然语言描述，生成适用于本地 PySide6 + PennyLane 量子电路仿真器的电路配置。
必须输出合法 JSON，不要输出 Markdown 代码块或 JSON 以外的文字。请输出包含 num_wires、gates、explanation 的 JSON 对象。
结构示例：
{
  "num_wires": 2,
  "gates": [
    {"name": "H", "wire": 0, "col": 0},
    {"name": "CNOT", "control": 0, "target": 1, "col": 1}
  ],
  "explanation": "先对 q0 施加 H 门，再以 q0 控制 q1 执行 CNOT，构造 Bell 态。"
}
允许的 name 只有 H、X、Y、Z、RX、RY、RZ、CNOT。RX/RY/RZ 的 param 使用弧度，未指定时可以用 1.57079632679。
num_wires 必须是 1 到 8 的整数；wire、control、target 都是从 0 开始的线路编号；col 是从 0 开始的非负整数。
CNOT 必须指定 control 和 target，两者不能相同。同一列内，不同量子门不能重复占用同一条量子线路。请尽量按电路执行顺序安排 col。
输出必须包含 "json" 格式内容且符合上述结构。不要捏造当前仿真结果，因为此任务只负责提出电路方案。
""".strip()
            user_prompt = (
                f"用户的电路需求：\n{request_text}\n\n"
                f"当前画布量子比特数为 {self.canvas.num_wires}。"
                "如果用户明确指定比特数，以用户指定值为准；否则优先采用当前画布比特数。"
            )

            def task():
                raw = self._call_deepseek(system_prompt, user_prompt, json_mode=True, max_tokens=4000)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"模型返回的内容不是有效 JSON：{exc}\n\n原始输出：\n{raw[:2000]}") from exc
                return self._validate_ai_circuit(payload)

            def on_success(candidate):
                pending["circuit"] = candidate
                self.ai_pending_circuit = candidate
                apply_button.setEnabled(True)
                gate_descriptions = []
                for i, gate in enumerate(candidate["gates"], start=1):
                    if gate["name"] == "CNOT":
                        detail = f"CNOT：q{gate['control']} 控制 q{gate['target']}"
                    else:
                        detail = f"{gate['name']}：作用于 q{gate['wire']}"
                        if gate["name"] in ROTATION_GATES:
                            detail += f"，角度 {np.degrees(gate['param']):.2f}°"
                    gate_descriptions.append(f"{i}. 第 {gate['col']} 列：{detail}")
                output_edit.setPlainText(
                    "AI 已生成电路方案，尚未修改当前画布。\n\n"
                    f"量子比特数：{candidate['num_wires']}\n"
                    f"量子门数量：{len(candidate['gates'])}\n\n"
                    "【门序列】\n" + ("\n".join(gate_descriptions) or "（空电路）")
                    + "\n\n【方案说明】\n" + candidate["explanation"]
                    + "\n\n请检查方案后点击“应用生成的电路”。应用后可用 Ctrl+Z 撤销。"
                )
                self.set_status("AI 已生成并通过本地规则检查的电路方案")

            run_action(generate_button, task, on_success)

        def apply_generated_circuit():
            candidate = pending.get("circuit")
            if candidate is None:
                return
            answer = QMessageBox.question(
                dialog, "应用 AI 生成的电路？",
                "应用后会替换当前画布中的电路，但可以使用 Ctrl+Z 撤销。\n是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.record_history()
            self.wire_spin.blockSignals(True)
            self.wire_spin.setValue(candidate["num_wires"])
            self.wire_spin.blockSignals(False)
            self.canvas.num_wires = candidate["num_wires"]
            self.canvas.gates = copy.deepcopy(candidate["gates"])
            self.canvas.select_gate(None)
            self.canvas.adjust_canvas()
            self.canvas.update()
            self.refresh_result_status()
            self.set_status("已将 AI 生成的电路应用到画布")
            output_edit.appendPlainText("\n\n已应用到画布。请点击“运行仿真”计算真实结果。")

        def start_explain_current():
            focus = prompt_edit.toPlainText().strip() or "请先解释电路的工作原理，再解释理论概率、有限次测量和噪声结果。"
            context_json = json.dumps(self._ai_current_context(include_results=True), ensure_ascii=False, indent=2)
            system_prompt = """
你是一名量子计算教学与实验助手。请使用中文，面向本科生解释电路和仿真结果。
必须优先依据提供的结构化上下文，不要编造概率、测量次数、噪声数据或程序未提供的结论。
清楚区分理论概率与有限次采样频率；有噪声时说明指定噪声通道和参数。若结果状态是已过期或尚未运行，必须明确说明当前没有与当前设置对应的有效仿真结果，不可把旧数据说成当前结果。
可以讲解量子门的作用、量子态演化的直觉、实验现象、可能的统计波动和下一步可做的对照实验。不要声称程序连接了真实量子硬件。
""".strip()
            user_prompt = (
                "下面是本地程序当前电路、设置和（如有效）实际仿真结果，请基于这些数据进行解释。\n"
                "结构化上下文 JSON：\n" + context_json
                + "\n\n用户希望重点了解：\n" + focus
            )
            def task():
                return self._call_deepseek(system_prompt, user_prompt, json_mode=False, max_tokens=5000, thinking=False)
            def on_success(response_text):
                pending["circuit"] = None
                apply_button.setEnabled(False)
                output_edit.setPlainText(response_text)
                self.set_status("AI 已完成电路与实验解释")
            run_action(explain_button, task, on_success)

        def start_generate_report():
            data = self.get_exportable_results()
            if data is None:
                output_edit.setPlainText(
                    "没有可供报告使用的最新仿真结果。请先运行仿真；如果结果已过期，请重新运行后再生成报告。"
                )
                return
            report_data = self._ai_current_context(include_results=True)
            report_data["report_requirements"] = prompt_edit.toPlainText().strip() or "生成适合本科毕业设计实验章节的中文 Markdown 报告。"
            report_data["report_scope"] = {
                "include_circuit_description": True,
                "include_method": True,
                "include_measured_results": True,
                "include_noise_comparison_if_available": True,
                "include_metrics": True,
                "include_limitations": True,
            }
            system_prompt = """
你是一名严谨的量子计算实验报告撰写助手。根据用户提供的本地程序实际数据，用中文输出完整 Markdown 报告。
报告建议包括：标题、实验目的、实验环境与参数、量子门序列、仿真方法、理论概率与实际测量结果、噪声对比（仅在提供数据时）、指标解释、讨论与局限、结论。
严禁捏造数据、理论值、引用文献、设备运行记录或未提供的实验。所有数值必须来自输入数据，允许四舍五入但应保持一致。若某项数据没有提供，应明确写“本次数据未提供”或不讨论。解释理论分布与采样波动的差异时应谨慎。只输出 Markdown 正文，不要在开头加“以下是报告”等客套语。
""".strip()
            user_prompt = "请用下面的真实实验数据撰写报告。不要把 AI 的推测当作实验测量值。\n\n" + json.dumps(report_data, ensure_ascii=False, indent=2)
            def task():
                return self._call_deepseek(system_prompt, user_prompt, json_mode=False, max_tokens=7000)
            def on_success(report):
                output_edit.setPlainText(report)
                pending["circuit"] = None
                apply_button.setEnabled(False)
                path, _ = QFileDialog.getSaveFileName(
                    dialog, "保存 AI 实验报告", "ai_quantum_experiment_report.md",
                    "Markdown 报告 (*.md);;文本文件 (*.txt)",
                )
                if path:
                    if Path(path).suffix.lower() not in (".md", ".txt"):
                        path += ".md"
                    with open(path, "w", encoding="utf-8") as file:
                        file.write(report.rstrip() + "\n")
                    QMessageBox.information(dialog, "报告已保存", f"AI 实验报告已保存到：\n{path}")
                    self.set_status(f"AI 实验报告已保存：{path}")
            run_action(report_button, task, on_success)

        generate_button.clicked.connect(start_generate_circuit)
        explain_button.clicked.connect(start_explain_current)
        report_button.clicked.connect(start_generate_report)
        apply_button.clicked.connect(apply_generated_circuit)

        dialog.exec()
