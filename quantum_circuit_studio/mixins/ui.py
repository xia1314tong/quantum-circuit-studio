"""主界面构建、状态管理与控件事件。"""
from ..common import (
    Figure, FigureCanvasQTAgg, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QVBoxLayout, QWidget, Qt, ROTATION_GATES, copy,
    np
)
from ..widgets import CircuitCanvas, GatePalette
from ..algorithm_lab import AlgorithmLabDialog


class UIMixin:
    def open_algorithm_lab(self):
        """打开独立算法实验室；不改变原有量子演化教学演示。"""
        dialog = AlgorithmLabDialog(self)
        dialog.exec()

    def build_ui(self):

        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)

        title = QLabel("Quantum Circuit Studio")

        title.setStyleSheet(
            "font-size:22px;"
            "font-weight:bold;"
            "color:#1e293b;"
        )

        layout.addWidget(title)

        intro_row = QHBoxLayout()
        intro_row.addWidget(
            QLabel(
                "拖拽量子门构建线路，"
                "使用 PennyLane 执行量子仿真"
            )
        )
        intro_row.addStretch(1)
        self.ai_assistant_button = QPushButton("AI量子助手")
        self.ai_assistant_button.setMinimumHeight(30)
        self.ai_assistant_button.clicked.connect(self.open_ai_assistant)
        intro_row.addWidget(self.ai_assistant_button)
        self.quantum_demo_button = QPushButton("量子演化演示")
        self.quantum_demo_button.setMinimumHeight(30)
        self.quantum_demo_button.clicked.connect(self.open_quantum_demo)
        intro_row.addWidget(self.quantum_demo_button)
        self.algorithm_lab_button = QPushButton("量子算法")
        self.algorithm_lab_button.setMinimumHeight(30)
        self.algorithm_lab_button.clicked.connect(self.open_algorithm_lab)
        intro_row.addWidget(self.algorithm_lab_button)
        layout.addLayout(intro_row)

        # 顶部工具栏
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("量子比特数："))

        self.wire_spin = QSpinBox()
        self.wire_spin.setRange(1, 8)
        self.wire_spin.setValue(2)
        self.wire_spin.setFixedWidth(70)
        self.wire_spin.valueChanged.connect(
            self.change_wires
        )
        toolbar.addWidget(self.wire_spin)

        # 测量次数设置
        toolbar.addWidget(QLabel("测量次数："))

        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(100, 100000)
        self.shots_spin.setValue(1000)
        self.shots_spin.setSingleStep(100)
        self.shots_spin.setFixedWidth(100)
        self.shots_spin.valueChanged.connect(self.refresh_result_status)
        toolbar.addWidget(self.shots_spin)

        self.undo_button = QPushButton("撤销")
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)
        toolbar.addWidget(self.undo_button)

        self.redo_button = QPushButton("重做")
        self.redo_button.clicked.connect(self.redo)
        self.redo_button.setEnabled(False)
        toolbar.addWidget(self.redo_button)

        buttons = [
            ("检查电路", self.check_circuit),
            ("运行仿真", self.run_simulation),
            ("生成 Python 代码", self.show_generated_code),
            ("清空线路", self.clear_circuit),
            ("保存电路", self.save_circuit),
            ("加载电路", self.load_circuit),
            ("Bell 态示例", self.bell_example)
        ]

        for text, callback in buttons:
            button = QPushButton(text)
            button.clicked.connect(callback)
            toolbar.addWidget(button)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 主工作区：上下可拖动调整高度
        # 最外层垂直分隔器：上方为量子线路画布，下方为参数、噪声与结果。
        # 拖动两者之间的横向分隔条，可以自由调整画布高度。
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)

        # 上方：工具箱 + 线路画布
        workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        workspace_splitter.setChildrenCollapsible(False)
        workspace_splitter.setHandleWidth(8)
        self.workspace_splitter = workspace_splitter

        toolbox = QGroupBox("量子门工具箱")
        toolbox_layout = QVBoxLayout(toolbox)
        toolbox_layout.addWidget(
            QLabel("按住门拖到右侧线路：")
        )

        self.palette = GatePalette()
        toolbox_layout.addWidget(self.palette)
        toolbox_layout.addWidget(
            QLabel("旋转门可调角度")
        )
        toolbox.setMinimumWidth(215)
        toolbox.setMaximumWidth(280)
        workspace_splitter.addWidget(toolbox)

        self.canvas = CircuitCanvas(
            self.set_status,
            self.on_gate_selected,
            self.record_history
        )

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        scroll.setMinimumWidth(400)
        scroll.setMinimumHeight(270)
        self.canvas_scroll = scroll
        workspace_splitter.addWidget(scroll)
        workspace_splitter.setStretchFactor(0, 0)
        workspace_splitter.setStretchFactor(1, 1)
        workspace_splitter.setSizes([235, 1000])

        self.main_splitter.addWidget(workspace_splitter)

        # 下方独立滚动区：可独立于上方线路画布滚动，避免结果图被挤出可视区域。
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(8, 8, 8, 8)
        bottom_layout.setSpacing(8)
        self.bottom_panel = bottom_panel

        # 合并面板：选中量子门参数 + 噪声设置
        settings_box = QGroupBox("量子门参数与噪声设置")
        settings_layout = QHBoxLayout(settings_box)
        settings_layout.setContentsMargins(12, 12, 12, 10)
        settings_layout.setSpacing(8)

        settings_layout.addWidget(QLabel("当前门："))
        self.selected_label = QLabel("未选择")
        self.selected_label.setMinimumWidth(95)
        settings_layout.addWidget(self.selected_label)

        settings_layout.addWidget(QLabel("角度："))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-3600.0, 3600.0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setSingleStep(15.0)
        self.angle_spin.setValue(90.0)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setFixedWidth(110)
        self.angle_spin.setEnabled(False)
        self.angle_spin.valueChanged.connect(self.change_gate_angle)
        settings_layout.addWidget(self.angle_spin)

        self.angle_radians_label = QLabel("弧度：—")
        self.angle_radians_label.setMinimumWidth(105)
        settings_layout.addWidget(self.angle_radians_label)

        # 视觉分隔符：参数设置与噪声设置仍属于同一个面板
        separator = QLabel("│")
        separator.setStyleSheet("color: #D0D5DD; font-size: 18px;")
        settings_layout.addWidget(separator)

        self.noise_enabled = QCheckBox("启用噪声")
        self.noise_enabled.setChecked(False)
        settings_layout.addWidget(self.noise_enabled)

        settings_layout.addWidget(QLabel("模型："))
        self.noise_model_combo = QComboBox()
        self.noise_model_combo.addItem("比特翻转", "bitflip")
        self.noise_model_combo.addItem("相位翻转", "phaseflip")
        self.noise_model_combo.addItem("退极化", "depolarizing")
        self.noise_model_combo.setMinimumWidth(130)
        settings_layout.addWidget(self.noise_model_combo)

        settings_layout.addWidget(QLabel("概率："))
        self.noise_probability = QDoubleSpinBox()
        self.noise_probability.setRange(0.0, 1.0)
        self.noise_probability.setDecimals(3)
        self.noise_probability.setSingleStep(0.01)
        self.noise_probability.setValue(0.05)
        self.noise_probability.setFixedWidth(95)
        settings_layout.addWidget(self.noise_probability)
        settings_layout.addStretch(1)

        self.noise_enabled.toggled.connect(self.update_noise_controls)
        self.noise_enabled.toggled.connect(self.refresh_result_status)
        self.noise_model_combo.currentIndexChanged.connect(self.refresh_result_status)
        self.noise_probability.valueChanged.connect(self.refresh_result_status)
        self.update_noise_controls()
        bottom_layout.addWidget(settings_box)

        # 结果区：明细和图表左右排列，可独立调整宽度
        result_box = QGroupBox("仿真结果")
        self.result_box = result_box
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(8, 12, 8, 8)

        # 当前实验结果状态：未运行、最新或已过期
        result_state_row = QHBoxLayout()
        result_state_row.addWidget(QLabel("结果状态："))
        self.result_state_label = QLabel("尚未运行仿真")
        self.result_state_label.setStyleSheet(
            "color: #667085; font-weight: 600; padding: 3px 6px;"
        )
        result_state_row.addWidget(self.result_state_label)
        result_state_row.addStretch(1)
        result_layout.addLayout(result_state_row)

        result_splitter = QSplitter(Qt.Orientation.Horizontal)
        result_splitter.setChildrenCollapsible(False)
        result_splitter.setHandleWidth(8)
        self.result_splitter = result_splitter
        result_splitter.setMinimumHeight(390)

        text_panel = QGroupBox("计算结果明细")
        text_layout = QVBoxLayout(text_panel)
        self.result_label = QLabel("点击“运行仿真”查看结果。")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.result_label.setMinimumWidth(0)

        result_text_scroll = QScrollArea()
        result_text_scroll.setWidgetResizable(True)
        result_text_scroll.setMinimumWidth(0)
        result_text_scroll.setWidget(self.result_label)
        text_layout.addWidget(result_text_scroll)
        text_panel.setMinimumWidth(260)
        result_splitter.addWidget(text_panel)

        chart_panel = QGroupBox("概率分布")
        chart_layout = QVBoxLayout(chart_panel)
        self.figure = Figure(figsize=(7.0, 4.0), tight_layout=True)
        self.chart = FigureCanvasQTAgg(self.figure)
        self.chart.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        # 给图表足够的最小空间，防止被压成看不见的窄条
        self.chart.setMinimumSize(400, 310)
        chart_layout.addWidget(self.chart)
        chart_panel.setMinimumWidth(420)
        result_splitter.addWidget(chart_panel)

        result_splitter.setStretchFactor(0, 2)
        result_splitter.setStretchFactor(1, 3)
        result_splitter.setSizes([440, 700])
        result_layout.addWidget(result_splitter, 1)
        result_box.setMinimumHeight(430)
        bottom_layout.addWidget(result_box, 0)

        # 实验分析：独立面板，不再挤占图表区域
        analysis_box = QGroupBox("实验分析")
        self.analysis_box = analysis_box
        analysis_layout = QVBoxLayout(analysis_box)
        self.analysis_label = QLabel("运行仿真后，这里会显示电路深度、熵和分布差异等分析指标。")
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.analysis_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.analysis_label.setMinimumWidth(0)
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setMinimumHeight(90)
        analysis_scroll.setMaximumHeight(150)
        analysis_scroll.setWidget(self.analysis_label)
        analysis_layout.addWidget(analysis_scroll)
        analysis_box.setMinimumHeight(125)
        bottom_layout.addWidget(analysis_box, 0)

        # 导出操作独立排列
        export_box = QGroupBox("实验结果导出")
        export_row = QHBoxLayout(export_box)
        export_row.addWidget(QLabel("保存结果："))

        self.export_csv_button = QPushButton("导出 CSV")
        self.export_csv_button.clicked.connect(self.export_csv)
        export_row.addWidget(self.export_csv_button)

        self.export_chart_button = QPushButton("保存概率图 PNG")
        self.export_chart_button.clicked.connect(self.export_chart)
        export_row.addWidget(self.export_chart_button)

        self.export_report_button = QPushButton("导出实验报告")
        self.export_report_button.clicked.connect(self.export_report)
        export_row.addWidget(self.export_report_button)
        export_row.addStretch(1)
        bottom_layout.addWidget(export_box)

        # -------------------------
        # 实验配置管理：保存 / 恢复线路、测量和噪声设置
        # -------------------------
        config_box = QGroupBox("实验配置管理")
        config_layout = QHBoxLayout(config_box)
        config_layout.addWidget(
            QLabel("保存完整实验设置，方便下次继续复现实验：")
        )

        self.save_config_button = QPushButton("保存实验配置")
        self.save_config_button.clicked.connect(self.save_experiment_config)
        config_layout.addWidget(self.save_config_button)

        self.load_config_button = QPushButton("加载实验配置")
        self.load_config_button.clicked.connect(self.load_experiment_config)
        config_layout.addWidget(self.load_config_button)

        config_layout.addStretch(1)
        bottom_layout.addWidget(config_box)

        # 内容高度大于视口时，以下滚动区会显示独立的垂直滚动条。
        # 因此上方画布和下方参数/结果可以各自滚动，不互相抢占空间。
        bottom_panel.setMinimumHeight(850)
        bottom_scroll = QScrollArea()
        bottom_scroll.setObjectName("bottomResultsScroll")
        bottom_scroll.setWidgetResizable(True)
        bottom_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        bottom_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        bottom_scroll.setMinimumHeight(260)
        bottom_scroll.setWidget(bottom_panel)
        self.bottom_scroll = bottom_scroll

        self.main_splitter.addWidget(bottom_scroll)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([430, 380])

        layout.addWidget(self.main_splitter, 1)
        self.statusBar().showMessage("就绪")
        self.refresh_result_status()

    def update_noise_controls(self, *_):
        """根据是否启用噪声，切换噪声配置控件的可用状态。"""
        enabled = self.noise_enabled.isChecked()
        self.noise_model_combo.setEnabled(enabled)
        self.noise_probability.setEnabled(enabled)

    def set_status(self, text):

        self.statusBar().showMessage(text, 4000)
        self.refresh_result_status()

    def experiment_settings_snapshot(self):
        """返回当前电路、测量次数和噪声设置。"""
        return {
            "num_wires": self.canvas.num_wires,
            "gates": copy.deepcopy(self.canvas.gates),
            "shots": self.shots_spin.value(),
            "noise_enabled": self.noise_enabled.isChecked(),
            "noise_model": self.noise_model_combo.currentData(),
            "noise_probability": float(self.noise_probability.value())
        }

    def result_is_stale(self):
        """判断上次结果是否仍对应当前电路与仿真参数。"""
        data = self.last_results
        if data is None:
            return False

        current = self.experiment_settings_snapshot()
        return (
                data.get("num_wires") != current["num_wires"]
                or data.get("gates") != current["gates"]
                or data.get("shots") != current["shots"]
                or data.get("noise_enabled") != current["noise_enabled"]
                or data.get("noise_model") != current["noise_model"]
                or not np.isclose(
            float(data.get("noise_probability", -1.0)),
            current["noise_probability"]
        )
        )

    def refresh_result_status(self, *_):
        """在结果面板中显示结果状态。"""
        label = getattr(self, "result_state_label", None)
        if label is None:
            return

        if self.last_results is None:
            label.setText("尚未运行仿真，请运行后再导出结果")
            label.setStyleSheet(
                "color: #667085; font-weight: 600; padding: 3px 6px;"
            )
        elif self.result_is_stale():
            label.setText("结果已过期：电路或设置已改变，请重新运行仿真")
            label.setStyleSheet(
                "color: #B42318; background-color: #FEF3F2; "
                "font-weight: 700; padding: 4px 8px; border-radius: 4px;"
            )
        else:
            timestamp = self.last_results.get("timestamp", "")
            label.setText(f"结果是最新的  ·  仿真时间：{timestamp}")
            label.setStyleSheet(
                "color: #027A48; background-color: #ECFDF3; "
                "font-weight: 700; padding: 4px 8px; border-radius: 4px;"
            )

    def on_gate_selected(self, gate):

        self.angle_spin.blockSignals(True)

        if gate is None:

            self.selected_label.setText("未选择")
            self.angle_spin.setEnabled(False)
            self.angle_radians_label.setText("弧度：—")

        else:

            name = gate["name"]

            if name in ROTATION_GATES:

                radians = float(
                    gate.get("param", np.pi / 2)
                )

                degrees = float(np.degrees(radians))

                self.selected_label.setText(
                    f"{name}（q{gate['wire']}）"
                )

                self.angle_spin.setEnabled(True)
                self.angle_spin.setValue(degrees)

                self.angle_radians_label.setText(
                    f"弧度：{radians:.6f}"
                )

            elif name == "CNOT":

                self.selected_label.setText(
                    f"CNOT（q{gate['control']}→q{gate['target']}）"
                )

                self.angle_spin.setEnabled(False)
                self.angle_radians_label.setText("弧度：不适用")

            else:

                self.selected_label.setText(
                    f"{name}（q{gate['wire']}）"
                )

                self.angle_spin.setEnabled(False)
                self.angle_radians_label.setText("弧度：不适用")

        self.angle_spin.blockSignals(False)
        self.refresh_result_status()

    def change_gate_angle(self, degrees):

        index = self.canvas.selected

        if index is None:
            return

        if index >= len(self.canvas.gates):
            return

        gate = self.canvas.gates[index]

        if gate["name"] not in ROTATION_GATES:
            return

        radians = float(np.radians(degrees))

        old_value = float(
            gate.get("param", np.pi / 2)
        )

        if np.isclose(old_value, radians):
            return

        # 修改前保存历史状态
        self.record_history()

        gate["param"] = radians

        self.angle_radians_label.setText(
            f"弧度：{radians:.6f}"
        )

        self.canvas.update()

        self.set_status(
            f"{gate['name']} 门角度已更新为 {degrees:.2f}°"
        )

    def change_wires(self, count):

        self.canvas.set_wire_count(count)
        self.refresh_result_status()
