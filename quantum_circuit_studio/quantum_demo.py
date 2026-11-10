"""量子电路动态演示：1-4 比特状态演化、布洛赫球、概率图与 AI 讲解。"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QPlainTextEdit, QGroupBox, QMessageBox, QSplitter
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _gate_matrix(gate):
    name = gate['name']
    if name == 'H': return H2
    if name == 'X': return X2
    if name == 'Y': return Y2
    if name == 'Z': return Z2
    if name in ('RX', 'RY', 'RZ'):
        a = float(gate.get('param', np.pi / 2))
        p = {'RX': X2, 'RY': Y2, 'RZ': Z2}[name]
        return np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * p
    return None


def _apply_single(state, matrix, wire, n):
    tensor = state.reshape([2] * n)
    changed = np.tensordot(matrix, tensor, axes=([1], [wire]))
    changed = np.moveaxis(changed, 0, wire)
    return changed.reshape(-1)


def _apply_cnot(state, control, target, n):
    out = np.zeros_like(state)
    for index, amp in enumerate(state):
        bits = [(index >> (n - 1 - q)) & 1 for q in range(n)]
        new_bits = bits.copy()
        if bits[control]:
            new_bits[target] ^= 1
        new_index = 0
        for bit in new_bits:
            new_index = (new_index << 1) | bit
        out[new_index] += amp
    return out


def _apply_gate(state, gate, n):
    if gate['name'] == 'CNOT':
        return _apply_cnot(state, int(gate['control']), int(gate['target']), n)
    return _apply_single(state, _gate_matrix(gate), int(gate['wire']), n)


def _bloch_vector(state, n, wire):
    """从纯态计算选定比特的约化态 Bloch 向量；纠缠时向量可位于球内。"""
    vals = []
    for pauli in (X2, Y2, Z2):
        transformed = _apply_single(state, pauli, wire, n)
        vals.append(float(np.real(np.vdot(state, transformed))))
    return np.asarray(vals)


class ExplainWorker(QThread):
    done = Signal(str)
    failed = Signal(str)
    def __init__(self, callback, prompt, parent=None):
        super().__init__(parent)
        self.callback, self.prompt = callback, prompt
    def run(self):
        try:
            answer = self.callback(
                '你是一名严谨的量子计算助教。必须依据给定的电路步骤和计算数据讲解，不得改写或编造状态、概率。用中文回答，结构清楚、适合本科生。',
                self.prompt, max_tokens=1200, thinking=False
            )
            self.done.emit(answer)
        except Exception as exc:
            self.failed.emit(str(exc))


class QuantumDemoDialog(QDialog):
    def __init__(self, num_wires, gates, ai_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('量子电路动态演示 · AI 量子讲解')
        self.resize(1120, 780)
        self.n = int(num_wires)
        self.gates = sorted([dict(g) for g in gates], key=lambda g: int(g.get('col', 0)))
        self.ai_callback = ai_callback
        self.snapshots = []
        self.step = 0
        self.worker = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_step)
        self._build_ui()
        self._prepare_snapshots()
        self._render_step()

    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel('播放控制：完整播放 / 分步前进 / 回退'))
        top.addWidget(QLabel('查看局部状态：'))
        self.wire_combo = QComboBox()
        for i in range(self.n): self.wire_combo.addItem(f'q{i}', i)
        self.wire_combo.currentIndexChanged.connect(self._render_step)
        top.addWidget(self.wire_combo)
        top.addWidget(QLabel('播放速度：'))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        self.speed_slider.setMaximumWidth(130)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        top.addWidget(self.speed_slider)
        root.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); left_layout = QVBoxLayout(left)
        self.step_label = QLabel('准备演示')
        self.step_label.setStyleSheet('font-size:16px;font-weight:bold;color:#1D4ED8;')
        left_layout.addWidget(self.step_label)
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        left_layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.reset_btn = QPushButton('⏮ 重置'); self.prev_btn = QPushButton('◀ 上一步')
        self.play_btn = QPushButton('▶ 完整播放'); self.next_btn = QPushButton('下一步 ▶')
        self.reset_btn.clicked.connect(self.reset_demo)
        self.prev_btn.clicked.connect(self.prev_step)
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn.clicked.connect(self.next_step)
        for b in (self.reset_btn, self.prev_btn, self.play_btn, self.next_btn): controls.addWidget(b)
        left_layout.addLayout(controls)
        self.progress = QLabel(''); left_layout.addWidget(self.progress)
        split.addWidget(left)

        right = QWidget(); right_layout = QVBoxLayout(right)
        explain_box = QGroupBox('AI 量子讲解员')
        explain_layout = QVBoxLayout(explain_box)
        self.explain_text = QPlainTextEdit()
        self.explain_text.setReadOnly(True)
        self.explain_text.setPlaceholderText('点击“讲解当前步骤”，AI 将结合真实计算状态进行解释。')
        explain_layout.addWidget(self.explain_text, 1)
        self.level_combo = QComboBox(); self.level_combo.addItems(['入门讲解', '原理讲解', '数学推导'])
        explain_layout.addWidget(self.level_combo)
        self.ask_text = QPlainTextEdit(); self.ask_text.setMaximumHeight(82)
        self.ask_text.setPlaceholderText('也可以提问，例如：为什么这个门会改变概率？')
        explain_layout.addWidget(self.ask_text)
        row = QHBoxLayout()
        self.explain_btn = QPushButton('✦ 讲解当前步骤')
        self.ask_btn = QPushButton('发送问题')
        self.explain_btn.clicked.connect(self.explain_current)
        self.ask_btn.clicked.connect(self.ask_question)
        row.addWidget(self.explain_btn); row.addWidget(self.ask_btn)
        explain_layout.addLayout(row)
        right_layout.addWidget(explain_box, 1)
        note = QLabel('说明：布洛赫球展示选中量子比特的局部状态；纠缠时向量可能位于球内。联合概率图展示完整计算基态分布。')
        note.setWordWrap(True); note.setStyleSheet('color:#667085;')
        right_layout.addWidget(note)
        split.addWidget(right); split.setSizes([680, 420])
        root.addWidget(split, 1)

    def _prepare_snapshots(self):
        state = np.zeros(2 ** self.n, dtype=complex); state[0] = 1
        self.snapshots = [{'state': state.copy(), 'gate': None, 'description': '初始状态'}]
        for gate in self.gates:
            state = _apply_gate(state, gate, self.n)
            norm = np.linalg.norm(state)
            if norm: state = state / norm
            self.snapshots.append({'state': state.copy(), 'gate': gate, 'description': self._gate_description(gate)})

    def _gate_description(self, g):
        name = g['name']
        if name == 'CNOT': return f"CNOT：控制 q{g['control']}，目标 q{g['target']}"
        wire = g.get('wire', 0)
        if name in ('RX', 'RY', 'RZ'):
            return f"{name}(q{wire}, {float(g.get('param', np.pi/2)):.3f} rad)"
        return f'{name}(q{wire})'

    def _render_step(self, *_):
        if not self.snapshots: return
        item = self.snapshots[self.step]
        state = item['state']
        self.figure.clear()
        ax = self.figure.add_subplot(121, projection='3d')
        u = np.linspace(0, 2*np.pi, 36); v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v)); z = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color='#C7D2E0', linewidth=.45, alpha=.65)
        ax.plot([-1,1],[0,0],[0,0], color='#64748B', linewidth=.8)
        ax.plot([0,0],[-1,1],[0,0], color='#64748B', linewidth=.8)
        ax.plot([0,0],[0,0],[-1,1], color='#64748B', linewidth=.8)
        wire = int(self.wire_combo.currentData() or 0)
        vec = _bloch_vector(state, self.n, wire)
        ax.quiver(0,0,0,*vec,color='#2563EB',linewidth=2.5,arrow_length_ratio=.15)
        ax.text(1.08,0,0,'X'); ax.text(0,1.08,0,'Y'); ax.text(0,0,1.08,'Z')
        ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.15,1.15); ax.set_zlim(-1.15,1.15)
        ax.set_box_aspect((1,1,1)); ax.set_title(f'q{wire} 布洛赫球')
        ax.view_init(elev=22, azim=35); ax.set_axis_off()
        ax2 = self.figure.add_subplot(122)
        probs = np.abs(state) ** 2
        labels = [format(i, f'0{self.n}b') for i in range(2**self.n)]
        ax2.bar(labels, probs, color='#60A5FA', edgecolor='#2563EB')
        ax2.set_ylim(0, 1.05); ax2.set_ylabel('概率'); ax2.set_title('联合测量概率')
        ax2.tick_params(axis='x', labelrotation=65, labelsize=8)
        ax2.grid(axis='y', alpha=.2)
        self.canvas.draw_idle()
        self.step_label.setText(f'第 {self.step}/{len(self.snapshots)-1} 步：{item["description"]}')
        self.progress.setText(f'进度：{self.step+1}/{len(self.snapshots)}    |ψ⟩ 的非零振幅：{sum(abs(state)>1e-8)} 项')
        self.prev_btn.setEnabled(self.step > 0)
        self.next_btn.setEnabled(self.step < len(self.snapshots)-1)
        if self.step == len(self.snapshots)-1 and self.timer.isActive():
            self.timer.stop(); self.play_btn.setText('▶ 完整播放')

    def _speed_changed(self, value):
        if self.timer.isActive(): self.timer.setInterval(int(2200 - value*190))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop(); self.play_btn.setText('▶ 继续播放')
            return
        if self.step >= len(self.snapshots)-1: self.step = 0; self._render_step()
        self.timer.start(int(2200 - self.speed_slider.value()*190))
        self.play_btn.setText('⏸ 暂停')

    def next_step(self):
        if self.step < len(self.snapshots)-1:
            self.step += 1; self._render_step()
        else:
            self.timer.stop(); self.play_btn.setText('▶ 完整播放')

    def prev_step(self):
        self.timer.stop(); self.play_btn.setText('▶ 完整播放')
        self.step = max(0, self.step-1); self._render_step()

    def reset_demo(self):
        self.timer.stop(); self.play_btn.setText('▶ 完整播放')
        self.step = 0; self._render_step()

    def _current_prompt(self, question=None):
        item = self.snapshots[self.step]
        state = item['state']
        nonzero = []
        for i, amp in enumerate(state):
            if abs(amp) > 1e-8:
                nonzero.append({'basis': '|' + format(i, f'0{self.n}b') + '>', 'real': round(float(amp.real), 6), 'imag': round(float(amp.imag), 6), 'probability': round(float(abs(amp)**2), 6)})
        return (
            f'请用{self.level_combo.currentText()}讲解当前量子电路步骤。\n'
            f'量子比特数：{self.n}；当前步骤：{self.step}/{len(self.snapshots)-1}；'
            f'当前操作：{item["description"]}。\n'
            f'当前量子态非零振幅（q0 为最高位）：{nonzero}\n'
            f'选中 q{self.wire_combo.currentData()} 的 Bloch 向量：{np.round(_bloch_vector(state,self.n,int(self.wire_combo.currentData())),6).tolist()}。\n'
            f'联合概率：{[(format(i, f"0{self.n}b"), round(float(abs(a)**2),6)) for i,a in enumerate(state) if abs(a)**2>1e-8]}。\n'
            + (f'用户追问：{question}\n' if question else '请解释量子门的作用、输入输出状态变化和概率含义。')
        )

    def explain_current(self): self._request_ai(None)
    def ask_question(self):
        question = self.ask_text.toPlainText().strip()
        if not question:
            QMessageBox.information(self, '提示', '请先输入问题。'); return
        self._request_ai(question)

    def _request_ai(self, question):
        if not callable(self.ai_callback):
            self.explain_text.setPlainText('AI 接口不可用。动画与概率图仍可正常使用；请检查主程序 DeepSeek 配置。')
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, '提示', 'AI 正在生成讲解，请稍候。'); return
        self.explain_btn.setEnabled(False); self.ask_btn.setEnabled(False)
        self.explain_text.setPlainText('AI 正在结合当前步骤和真实状态数据生成讲解……')
        self.worker = ExplainWorker(self.ai_callback, self._current_prompt(question), self)
        self.worker.done.connect(self._ai_done); self.worker.failed.connect(self._ai_failed)
        self.worker.finished.connect(self._ai_finished)
        self.worker.start()

    def _ai_done(self, text): self.explain_text.setPlainText(text)
    def _ai_failed(self, message): self.explain_text.setPlainText('AI 讲解暂时失败：' + message)
    def _ai_finished(self): self.explain_btn.setEnabled(True); self.ask_btn.setEnabled(True)

    def closeEvent(self, event):
        self.timer.stop()
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, 'AI 正在讲解', '请等待当前 AI 请求完成后再关闭演示窗口。')
            event.ignore()
            return
        super().closeEvent(event)
