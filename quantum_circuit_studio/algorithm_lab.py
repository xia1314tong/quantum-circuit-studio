"""量子算法实验室：Grover、Deutsch–Jozsa、隐形传态与 QFT。

该模块独立于原有 quantum_demo 教学演示，避免覆盖旧功能。
算法结果使用状态向量/矩阵进行数值验证；代码导出为 PennyLane 可运行脚本。
"""
from pathlib import Path
import math
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QPlainTextEdit, QGroupBox,
    QSplitter, QFileDialog, QMessageBox, QDoubleSpinBox
)


class AlgorithmLabDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("量子算法实验室")
        self.resize(1100, 780)
        self.last_code = ""
        root = QVBoxLayout(self)
        intro = QLabel("量子算法实验室 · 数学原理 / 电路步骤 / 数值验证 / PennyLane 代码导出")
        intro.setStyleSheet("font-size:17px;font-weight:bold;color:#1d4ed8;padding:5px")
        root.addWidget(intro)
        form_box = QGroupBox("实验参数")
        form = QFormLayout(form_box)
        self.algorithm = QComboBox()
        self.algorithm.addItems(["Grover 量子搜索", "Deutsch–Jozsa", "量子隐形传态", "量子傅里叶变换 QFT"])
        self.n_spin = QSpinBox(); self.n_spin.setRange(2, 6); self.n_spin.setValue(3)
        self.target = QComboBox()
        self.iter_spin = QSpinBox(); self.iter_spin.setRange(0, 20); self.iter_spin.setValue(1)
        self.dj_function = QComboBox(); self.dj_function.addItems(["常数函数 f(x)=0", "平衡函数 f(x)=最低位"])
        self.tele_theta = QDoubleSpinBox(); self.tele_theta.setRange(0.0, math.pi); self.tele_theta.setDecimals(4); self.tele_theta.setSingleStep(0.1); self.tele_theta.setValue(1.2310)
        self.tele_phi = QDoubleSpinBox(); self.tele_phi.setRange(-math.pi, math.pi); self.tele_phi.setDecimals(4); self.tele_phi.setSingleStep(0.1); self.tele_phi.setValue(math.pi/2)
        self.qft_basis = QSpinBox(); self.qft_basis.setRange(0, 63); self.qft_basis.setValue(1)
        self.shots = QSpinBox(); self.shots.setRange(100, 100000); self.shots.setSingleStep(100); self.shots.setValue(2000)
        form.addRow("算法", self.algorithm)
        form.addRow("量子比特数（搜索/DJ/QFT）", self.n_spin)
        form.addRow("Grover 目标态", self.target)
        form.addRow("Grover 迭代轮数（0=自动推荐）", self.iter_spin)
        form.addRow("Deutsch–Jozsa 函数类型", self.dj_function)
        form.addRow("传态输入态 θ（弧度）", self.tele_theta)
        form.addRow("传态输入态 φ（弧度）", self.tele_phi)
        form.addRow("QFT 输入基态编号", self.qft_basis)
        form.addRow("采样次数 shots", self.shots)
        root.addWidget(form_box)
        controls = QHBoxLayout()
        self.run_btn = QPushButton("运行算法实验")
        self.code_btn = QPushButton("生成 PennyLane 代码")
        self.save_btn = QPushButton("保存 .py")
        controls.addWidget(self.run_btn); controls.addWidget(self.code_btn); controls.addWidget(self.save_btn); controls.addStretch(1)
        root.addLayout(controls)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); ll = QVBoxLayout(left); ll.addWidget(QLabel("算法步骤与数学分析"))
        self.analysis = QPlainTextEdit(); self.analysis.setReadOnly(True); ll.addWidget(self.analysis)
        right = QWidget(); rl = QVBoxLayout(right); rl.addWidget(QLabel("数值结果 / 代码预览"))
        self.output = QPlainTextEdit(); self.output.setReadOnly(True); rl.addWidget(self.output)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([520, 520]); root.addWidget(splitter, 1)
        self.algorithm.currentIndexChanged.connect(self._update_controls)
        self.n_spin.valueChanged.connect(self._refresh_targets)
        self.n_spin.valueChanged.connect(lambda n: self.qft_basis.setMaximum(2**n-1))
        self.dj_function.currentIndexChanged.connect(self.run_experiment)
        self.tele_theta.valueChanged.connect(self.run_experiment)
        self.tele_phi.valueChanged.connect(self.run_experiment)
        self.qft_basis.valueChanged.connect(self.run_experiment)
        self.shots.valueChanged.connect(self.run_experiment)
        self.run_btn.clicked.connect(self.run_experiment)
        self.code_btn.clicked.connect(self.generate_code)
        self.save_btn.clicked.connect(self.save_code)
        self._refresh_targets(); self.qft_basis.setMaximum(2**self.n_spin.value()-1); self._update_controls(); self.run_experiment()

    def _refresh_targets(self):
        old = self.target.currentText()
        n = self.n_spin.value(); states = [format(i, f"0{n}b") for i in range(2**n)]
        self.target.blockSignals(True); self.target.clear(); self.target.addItems(states)
        if old in states: self.target.setCurrentText(old)
        else: self.target.setCurrentText(states[-1])
        self.target.blockSignals(False)

    def _update_controls(self, *_):
        algo = self.algorithm.currentIndex()
        # 按算法隐藏无关参数行，避免切换算法后仍显示 Grover 参数。
        form = self.findChild(QFormLayout)
        for widget, visible in ((self.n_spin, algo in (0, 1, 3)),
                                (self.target, algo == 0),
                                (self.iter_spin, algo == 0),
                                (self.dj_function, algo == 1),
                                (self.tele_theta, algo == 2),
                                (self.tele_phi, algo == 2),
                                (self.qft_basis, algo == 3),
                                (self.shots, algo in (0, 1, 3))):
            widget.setVisible(visible)
            label = form.labelForField(widget) if form else None
            if label is not None:
                label.setVisible(visible)
        self.run_experiment()

    @staticmethod
    def _grover(n, target, rounds):
        N = 2**n; state = np.ones(N, dtype=complex) / math.sqrt(N)
        history = [state.copy()]
        idx = int(target, 2)
        theta = math.asin(math.sqrt(1/N))
        recommended = max(0, int(round(math.pi/(4*theta)-0.5)))
        if rounds == 0: rounds = recommended
        for _ in range(rounds):
            state[idx] *= -1  # Oracle：目标态相位翻转
            state = 2 * np.mean(state) - state  # Diffusion：关于平均振幅反射
            history.append(state.copy())
        return state, history, recommended, idx

    @staticmethod
    def _deutsch_jozsa(n, balanced=True):
        N = 2**n
        # 选用可复现的平衡函数 f(x)=最低位；常数函数 f(x)=0。
        values = np.array([(x & 1) if balanced else 0 for x in range(N)], dtype=int)
        state = np.ones(N, dtype=complex) / math.sqrt(N)
        state *= np.where(values == 1, -1, 1)
        # 最后一层 H^n；矩阵元素 (-1)^(x·y)/sqrt(N)
        y = np.arange(N); x = np.arange(N)
        had = np.array([[(-1)**((int(a & b).bit_count()) % 2) for a in x] for b in y], dtype=float)/math.sqrt(N)
        return had @ state, values

    @staticmethod
    def _qft(state):
        N = len(state)
        j = np.arange(N); k = np.arange(N)
        matrix = np.exp(2j*np.pi*np.outer(k, j)/N)/math.sqrt(N)
        return matrix @ state, matrix

    @staticmethod
    def _teleport_branches(input_state):
        """精确模拟三比特传态四个测量分支；basis 顺序为 |q0 q1 q2>。"""
        bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        state = np.kron(input_state, bell)
        # CNOT control q0 -> target q1
        out = np.zeros_like(state)
        for idx, amp in enumerate(state):
            b0, b1, b2 = (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
            if b0: b1 ^= 1
            out[b0*4+b1*2+b2] += amp
        state = out
        # H on q0
        out = state.copy()
        for base in range(4):
            a, b = state[base], state[4+base]
            out[base], out[4+base] = (a+b)/math.sqrt(2), (a-b)/math.sqrt(2)
        state = out
        X = np.array([[0,1],[1,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        branches = []
        for m0 in (0,1):
            for m1 in (0,1):
                bob = np.array([state[m0*4+m1*2], state[m0*4+m1*2+1]], dtype=complex)
                probability = float(np.vdot(bob,bob).real)
                if probability > 1e-15: bob = bob / math.sqrt(probability)
                if m1: bob = X @ bob
                if m0: bob = Z @ bob
                fidelity = float(abs(np.vdot(input_state,bob))**2)
                branches.append((f"{m0}{m1}", probability, bob, fidelity))
        return branches

    def run_experiment(self):
        try:
            algo = self.algorithm.currentIndex(); n = self.n_spin.value(); shots = self.shots.value()
            lines = []; result = []
            if algo == 0:
                target = self.target.currentText() or "0"*n
                state, hist, recommended, idx = self._grover(n, target, self.iter_spin.value())
                rounds = len(hist)-1; probs = np.abs(state)**2
                lines = ["Grover 量子搜索算法", "目标：在 N 个基态中寻找标记态。", f"N=2^n={2**n}，M=1，θ=arcsin(√(M/N))={math.asin(1/math.sqrt(2**n)):.6f} rad", f"推荐迭代次数约为 round(π/(4θ)-1/2)={recommended}。", "每轮由 Oracle 相位标记和 Diffusion 关于平均振幅反射组成。", f"当前执行 {rounds} 轮；目标态概率理论值 sin²((2r+1)θ)={math.sin((2*rounds+1)*math.asin(1/math.sqrt(2**n)))**2:.6f}。", "注意：小规模搜索中迭代次数取整会影响成功率；过度迭代可能降低成功率。"]
                result = [f"目标态 |{target}⟩ 概率：{probs[idx]:.8f}", f"最高概率态：|{format(int(np.argmax(probs)), f'0{n}b')}⟩，概率 {probs.max():.8f}", "各轮目标态概率："] + [f"  第 {i} 轮：{abs(st[idx])**2:.8f}" for i, st in enumerate(hist)]
                # 理论/数值一致性
                result.append(f"状态归一化检查 Σ|α|²={probs.sum():.12f}")
                rng = np.random.default_rng()
                counts = rng.multinomial(shots, probs / probs.sum())
                result.append(f"有限 shots={shots} 采样结果：")
                result.extend([f"  |{format(i, f'0{n}b')}⟩：{int(c)} 次（{c/shots:.4%}）" for i, c in enumerate(counts) if c])
            elif algo == 1:
                balanced = self.dj_function.currentIndex() == 1
                state, oracle = self._deutsch_jozsa(n, balanced=balanced)
                p0 = abs(state[0])**2; probs = abs(state)**2
                function_name = "平衡函数 f(x)=最低位" if balanced else "常数函数 f(x)=0"
                lines = ["Deutsch–Jozsa 算法", "承诺问题：函数只能是常数函数或平衡函数。", f"本次选择：{function_name}；Oracle 对 f(x)=1 的基态施加相位 (-1)^f(x)。", "输入寄存器先制备均匀叠加态，Oracle 相位编码后再施加 H^⊗n。", "理想判据：测得全零串表示常数函数；测得非全零串表示平衡函数。", "经典确定性最坏情况需要 2^(n-1)+1 次查询；量子算法只需 1 次 Oracle 查询（在承诺成立时）。"]
                result = [f"Oracle 真值表：{oracle.tolist()}", f"最终全零态概率：{p0:.10f}", f"非零结果总概率：{1-p0:.10f}", "最终测量概率："] + [f"  |{format(i, f'0{n}b')}⟩ : {p:.8f}" for i,p in enumerate(probs) if p > 1e-10]
                result.append("结论：当前函数为" + ("平衡函数，理想结果不应出现全零串。" if balanced else "常数函数，理想结果应全部为全零串。"))
                counts = np.random.default_rng().multinomial(shots, probs / probs.sum())
                result.append(f"有限 shots={shots} 采样结果：")
                result.extend([f"  |{format(i, f'0{n}b')}⟩：{int(c)} 次（{c/shots:.4%}）" for i, c in enumerate(counts) if c])
            elif algo == 2:
                theta = self.tele_theta.value(); phi = self.tele_phi.value()
                input_state = np.array([math.cos(theta/2), np.exp(1j*phi)*math.sin(theta/2)], dtype=complex)
                branches = self._teleport_branches(input_state)
                lines = ["量子隐形传态（精确数值分支模拟）", f"输入态 |ψ⟩=cos(θ/2)|0⟩+e^(iφ)sin(θ/2)|1⟩，θ={theta:.4f}，φ={phi:.4f}。", "线路：q1-q2 制备 Bell 态；CNOT(q0,q1)；H(q0)；测量 q0、q1。", "分支修正规则：测量 m0m1 后，Bob 执行 X^m1 Z^m0。", "每个分支均先投影、归一化，再施加修正门，并与输入态计算保真度。"]
                result = [f"输入态振幅：α={input_state[0].real:+.6f}{input_state[0].imag:+.6f}i，β={input_state[1].real:+.6f}{input_state[1].imag:+.6f}i", "测量分支："]
                for bits, probability, bob, fidelity in branches:
                    result.append(f"  分支 {bits}：概率={probability:.8f}；Bob态=[{bob[0].real:+.5f}{bob[0].imag:+.5f}i, {bob[1].real:+.5f}{bob[1].imag:+.5f}i]；保真度={fidelity:.10f}")
                result.append(f"四分支概率之和：{sum(b[1] for b in branches):.12f}")
                result.append(f"最差分支保真度：{min(b[3] for b in branches):.10f}")
                result.append("结论：理想情况下四个分支经条件修正后均恢复输入态；当前为精确状态向量分支模拟，不含信道噪声。")
            else:
                N = 2**n
                basis_index = min(self.qft_basis.value(), N-1)
                initial = np.zeros(N, dtype=complex); initial[basis_index] = 1
                final, matrix = self._qft(initial)
                probs = abs(final)**2
                lines = ["量子傅里叶变换 QFT", f"维度 N=2^n={N}。", "定义：QFT|j⟩=(1/√N)Σₖ exp(2πijk/N)|k⟩。", "电路分解使用 Hadamard 门、受控相位旋转门，最后按实现约定执行交换门。", "QFT 是酉变换：理论上保持态向量范数；它改变振幅分布与相位结构。", f"本例输入基态编号 j={basis_index}，输出为各频率分量的复振幅。"]
                result = [f"输入态：|{format(basis_index, f'0{n}b')}⟩", f"输出归一化检查 Σ|α|²={probs.sum():.12f}", "输出振幅与概率："] + [f"  |{format(i, f'0{n}b')}⟩：振幅 {final[i].real:+.5f}{final[i].imag:+.5f}i，概率 {probs[i]:.8f}" for i in range(N)]
                expected = np.exp(2j*np.pi*np.arange(N)*basis_index/N)/math.sqrt(N)
                result.append(f"与独立 QFT 公式的最大振幅误差：{np.max(np.abs(final-expected)):.3e}")
                counts = np.random.default_rng().multinomial(shots, probs / probs.sum())
                result.append(f"有限 shots={shots} 采样结果：")
                result.extend([f"  |{format(i, f'0{n}b')}⟩：{int(c)} 次（{c/shots:.4%}）" for i, c in enumerate(counts) if c])
            self.analysis.setPlainText("\n".join(lines))
            self.output.setPlainText("\n".join(result) + f"\n\nshots 设置：{shots}。理论概率由数值状态向量计算，采样频率通过多项分布模拟有限测量次数。")
        except Exception as exc:
            QMessageBox.critical(self, "算法实验失败", str(exc))

    def generate_code(self):
        algo = self.algorithm.currentIndex(); n = self.n_spin.value(); target = self.target.currentText(); rounds = self.iter_spin.value(); shots = self.shots.value()
        header = '''"""由 Quantum Circuit Studio 算法实验室生成。\n运行前安装：pip install pennylane numpy matplotlib\n"""\nimport pennylane as qml\nimport numpy as np\n'''
        if algo == 0:
            theta = math.asin(1/math.sqrt(2**n)); auto = max(0, round(math.pi/(4*theta)-0.5))
            r = auto if rounds == 0 else rounds
            body = f'''\nn = {n}\ntarget = "{target}"\nrounds = {r}\nshots = {shots}\ndev = qml.device("default.qubit", wires=n, shots=shots)\n\ndef oracle():\n    # 对目标计算基态施加 -1 相位：X 将目标映射到全 1，再以多控 Z 标记\n    for wire, bit in enumerate(target):\n        if bit == "0": qml.PauliX(wires=wire)\n    if n == 1: qml.PauliZ(wires=0)\n    else: qml.ctrl(qml.PauliX, control=list(range(n-1)))(wires=n-1)\n    for wire, bit in enumerate(target):\n        if bit == "0": qml.PauliX(wires=wire)\n\ndef diffusion():\n    for w in range(n): qml.Hadamard(wires=w)\n    for w in range(n): qml.PauliX(wires=w)\n    if n == 1: qml.PauliZ(wires=0)\n    else: qml.ctrl(qml.PauliX, control=list(range(n-1)))(wires=n-1)\n    for w in range(n): qml.PauliX(wires=w)\n    for w in range(n): qml.Hadamard(wires=w)\n\n@qml.qnode(dev)\ndef circuit():\n    for w in range(n): qml.Hadamard(wires=w)\n    for _ in range(rounds):\n        oracle(); diffusion()\n    return qml.sample(wires=range(n))\n\nsamples = circuit()\nstrings = ["".join(str(int(bit)) for bit in row) for row in np.atleast_2d(samples)]\nfrom collections import Counter\ncounts = Counter(strings)\nprint("counts:", counts)\nprint("target frequency:", counts.get(target, 0) / shots)\n'''
        elif algo == 1:
            body = f'''\nn = {n}\nshots = {shots}\ndev = qml.device("default.qubit", wires=n, shots=shots)\n\n@qml.qnode(dev)\ndef circuit():\n    # Deutsch–Jozsa 平衡 Oracle f(x)=最低位：相位翻转等价于 Z 作用在最低位\n    for w in range(n): qml.Hadamard(wires=w)\n    qml.PauliZ(wires=n-1)\n    for w in range(n): qml.Hadamard(wires=w)\n    return qml.sample(wires=range(n))\n\nsamples = circuit()\nstrings = ["".join(str(int(bit)) for bit in row) for row in np.atleast_2d(samples)]\nprint("measurement counts:", __import__("collections").Counter(strings))\nprint("Interpretation: all-zero means constant; nonzero means balanced (promise problem).")\n'''
        elif algo == 2:
            theta_in = self.tele_theta.value(); phi_in = self.tele_phi.value()
            body = f"""
shots = {shots}
theta = {theta_in!r}
phi = {phi_in!r}
alpha = np.cos(theta / 2)
beta = np.exp(1j * phi) * np.sin(theta / 2)
dev = qml.device("default.qubit", wires=3, shots=shots)

@qml.qnode(dev)
def teleportation():
    qml.StatePrep(np.array([alpha, beta]), wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)
    m0 = qml.measure(0)
    m1 = qml.measure(1)
    qml.cond(m1, qml.PauliX)(wires=2)
    qml.cond(m0, qml.PauliZ)(wires=2)
    return qml.probs(wires=[2])

print("Input state [alpha,beta]:", np.array([alpha, beta]))
print("Bob output probabilities after feed-forward correction:", teleportation())
"""
        else:
            basis_index = min(self.qft_basis.value(), 2**n-1)
            bitstring = ', '.join('1' if (basis_index >> (n-1-i)) & 1 else '0' for i in range(n))
            body = f"""
n = {n}
basis_index = {basis_index}
shots = {shots}
dev = qml.device("default.qubit", wires=n, shots=None)

@qml.qnode(dev)
def qft_circuit():
    qml.BasisState(np.array([{bitstring}], dtype=int), wires=range(n))
    qml.QFT(wires=range(n))
    return qml.state()

state = qft_circuit()
probs = np.abs(state)**2
expected = np.exp(2j*np.pi*np.arange(2**n)*basis_index/(2**n))/np.sqrt(2**n)
print("QFT state:", state)
print("probabilities:", probs)
print("norm:", np.sum(probs))
print("max error vs analytical QFT:", np.max(np.abs(state-expected)))
"""
        self.last_code = header + body
        self.output.setPlainText(self.last_code)

    def save_code(self):
        if not self.last_code:
            self.generate_code()
        path, _ = QFileDialog.getSaveFileName(self, "保存 PennyLane 代码", "quantum_algorithm.py", "Python 文件 (*.py)")
        if path:
            try:
                Path(path).write_text(self.last_code, encoding="utf-8")
                QMessageBox.information(self, "保存成功", f"代码已保存到：\n{path}")
            except OSError as exc:
                QMessageBox.critical(self, "保存失败", str(exc))
