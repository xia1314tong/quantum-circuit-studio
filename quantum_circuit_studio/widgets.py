"""可复用的 Qt 控件：量子门工具箱、线路画布与 AI 请求线程。"""
from .common import (
    GATE_NAMES, MIME_TYPE, QBrush, QColor, QDialog, QDrag,
    QFont, QListWidget, QListWidgetItem, QMessageBox, QMimeData, QObject,
    QPainter, QPen, QSize, QThread, QWidget, Qt,
    ROTATION_GATES, Signal, Slot, copy, np
)


class DeepSeekWorker(QThread):
    """在后台线程执行网络请求，避免 GUI 卡住，令等待动画持续运行。"""
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, task, parent=None):
        super().__init__(parent)
        self.task = task

    def run(self):
        try:
            self.result_ready.emit(self.task())
        except Exception as exc:
            self.failed.emit(str(exc))


class DeepSeekResultReceiver(QObject):
    """将工作线程的结果切回 GUI 线程处理。"""
    def __init__(self, on_result, on_error, parent=None):
        super().__init__(parent)
        self.on_result_callback = on_result
        self.on_error_callback = on_error

    @Slot(object)
    def receive_result(self, result):
        self.on_result_callback(result)

    @Slot(str)
    def receive_error(self, message):
        self.on_error_callback(message)


class SafeAssistantDialog(QDialog):
    """避免用户在请求进行中关闭窗口，导致后台线程被销毁。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.busy_check = None

    def _is_busy(self):
        return callable(self.busy_check) and bool(self.busy_check())

    def closeEvent(self, event):
        if self._is_busy():
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        if self._is_busy():
            return
        super().reject()


class GatePalette(QListWidget):

    def __init__(self):
        super().__init__()

        self.setDragEnabled(True)
        self.setSpacing(4)

        for name in GATE_NAMES:
            item = QListWidgetItem(name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(QSize(100, 38))
            item.setBackground(QColor("#e3edff"))
            self.addItem(item)

    def startDrag(self, supported_actions):

        item = self.currentItem()

        if item is None:
            return

        mime = QMimeData()
        mime.setData(MIME_TYPE, item.text().encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)

        drag.exec(Qt.DropAction.CopyAction)


class CircuitCanvas(QWidget):
    LEFT = 75
    TOP = 60

    WIRE_GAP = 75
    COL_GAP = 75

    GATE_W = 46
    GATE_H = 40

    def __init__(
            self,
            status_callback,
            selection_callback,
            history_callback
    ):
        super().__init__()

        self.status_callback = status_callback
        self.selection_callback = selection_callback
        self.history_callback = history_callback

        self.num_wires = 2
        self.gates = []

        self.selected = None
        self.dragging = None
        self.drag_anchor = None

        self.drag_start_snapshot = None
        self.drag_history_recorded = False

        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(760, 300)

    def wire_y(self, wire):
        return self.TOP + wire * self.WIRE_GAP

    def col_x(self, col):
        return self.LEFT + 45 + col * self.COL_GAP

    def snapshot(self):
        """保存当前电路状态，用于撤销和重做。"""
        return {
            "num_wires": self.num_wires,
            "gates": copy.deepcopy(self.gates)
        }

    def snap(self, point):

        wire = round(
            (point.y() - self.TOP) / self.WIRE_GAP
        )

        wire = max(0, min(self.num_wires - 1, wire))

        col = round(
            (point.x() - (self.LEFT + 45)) / self.COL_GAP
        )

        return wire, max(0, col)

    def gate_rect(self, gate):

        x = self.col_x(gate["col"])

        if gate["name"] == "CNOT":
            y = (
                        self.wire_y(gate["control"])
                        + self.wire_y(gate["target"])
                ) / 2
        else:
            y = self.wire_y(gate["wire"])

        return (
            x - self.GATE_W / 2,
            y - self.GATE_H / 2,
            self.GATE_W,
            self.GATE_H
        )

    def find_gate(self, point):

        for i in range(len(self.gates) - 1, -1, -1):

            gate = self.gates[i]

            x, y, w, h = self.gate_rect(gate)

            if (
                    x <= point.x() <= x + w
                    and y <= point.y() <= y + h
            ):
                return i

            if gate["name"] == "CNOT":

                x0 = self.col_x(gate["col"])

                for wire in (
                        gate["control"],
                        gate["target"]
                ):
                    if (
                            abs(point.x() - x0) < 16
                            and abs(
                        point.y() - self.wire_y(wire)
                    ) < 16
                    ):
                        return i

        return None

    # ------------------------------
    # 添加量子门
    # ------------------------------
    def add_gate(self, name, point):

        if name == "CNOT" and self.num_wires < 2:
            QMessageBox.warning(
                self,
                "提示",
                "CNOT 门至少需要两个量子比特。"
            )
            return

        wire, col = self.snap(point)

        if name == "CNOT":

            target = (
                wire + 1
                if wire + 1 < self.num_wires
                else wire - 1
            )

            gate = {
                "name": name,
                "control": wire,
                "target": target,
                "col": col
            }

        else:

            gate = {
                "name": name,
                "wire": wire,
                "col": col,
                "param": float(np.pi / 2)
            }

        self.history_callback()

        self.gates.append(gate)

        self.select_gate(len(self.gates) - 1)

        self.adjust_canvas()
        self.update()

        self.status_callback(f"已添加 {name} 门")

    def select_gate(self, index):

        self.selected = index

        if index is None:
            self.selection_callback(None)
        else:
            self.selection_callback(self.gates[index])

    def adjust_canvas(self):

        max_col = max(
            (gate["col"] for gate in self.gates),
            default=0
        )

        self.setMinimumWidth(
            max(
                760,
                self.LEFT + 120
                + (max_col + 1) * self.COL_GAP
            )
        )

        self.setMinimumHeight(
            max(
                300,
                self.TOP
                + self.num_wires * self.WIRE_GAP
                + 70
            )
        )

    # 绘制线路与量子门
    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        painter.fillRect(self.rect(), QColor("white"))

        painter.setPen(QColor("#243247"))
        painter.setFont(
            QFont("Arial", 11, QFont.Weight.Bold)
        )

        painter.drawText(15, 25, "量子线路画布")

        for wire in range(self.num_wires):
            y = self.wire_y(wire)

            painter.setPen(
                QPen(QColor("#64748b"), 2)
            )

            painter.drawLine(
                self.LEFT,
                y,
                max(
                    self.width() - 20,
                    self.col_x(0) + 30
                ),
                y
            )

            painter.setPen(QColor("#1e293b"))
            painter.drawText(18, y + 5, f"q{wire}")

        ordered = sorted(
            enumerate(self.gates),
            key=lambda item: (
                item[1]["col"],
                item[0]
            )
        )

        for index, gate in ordered:

            x = self.col_x(gate["col"])
            selected = index == self.selected

            color = (
                QColor("#2563eb")
                if selected
                else QColor("#334155")
            )

            fill = (
                QColor("#dbeafe")
                if selected
                else QColor("#f1f5f9")
            )

            painter.setPen(
                QPen(color, 2 if selected else 1)
            )

            painter.setBrush(QBrush(fill))

            if gate["name"] == "CNOT":

                yc = self.wire_y(gate["control"])
                yt = self.wire_y(gate["target"])

                painter.drawLine(
                    int(x), int(yc),
                    int(x), int(yt)
                )

                painter.setBrush(QBrush(color))

                painter.drawEllipse(
                    int(x - 6), int(yc - 6),
                    12, 12
                )

                painter.setBrush(
                    QBrush(QColor("white"))
                )

                painter.drawEllipse(
                    int(x - 15), int(yt - 15),
                    30, 30
                )

                painter.drawLine(
                    int(x - 10), int(yt),
                    int(x + 10), int(yt)
                )

                painter.drawLine(
                    int(x), int(yt - 10),
                    int(x), int(yt + 10)
                )

            else:

                y = self.wire_y(gate["wire"])

                rx = int(x - self.GATE_W / 2)
                ry = int(y - self.GATE_H / 2)

                painter.drawRoundedRect(
                    rx, ry,
                    self.GATE_W, self.GATE_H,
                    7, 7
                )

                painter.setPen(QColor("#172033"))

                painter.setFont(
                    QFont("Arial", 10, QFont.Weight.Bold)
                )

                label = gate["name"]

                if label in ROTATION_GATES:
                    label += "*"

                painter.drawText(
                    rx, ry,
                    self.GATE_W, self.GATE_H,
                    Qt.AlignmentFlag.AlignCenter,
                    label
                )

        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Arial", 8))

        painter.drawText(
            15,
            self.height() - 10,
            "拖入门添加；拖动已有门移动；"
            "按 Delete 删除；旋转门带 *"
        )

        painter.end()

    # 拖入量子门
    def dragEnterEvent(self, event):

        if event.mimeData().hasFormat(MIME_TYPE):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):

        if event.mimeData().hasFormat(MIME_TYPE):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):

        if not event.mimeData().hasFormat(MIME_TYPE):
            event.ignore()
            return

        name = bytes(
            event.mimeData().data(MIME_TYPE)
        ).decode("utf-8")

        self.add_gate(
            name,
            event.position().toPoint()
        )

        event.acceptProposedAction()

    # 选择量子门
    def mousePressEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:

            self.setFocus()

            index = self.find_gate(
                event.position().toPoint()
            )

            self.select_gate(index)
            self.dragging = index

            self.drag_history_recorded = False

            if index is not None:
                # 记录拖动开始前的状态，但暂不写入撤销栈
                self.drag_start_snapshot = self.snapshot()
                # 记下抓住的是 CNOT 的哪一端，拖动时据此保持连接
                self.drag_anchor = self.cnot_grab_anchor(
                    self.gates[index],
                    event.position().toPoint(),
                )
            else:
                self.drag_start_snapshot = None
                self.drag_anchor = None

            self.update()

            if index is not None:
                event.accept()
                return

        super().mousePressEvent(event)

    def cnot_grab_anchor(self, gate, point):
        """判断鼠标是按在 CNOT 的控制端还是目标端。

        拖动时必须知道"抓住的是哪一头"，否则从目标端横拖会被当成重新指定控制位，
        悄悄把 CNOT(1,2) 变成 CNOT(2,3)。
        """
        if gate.get("name") != "CNOT":
            return None
        control = int(gate["control"])
        target = int(gate["target"])
        distance_control = abs(point.y() - self.wire_y(control))
        distance_target = abs(point.y() - self.wire_y(target))
        return "target" if distance_target < distance_control else "control"

    def cnot_drag_endpoints(self, gate, wire, anchor="control"):
        """拖动 CNOT 时保持两个端点的相对位置。

        ``anchor`` 表示抓住的一端：抓住控制端时它跟随鼠标、目标端按相对偏移移动；
        抓住目标端时反过来。横向拖动（wire 不变）原样返回，连接不变；
        纵向拖动后任一端点越界则拒绝，避免悄悄换掉比特。
        """
        control = int(gate["control"])
        target = int(gate["target"])
        relative = target - control
        wire = int(wire)
        if wire == control and anchor == "control":
            return control, target
        if wire == target and anchor == "target":
            return control, target
        if anchor == "target":
            new_target, new_control = wire, wire - relative
        else:
            new_control, new_target = wire, wire + relative
        if 0 <= new_control < self.num_wires and 0 <= new_target < self.num_wires:
            if new_control != new_target:
                return new_control, new_target
        return control, target

    # 移动量子门
    def mouseMoveEvent(self, event):

        if (
                self.dragging is not None
                and event.buttons() & Qt.MouseButton.LeftButton
        ):

            wire, col = self.snap(
                event.position().toPoint()
            )

            gate = self.gates[self.dragging]

            # 先计算新的位置：CNOT 整体平移，横移只改列、不改连接
            if gate["name"] == "CNOT":

                control, target = self.cnot_drag_endpoints(
                    gate, wire, anchor=self.drag_anchor or "control"
                )

                changed = (
                        gate["col"] != col
                        or gate["control"] != control
                        or gate["target"] != target
                )

            else:

                changed = (
                        gate["col"] != col
                        or gate["wire"] != wire
                )

            # 实际发生移动时才记录一次历史
            if (
                    changed
                    and not self.drag_history_recorded
            ):
                self.history_callback(
                    self.drag_start_snapshot
                )

                self.drag_history_recorded = True

            gate["col"] = col

            if gate["name"] == "CNOT":

                gate["control"] = control
                gate["target"] = target

            else:

                gate["wire"] = wire

            self.adjust_canvas()
            self.update()

            self.selection_callback(gate)

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):

        self.dragging = None
        self.drag_start_snapshot = None
        self.drag_history_recorded = False
        self.drag_anchor = None

        super().mouseReleaseEvent(event)

    # ------------------------------
    # 删除量子门
    # ------------------------------
    def keyPressEvent(self, event):

        if (
                event.key() in (
                Qt.Key.Key_Delete,
                Qt.Key.Key_Backspace
        )
                and self.selected is not None
        ):
            self.history_callback()

            gate = self.gates.pop(self.selected)

            self.select_gate(None)

            self.adjust_canvas()
            self.update()

            self.status_callback(
                f"已删除 {gate['name']} 门"
            )

            event.accept()
            return

        super().keyPressEvent(event)

    # ------------------------------
    # 修改量子比特数量
    # ------------------------------
    def set_wire_count(self, count):

        if count == self.num_wires:
            return

        self.history_callback()

        self.num_wires = count

        kept = []

        for gate in self.gates:

            if gate["name"] == "CNOT":

                if count >= 2:

                    control = max(
                        0,
                        min(count - 1, gate["control"])
                    )

                    target = max(
                        0,
                        min(count - 1, gate["target"])
                    )

                    if control == target:
                        target = (
                            control + 1
                            if control + 1 < count
                            else control - 1
                        )

                    gate["control"] = control
                    gate["target"] = target

                    kept.append(gate)

            elif gate["wire"] < count:

                kept.append(gate)

        self.gates = kept

        self.select_gate(None)

        self.adjust_canvas()
        self.update()
