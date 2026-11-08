"""撤销重做与快捷键。"""
from ..common import (
    MAX_HISTORY, QKeySequence, QShortcut, Qt, copy
)


class HistoryMixin:
    def make_snapshot(self):

        return {
            "num_wires": self.canvas.num_wires,
            "gates": copy.deepcopy(self.canvas.gates)
        }

    def record_history(self, snapshot=None):

        if snapshot is None:
            snapshot = self.make_snapshot()

        self.undo_stack.append(copy.deepcopy(snapshot))

        if len(self.undo_stack) > MAX_HISTORY:
            self.undo_stack.pop(0)

        # 有新操作后，清除旧的重做记录
        self.redo_stack.clear()

        self.update_history_buttons()

    def restore_snapshot(self, snapshot):

        count = int(snapshot["num_wires"])

        # 避免恢复时触发新的线路修改事件
        self.wire_spin.blockSignals(True)
        self.wire_spin.setValue(count)
        self.wire_spin.blockSignals(False)

        self.canvas.num_wires = count
        self.canvas.gates = copy.deepcopy(
            snapshot["gates"]
        )

        self.canvas.select_gate(None)
        self.canvas.adjust_canvas()
        self.canvas.update()

    def undo(self):

        if not self.undo_stack:
            self.set_status("没有可撤销的操作")
            return

        # 当前状态放入重做栈
        self.redo_stack.append(self.make_snapshot())

        previous = self.undo_stack.pop()

        self.restore_snapshot(previous)

        self.update_history_buttons()
        self.set_status("已撤销上一步操作")

    def redo(self):

        if not self.redo_stack:
            self.set_status("没有可重做的操作")
            return

        # 当前状态放入撤销栈
        self.undo_stack.append(self.make_snapshot())

        if len(self.undo_stack) > MAX_HISTORY:
            self.undo_stack.pop(0)

        next_state = self.redo_stack.pop()

        self.restore_snapshot(next_state)

        self.update_history_buttons()
        self.set_status("已重做操作")

    def setup_shortcuts(self):

        self.shortcut_undo = QShortcut(
            QKeySequence("Ctrl+Z"),
            self
        )

        self.shortcut_undo.setContext(
            Qt.ShortcutContext.WindowShortcut
        )

        self.shortcut_undo.activated.connect(self.undo)

        self.shortcut_redo = QShortcut(
            QKeySequence("Ctrl+Y"),
            self
        )

        self.shortcut_redo.setContext(
            Qt.ShortcutContext.WindowShortcut
        )

        self.shortcut_redo.activated.connect(self.redo)

        # 常见的另一种重做快捷键
        self.shortcut_redo2 = QShortcut(
            QKeySequence("Ctrl+Shift+Z"),
            self
        )

        self.shortcut_redo2.setContext(
            Qt.ShortcutContext.WindowShortcut
        )

        self.shortcut_redo2.activated.connect(self.redo)

    def update_history_buttons(self):

        if not hasattr(self, "undo_button"):
            return

        self.undo_button.setEnabled(
            len(self.undo_stack) > 0
        )

        self.redo_button.setEnabled(
            len(self.redo_stack) > 0
        )
