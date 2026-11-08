"""pytest 公共配置。

所有测试都在无界面模式下运行：Qt 使用 offscreen 平台，
因此 CI 或无显示器的服务器上也能执行。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


@pytest.fixture(scope="session")
def qapp():
    """整个测试会话共用一个 QApplication。"""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch, qapp):
    """把模态对话框替换成记录器。

    offscreen 平台没有人能点"确定"，真正的 QMessageBox.exec() 会永久阻塞。
    """
    from PySide6.QtWidgets import QMessageBox

    recorded = []

    def fake(kind):
        def _inner(*args, **kwargs):
            recorded.append((kind, args[1] if len(args) > 1 else ""))
            return QMessageBox.StandardButton.Ok

        return _inner

    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(fake(kind)))

    yield recorded


@pytest.fixture()
def window(qapp, no_modal_dialogs):
    """构造一个真实主窗口；测试结束后销毁，避免状态串扰。"""
    from PySide6.QtCore import Qt
    from quantum_circuit_studio.main_window import MainWindow

    win = MainWindow()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qapp.processEvents()
