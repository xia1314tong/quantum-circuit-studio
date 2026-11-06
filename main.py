"""Quantum Circuit Studio 启动入口。"""
import sys
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from quantum_circuit_studio.common import get_app_icon_path
from quantum_circuit_studio.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    icon_path = get_app_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
