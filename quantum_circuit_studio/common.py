import sys

import os

import json

import copy

import traceback

import tempfile

import csv

from datetime import datetime

from pathlib import Path

import numpy as np

import pennylane as qml

from PySide6.QtCore import Qt, QMimeData, QSize, QPoint, QThread, Signal, Slot, QTimer, QObject

from PySide6.QtGui import (
    QDrag, QPainter, QPen, QColor, QBrush, QFont,
    QKeySequence, QShortcut, QPixmap, QPolygon, QIcon
)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QDialog, QLineEdit, QPlainTextEdit,
    QListWidgetItem, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSpinBox, QFileDialog, QMessageBox,
    QGroupBox, QSplitter, QScrollArea, QDoubleSpinBox, QSizePolicy,
    QCheckBox, QComboBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from matplotlib.figure import Figure

import matplotlib

def get_app_icon_path():
    """查找应用图标；兼容源码运行及 PyInstaller onefile/onedir。"""
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
        bundle_dir = getattr(sys, "_MEIPASS", None)
        if bundle_dir:
            candidates.append(Path(bundle_dir))
    else:
        # common.py 位于 quantum_circuit_studio 子目录，图标放在项目根目录。
        candidates.append(Path(__file__).resolve().parent.parent)
        candidates.append(Path(__file__).resolve().parent)

    filenames = ("app_icon.ico", "quantum_circuit_app_icon.ico", "量子电路仿真平台.ico")
    for base_dir in candidates:
        for filename in filenames:
            candidate = base_dir / filename
            if candidate.is_file():
                return candidate
    return None


def create_spinbox_arrow_icons():
    """生成黑色三角形 PNG，确保不同 Windows/Qt 主题下箭头都清晰可见。"""
    icon_dir = Path(tempfile.gettempdir()) / "quantum_circuit_studio_icons"
    icon_dir.mkdir(parents=True, exist_ok=True)

    shapes = {
        "up": [(4, 8), (8, 2), (12, 8)],
        "down": [(4, 3), (12, 3), (8, 9)],
    }

    paths = {}
    for name, points in shapes.items():
        pixmap = QPixmap(16, 12)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#000000"))
        painter.drawPolygon(
            QPolygon([QPoint(x, y) for x, y in points])
        )
        painter.end()

        icon_path = icon_dir / f"spinbox_{name}.png"
        if not pixmap.save(str(icon_path), "PNG"):
            raise RuntimeError(f"无法创建数字输入框箭头图标：{icon_path}")
        paths[name] = icon_path.as_posix()

    return paths["up"], paths["down"]

MIME_TYPE = "application/x-quantum-gate"

GATE_NAMES = ["H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT"]

ROTATION_GATES = {"RX", "RY", "RZ"}

MAX_HISTORY = 100

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun"
]
matplotlib.rcParams["axes.unicode_minus"] = False
