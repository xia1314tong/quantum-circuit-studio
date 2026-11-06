"""主窗口组合层。界面和业务职责由各 Mixin 分担。"""
from .common import (
    Path, QIcon, QMainWindow, create_spinbox_arrow_icons, get_app_icon_path, json,
    os, sys
)
from .widgets import (
    CircuitCanvas, GatePalette, DeepSeekWorker, DeepSeekResultReceiver, SafeAssistantDialog
)
from .mixins.assistant import AssistantMixin
from .mixins.circuit import CircuitMixin
from .mixins.exports import ExportMixin
from .mixins.history import HistoryMixin
from .mixins.simulation import SimulationMixin
from .mixins.ui import UIMixin


class MainWindow(AssistantMixin, HistoryMixin, UIMixin, CircuitMixin, SimulationMixin, ExportMixin, QMainWindow):
    def __init__(self):
        super().__init__()

        icon_path = get_app_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

        self.setWindowTitle(
            "Quantum Circuit Studio - 量子电路仿真平台"
        )

        self.resize(1250, 1000)

        # 撤销和重做栈
        self.undo_stack = []
        self.redo_stack = []

        # DeepSeek AI 配置：密钥从程序旁边的 deepseek_config.json 或环境变量读取，界面不再要求输入。
        # 注意：配置文件含 API Key 时属于机密，不要分享/上传该文件或将其打包分发。
        config_candidates = []
        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            config_candidates.append(exe_dir / "deepseek_config.json")
            # 若打包时把配置一并封进了数据目录，也能直接读到，实现免配置分发
            bundle_dir = getattr(sys, "_MEIPASS", None)
            if bundle_dir:
                config_candidates.append(Path(bundle_dir) / "deepseek_config.json")
        else:
            config_candidates.append(
                Path(__file__).resolve().parent.parent / "deepseek_config.json"
            )
        self.deepseek_config_path = config_candidates[0]
        self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.deepseek_model = "deepseek-flash"
        for candidate in config_candidates:
            if not candidate.exists():
                continue
            self.deepseek_config_path = candidate
            try:
                with candidate.open("r", encoding="utf-8") as config_file:
                    saved_config = json.load(config_file)
                if isinstance(saved_config, dict):
                    saved_key = str(saved_config.get("api_key", "")).strip()
                    if saved_key:
                        self.deepseek_api_key = saved_key
                    saved_model = str(saved_config.get("model", "")).strip()
                    if saved_model:
                        self.deepseek_model = saved_model
                break
            except Exception as config_error:
                print(f"读取 DeepSeek 配置失败：{config_error}")
        self.ai_pending_circuit = None

        # 最近一次仿真结果，供 CSV、图表和实验报告导出使用
        self.last_results = None

        self.build_ui()
        self.setup_shortcuts()

        # 生成黑色三角形箭头图标，避免 Qt 样式表边框绘制方式在部分主题中失效
        up_arrow_path, down_arrow_path = create_spinbox_arrow_icons()

        # 统一界面样式：采用参考界面的蓝白灰配色，保留当前控件与布局形态
        style_sheet = """
            QMainWindow, QWidget {
                background-color: #F8FAFC;
                color: #344054;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
                font-size: 9pt;
            }

            QLabel {
                background: transparent;
                color: #344054;
                font-size: 9pt;
            }

            QGroupBox {
                background-color: #FFFFFF;
                color: #344054;
                border: 1px solid #E4E7EC;
                border-radius: 9px;
                margin-top: 13px;
                padding: 12px 8px 8px 8px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                color: #101828;
                background-color: #F8FAFC;
            }

            QPushButton {
                background-color: #2563EB;
                color: #FFFFFF;
                border: 1px solid #2563EB;
                border-radius: 6px;
                padding: 7px 12px;
                min-height: 22px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #1D4ED8;
                border-color: #1D4ED8;
            }

            QPushButton:pressed {
                background-color: #1E40AF;
                border-color: #1E40AF;
                padding-top: 8px;
                padding-bottom: 6px;
            }

            QPushButton:disabled {
                background-color: #EAECF0;
                color: #98A2B3;
                border-color: #EAECF0;
            }

            QSpinBox, QDoubleSpinBox {
                background-color: #FFFFFF;
                color: #101828;
                border: 1px solid #D0D5DD;
                border-radius: 6px;
                padding: 4px;
                padding-right: 29px;
                min-height: 28px;
                font-size: 10pt;
                selection-background-color: #2563EB;
                selection-color: #FFFFFF;
            }

            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #2563EB;
            }

            /* 保留目前的上下三角形样式，箭头改为黑色 */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                height: 18px;
                margin-right: 1px;
                margin-bottom: 1px;
                background-color: #F2F4F7;
                border-left: 1px solid #D0D5DD;
                border-top-right-radius: 5px;
            }

            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                height: 18px;
                margin-right: 1px;
                margin-top: 1px;
                background-color: #F2F4F7;
                border-left: 1px solid #D0D5DD;
                border-bottom-right-radius: 5px;
            }

            /* 使用真实的黑色三角形 PNG，保证箭头形状明确可见 */
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url("__UP_ARROW_PATH__");
                width: 10px;
                height: 8px;
            }

            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url("__DOWN_ARROW_PATH__");
                width: 10px;
                height: 8px;
            }

            /* 鼠标悬停：浅蓝色 */
            QSpinBox::up-button:hover,
            QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover {
                background-color: #DBEAFE;
                border: 1px solid #60A5FA;
            }

            /* 鼠标按下：明显变成深蓝色 */
            QSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed,
            QDoubleSpinBox::up-button:pressed,
            QDoubleSpinBox::down-button:pressed {
                background-color: #60A5FA;
                border: 1px solid #2563EB;
            }

            QListWidget {
                background-color: #FFFFFF;
                color: #344054;
                border: 1px solid #E4E7EC;
                border-radius: 7px;
                padding: 5px;
                outline: none;
            }

            QListWidget::item {
                background-color: #DBEAFE;
                color: #1D4ED8;
                border: 1px solid transparent;
                border-radius: 5px;
                padding: 6px;
                margin: 2px;
                font-weight: 600;
            }

            QListWidget::item:hover {
                background-color: #BFDBFE;
                border-color: #93C5FD;
            }

            QListWidget::item:selected {
                background-color: #2563EB;
                color: #FFFFFF;
                border-color: #1D4ED8;
            }

            QScrollArea {
                background-color: #FFFFFF;
                border: 1px solid #EAECF0;
                border-radius: 6px;
            }

            QScrollBar:vertical {
                background: #F2F4F7;
                width: 10px;
                margin: 2px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: #C7D7FE;
                min-height: 24px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background: #84ADFF;
            }

            QScrollBar:horizontal {
                background: #F2F4F7;
                height: 10px;
                margin: 2px;
                border-radius: 5px;
            }

            QScrollBar::handle:horizontal {
                background: #C7D7FE;
                min-width: 24px;
                border-radius: 5px;
            }

            QSplitter::handle {
                background-color: #EAECF0;
                border-radius: 3px;
            }

            QSplitter::handle:hover {
                background-color: #84ADFF;
            }

            QStatusBar {
                background-color: #FFFFFF;
                color: #667085;
                border-top: 1px solid #EAECF0;
            }

            QMenuBar {
                background-color: #FFFFFF;
                color: #344054;
                border-bottom: 1px solid #EAECF0;
            }

            QMenuBar::item {
                padding: 6px 10px;
                background: transparent;
            }

            QMenuBar::item:selected {
                background-color: #DBEAFE;
                color: #1D4ED8;
                border-radius: 4px;
            }

            QMenu {
                background-color: #FFFFFF;
                color: #344054;
                border: 1px solid #D0D5DD;
                padding: 4px;
            }

            QMenu::item {
                padding: 6px 28px 6px 18px;
                border-radius: 4px;
            }

            QMenu::item:selected {
                background-color: #DBEAFE;
                color: #1D4ED8;
            }
        """

        # Windows 路径统一使用正斜杠，交给 Qt 样式表读取本地图标
        style_sheet = style_sheet.replace("__UP_ARROW_PATH__", up_arrow_path)
        style_sheet = style_sheet.replace("__DOWN_ARROW_PATH__", down_arrow_path)
        self.setStyleSheet(style_sheet)
