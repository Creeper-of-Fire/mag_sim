# gui/main_window.py

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QLabel, QPushButton, QListWidget,
    QTextEdit, QSplitter, QGroupBox, QListWidgetItem, QMessageBox,
    QFileDialog, QFrame, QScrollArea
)

from gui.app_config import DATA_DIR
from utils.project_config import (
    PROJECT_ROOT, FILENAME_HISTORY, get_conda_activation_command,
    PROJECT_ROOT_WSL, FILENAME_TASKS_CSV, STATUS_FAILED, STATUS_COMPLETED,
    STATUS_PENDING, get_wsl_path, COLUMN_TASK_NAME
)

# 常量定义
GLOBAL_STATE_FILE = os.path.join(DATA_DIR, "gui_state.json")
JOB_CONFIG_NAME = "job_config.json"


class SimulationControllerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("等离子体模拟任务管理器 (执行驱动)")
        self.setGeometry(100, 100, 1200, 800)

        # 状态变量
        self.current_job_dir = None
        self.tasks_from_csv = []
        self.history_hashes = set()
        self.batch_process = None
        self.wsl_pid = None
        self.pid_capture_buffer = ""

        self.setup_ui()
        self.load_global_state()  # 自动恢复上次的目录
        self.apply_stylesheet()

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QListWidget::item:selected { background-color: #0078d7; color: white; }
            QLineEdit[readOnly="true"] { background-color: #f0f0f0; color: #555; }
            QGroupBox { font-weight: bold; }
        """)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- 顶部：工作目录选择 ---
        dir_group = QGroupBox("项目目录控制")
        dir_group.setMaximumHeight(70)
        dir_layout = QHBoxLayout()
        self.job_dir_entry = QLineEdit()
        self.job_dir_entry.setReadOnly(True)
        btn_select_dir = QPushButton("选择项目目录...")
        btn_select_dir.clicked.connect(self.select_job_dir)
        self.btn_open_folder = QPushButton("打开文件夹")
        self.btn_open_folder.clicked.connect(self.open_current_job_dir)
        self.btn_open_folder.setEnabled(False)

        dir_layout.addWidget(QLabel("当前目录:"))
        dir_layout.addWidget(self.job_dir_entry)
        dir_layout.addWidget(btn_select_dir)
        dir_layout.addWidget(self.btn_open_folder)
        dir_group.setLayout(dir_layout)
        main_layout.addWidget(dir_group)

        # --- 中间：主分割面板 ---
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧：任务管理与设置
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        # 1. 脚本与参数配置 (Job-specific)
        config_group = QGroupBox("任务运行配置 (保存在项目内)")
        config_form = QFormLayout()
        self.script_name_entry = QLineEdit("csv_tool_constant_energy.py")
        self.extra_args_entry = QLineEdit("")
        self.extra_args_entry.setPlaceholderText("例如: --some-flag value")
        config_form.addRow("CSV工具脚本:", self.script_name_entry)
        config_form.addRow("额外转换参数:", self.extra_args_entry)
        config_group.setLayout(config_form)
        left_layout.addWidget(config_group)

        # 2. 任务列表展现
        queue_group = QGroupBox("任务列表 (只读，请编辑 tasks.csv)")
        queue_layout = QVBoxLayout()
        self.task_list_widget = QListWidget()
        self.task_list_widget.itemSelectionChanged.connect(self.on_task_item_selected)
        queue_layout.addWidget(self.task_list_widget)

        list_btn_layout = QHBoxLayout()
        self.btn_reload = QPushButton("刷新 CSV")
        self.btn_reload.clicked.connect(self.reload_csv)
        self.btn_create_template = QPushButton("初始化模板")
        self.btn_create_template.clicked.connect(self.create_template_in_job_dir)
        self.btn_start = QPushButton("开始批处理运行")
        self.btn_start.clicked.connect(self.start_batch_run)
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; height: 30px;")

        self.btn_stop = QPushButton("🛑 紧急停止 (KILL)")
        self.btn_stop.clicked.connect(self.stop_batch_run)
        self.btn_stop.setEnabled(False)  # 默认不可用
        self.btn_stop.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; height: 30px;")

        list_btn_layout.addWidget(self.btn_reload)
        list_btn_layout.addWidget(self.btn_create_template)
        queue_layout.addLayout(list_btn_layout)
        queue_layout.addWidget(self.btn_start)
        queue_layout.addWidget(self.btn_stop)
        queue_group.setLayout(queue_layout)
        left_layout.addWidget(queue_group)

        splitter.addWidget(left_container)

        # 右侧：详情与日志
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        # 1. 参数详情预览
        detail_group = QGroupBox("选中任务详情")
        detail_group.setMaximumHeight(250) # 限制详情框的最大高度，防止挤压日志
        detail_v_layout = QVBoxLayout(detail_group)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.detail_content_widget = QWidget()
        self.detail_layout = QFormLayout(self.detail_content_widget)
        self.scroll_area.setWidget(self.detail_content_widget)

        detail_v_layout.addWidget(self.scroll_area)
        right_layout.addWidget(detail_group)

        # 2. 日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        right_layout.addWidget(log_group, stretch=1)

        splitter.addWidget(right_container)
        splitter.setSizes([500, 700])

    # --- 逻辑控制 ---

    def log(self, message):
        self.log_text.append(message.strip())
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def select_job_dir(self):
        start_path = str(PROJECT_ROOT / "sim_jobs")
        path = QFileDialog.getExistingDirectory(self, "选择项目工作目录", start_path)
        if path:
            self.set_job_dir(Path(path))

    def set_job_dir(self, path: Path):
        self.current_job_dir = path
        self.job_dir_entry.setText(str(path))
        self.btn_open_folder.setEnabled(True)

        # 1. 保存到全局状态
        self.save_global_state()
        # 2. 加载项目专属配置
        self.load_job_config()
        # 3. 加载任务数据
        self.load_csv_data()
        self.log(f"日志: 已切换到目录 {path.name}")

    # --- 持久化设置 ---

    def load_global_state(self):
        if os.path.exists(GLOBAL_STATE_FILE):
            try:
                with open(GLOBAL_STATE_FILE, 'r') as f:
                    state = json.load(f)
                    last_dir = state.get("last_job_dir")
                    if last_dir and os.path.exists(last_dir):
                        self.set_job_dir(Path(last_dir))
            except:
                pass

    def save_global_state(self):
        if self.current_job_dir:
            try:
                with open(GLOBAL_STATE_FILE, 'w') as f:
                    json.dump({"last_job_dir": str(self.current_job_dir)}, f)
            except:
                pass

    def load_job_config(self):
        config_path = self.current_job_dir / JOB_CONFIG_NAME
        # 默认值
        self.script_name_entry.setText("csv_tool_constant_energy.py")
        self.extra_args_entry.setText("")

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    self.script_name_entry.setText(cfg.get("script_name", "csv_tool_constant_energy.py"))
                    self.extra_args_entry.setText(cfg.get("extra_args", ""))
            except:
                pass

    def save_job_config(self):
        if not self.current_job_dir: return
        config_path = self.current_job_dir / JOB_CONFIG_NAME
        cfg = {
            "script_name": self.script_name_entry.text().strip(),
            "extra_args": self.extra_args_entry.text().strip()
        }
        try:
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            self.log(f"警告: 无法保存项目配置: {e}")

    # --- CSV 数据处理 (只读) ---

    def reload_csv(self):
        if self.current_job_dir:
            self.load_csv_data()
            self.log("日志: CSV 列表已刷新。")

    def load_csv_data(self):
        self.tasks_from_csv.clear()
        csv_path = self.current_job_dir / FILENAME_TASKS_CSV
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    self.tasks_from_csv = [row for row in reader if any(row.values())]
            except Exception as e:
                self.log(f"错误: 读取CSV失败: {e}")

        self.load_history_data()
        self.update_task_list_ui()

    def load_history_data(self):
        self.history_hashes.clear()
        history_file = self.current_job_dir / FILENAME_HISTORY
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        status = STATUS_COMPLETED if entry.get('status') == 'success' else STATUS_FAILED
                        self.history_hashes.add((entry['hash'], status))
                    except:
                        continue

    def update_task_list_ui(self):
        self.task_list_widget.clear()
        history_map = {h[0]: h[1] for h in self.history_hashes}

        for task in self.tasks_from_csv:
            name = task.get(COLUMN_TASK_NAME, "unnamed")
            # 计算哈希判断状态 (排除任务名列)
            params_only = {k: v for k, v in task.items() if k != COLUMN_TASK_NAME}
            p_str = json.dumps(params_only, sort_keys=True, separators=(',', ':'))
            p_hash = __import__('hashlib').sha256(p_str.encode()).hexdigest()[:12]

            status = history_map.get(p_hash, STATUS_PENDING)
            item = QListWidgetItem(f"[{status}] {name}")
            if status == STATUS_COMPLETED:
                item.setBackground(QColor("#eaffea"))
            elif status == STATUS_FAILED:
                item.setBackground(QColor("#ffeaea"))
            self.task_list_widget.addItem(item)

    def on_task_item_selected(self):
        # 清空旧详情
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        idx = self.task_list_widget.currentRow()
        if 0 <= idx < len(self.tasks_from_csv):
            task = self.tasks_from_csv[idx]
            for k, v in task.items():
                label = QLabel(f"{k}:")
                label.setStyleSheet("color: #888; font-size: 10px;")
                val = QLineEdit(str(v))
                val.setReadOnly(True)
                val.setFrame(False)
                self.detail_layout.addRow(label, val)

    # --- 执行逻辑 ---
    def create_template_in_job_dir(self):
        if not self.current_job_dir: return
        self.save_job_config()
        script = self.script_name_entry.text().strip()
        csv_tool = str(PROJECT_ROOT / "batch" / script)
        target_csv = str(self.current_job_dir / FILENAME_TASKS_CSV)

        cmd = [sys.executable, csv_tool, "generate-template", "-o", target_csv]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            QMessageBox.information(self, "成功", "模板已创建，请手动编辑 tasks.csv 后点击刷新。")
            self.reload_csv()
        else:
            QMessageBox.critical(self, "失败", res.stderr)

    def start_batch_run(self):
        if not self.current_job_dir:
            QMessageBox.warning(self, "错误", "请先选择工作目录。")
            return
        if self.batch_process and self.batch_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "提示", "批处理任务已在运行中。")

        self.save_job_config()  # 运行前保存当前参数

        # 0. 预准备
        conda_cmd = get_conda_activation_command()

        # 1. 转换 (使用 WSL)
        self.log("\n>>> 步骤 1: 转换任务清单...")
        wsl_csv = get_wsl_path(self.current_job_dir / FILENAME_TASKS_CSV)
        script_name = self.script_name_entry.text().strip()
        wsl_script = f"{PROJECT_ROOT_WSL.rstrip('/')}/batch/{script_name}"
        extra_args = self.extra_args_entry.text().strip()

        cmd_convert = f"{conda_cmd} && python {wsl_script} convert {wsl_csv} {extra_args}"

        convert_result = subprocess.run(
            ["wsl.exe", "-e", "bash", "-c", cmd_convert],
            capture_output=True, text=True, encoding='utf-8'
        )
        self.log(convert_result.stdout)
        if convert_result.returncode != 0:
            self.log(f"转换失败: {convert_result.stderr}")
            return

        # 2. 运行 batch_runner
        self.log("\n>>> 步骤 2: 启动批处理引擎...")
        wsl_job_dir = get_wsl_path(self.current_job_dir)
        wsl_runner = f"{PROJECT_ROOT_WSL.rstrip('/')}/batch/batch_runner.py"

        cmd_runner = (
            f"export TERM=dumb && "
            f"{conda_cmd} && "
            f"echo \"PID:$$\"; "
            f"exec python {wsl_runner} '{wsl_job_dir}'"
        )

        wsl_cmd = ["wsl.exe", "-e", "bash", "-c", cmd_runner]

        self.batch_process = QProcess()
        self.batch_process.setProcessChannelMode(QProcess.MergedChannels)
        self.batch_process.readyReadStandardOutput.connect(self.handle_output)
        self.batch_process.finished.connect(self.on_finished)

        self.batch_process.start(wsl_cmd[0], wsl_cmd[1:])

        self.btn_start.setEnabled(False)
        self.btn_start.setText("正在运行...")

        self.btn_start.setEnabled(False)
        self.btn_start.setText("正在运行...")
        self.btn_stop.setEnabled(True)  # 启用停止按钮

    def handle_output(self):
        data = self.batch_process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
        if self.wsl_pid is None:
            match = re.search(r'PID:(\d+)', data)
            if match:
                self.wsl_pid = match.group(1)
                self.log(f"系统: 捕获进程 PID {self.wsl_pid}")
        self.log(data)

    def on_finished(self):
        self.log("\n>>> 批处理运行结束。")
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始批处理运行")
        self.btn_stop.setEnabled(False)

        self.wsl_pid = None
        self.reload_csv()

    def open_current_job_dir(self):
        if self.current_job_dir: os.startfile(self.current_job_dir)

    def stop_batch_run(self):
        if not self.batch_process or self.batch_process.state() == QProcess.NotRunning:
            self.log("日志: 没有正在运行的任务。")
            return

        if self.wsl_pid:
            self.log(f"日志: 正在通过 WSL kill 命令终止进程组 {self.wsl_pid}...")
            kill_command = ["wsl.exe", "kill", "-9", f"-{self.wsl_pid}"]
            result = subprocess.run(kill_command, capture_output=True, text=True)
            if result.returncode == 0:
                self.log("日志: 进程组终止信号已成功发送。")
            else:
                self.log(f"日志: 发送终止信号失败: {result.stderr or result.stdout}")
        else:
            self.log("警告: 未能捕获到WSL PID，将尝试常规终止。")

        self.batch_process.terminate()  # 尝试优雅终止
        self.batch_process.waitForFinished(1000)  # 等待1秒
        if self.batch_process:
            self.batch_process.kill()  # 强制终止

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = SimulationControllerGUI()
    window.show()
    sys.exit(app.exec())
