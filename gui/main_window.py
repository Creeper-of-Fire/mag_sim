# gui/main_window.py

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QLabel, QPushButton, QListWidget,
    QTextEdit, QSplitter, QGroupBox, QListWidgetItem, QMessageBox,
    QFileDialog, QCheckBox
)

from gui.app_config import DATA_DIR  # DATA_DIR 仍然可以用于存储GUI本身的状态，如默认参数
from simulation.config import SimulationParameters
from utils.project_config import PROJECT_ROOT, FILENAME_HISTORY, get_conda_activation_command, \
    PROJECT_ROOT_WSL, FILENAME_DEFAULT_PARAMS, FILENAME_TASKS_CSV, COLUMN_TASK_NAME, STATUS_FAILED, STATUS_COMPLETED, \
    STATUS_PENDING, get_wsl_path


class SimulationControllerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("等离子体模拟控制器 (文件驱动)")
        self.setGeometry(100, 100, 1400, 900)

        # --- 新的状态变量 ---
        self.current_job_dir = None
        self.current_csv_path = None
        self.tasks_from_csv = []  # CSV在内存中的表示
        self.history_hashes = set()
        self.batch_process = None
        self.wsl_pid = None
        self.pid_capture_buffer = ""

        # --- 脚本路径 ---
        self.csv_tool_script = str(PROJECT_ROOT / "batch" / "csv_tool.py")
        self.batch_runner_script = str(PROJECT_ROOT / "batch" / "batch_runner.py")

        self.param_entries = {}
        self.default_params_file = os.path.join(DATA_DIR, FILENAME_DEFAULT_PARAMS)
        os.makedirs(DATA_DIR, exist_ok=True)

        self.setup_ui()
        self.load_default_parameters()
        self.apply_stylesheet()

    def apply_stylesheet(self):
        """应用一个主题样式表。"""
        style = """
        QListWidget::item:selected {
            background-color: #0078d7; color: white; border-radius: 3px;
        }
        """
        self.setStyleSheet(style)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- 左侧面板 ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        # 1. 模拟工作目录
        job_dir_group = QGroupBox("1. 模拟工作目录")
        job_dir_layout = QHBoxLayout()
        self.job_dir_entry = QLineEdit()
        self.job_dir_entry.setReadOnly(True)
        self.job_dir_entry.setPlaceholderText("请选择或创建一个工作目录...")
        select_dir_button = QPushButton("选择目录...")
        select_dir_button.clicked.connect(self.select_job_dir)

        # “打开目录”按钮
        self.open_dir_button = QPushButton("打开目录")
        self.open_dir_button.clicked.connect(self.open_current_job_dir)
        self.open_dir_button.setEnabled(False)  # 默认禁用

        job_dir_layout.addWidget(self.job_dir_entry)
        job_dir_layout.addWidget(select_dir_button)
        job_dir_layout.addWidget(self.open_dir_button)  # 添加到布局中
        job_dir_group.setLayout(job_dir_layout)
        left_layout.addWidget(job_dir_group)

        # 2. 参数配置
        param_group = QGroupBox("2. 参数配置")
        param_layout = QFormLayout()
        default_params = SimulationParameters()
        self.attributes_order = [a for a in dir(default_params) if not a.startswith('__') and not callable(getattr(default_params, a))]

        # 为任务名添加一个输入框
        self.task_name_entry = QLineEdit()
        param_layout.addRow(QLabel(f"{COLUMN_TASK_NAME}:"), self.task_name_entry)

        for attr_name in self.attributes_order:
            default_value = getattr(default_params, attr_name)
            entry = QCheckBox() if isinstance(default_value, bool) else QLineEdit(str(default_value))
            if isinstance(entry, QCheckBox): entry.setChecked(default_value)
            param_layout.addRow(QLabel(f"{attr_name}:"), entry)
            self.param_entries[attr_name] = entry

        param_button_layout = QHBoxLayout()
        add_button = QPushButton("添加到任务列表")
        add_button.clicked.connect(self.add_task_to_csv)
        self.update_button = QPushButton("更新选中项")
        self.update_button.clicked.connect(self.update_selected_task_in_csv)
        self.update_button.setEnabled(False)
        save_defaults_button = QPushButton("保存为默认值")
        save_defaults_button.clicked.connect(self.save_default_parameters)
        param_button_layout.addWidget(add_button)
        param_button_layout.addWidget(self.update_button)
        param_button_layout.addWidget(save_defaults_button)
        param_layout.addRow(param_button_layout)
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 3. 任务队列
        queue_group = QGroupBox(f"3. 任务列表 ('{FILENAME_TASKS_CSV}')")
        queue_layout = QVBoxLayout()
        self.task_list_widget = QListWidget()
        self.task_list_widget.itemSelectionChanged.connect(self.on_task_item_selected)
        queue_layout.addWidget(self.task_list_widget)

        # 文件操作按钮布局
        queue_file_layout = QHBoxLayout()
        self.create_template_button = QPushButton("创建/重置模板")
        self.create_template_button.clicked.connect(self.create_template_in_job_dir)
        self.create_template_button.setEnabled(False)  # 默认禁用
        self.reload_list_button = QPushButton("刷新列表")
        self.reload_list_button.clicked.connect(self.reload_csv)
        self.reload_list_button.setEnabled(False)  # 默认禁用
        queue_file_layout.addWidget(self.create_template_button)
        queue_file_layout.addWidget(self.reload_list_button)
        queue_layout.addLayout(queue_file_layout)  # 添加到主布局

        # 任务执行按钮布局
        button_layout = QHBoxLayout()
        self.start_stop_button = QPushButton("开始运行")
        self.start_stop_button.clicked.connect(self.start_batch_run)
        delete_button = QPushButton("删除选中")
        delete_button.clicked.connect(self.delete_selected_task_from_csv)
        clear_button = QPushButton("清空列表")
        clear_button.clicked.connect(self.clear_csv)
        button_layout.addWidget(self.start_stop_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(clear_button)
        queue_layout.addLayout(button_layout)
        queue_group.setLayout(queue_layout)
        left_layout.addWidget(queue_group)

        # --- 右侧面板 (日志) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        log_group = QGroupBox("实时日志输出")
        right_layout.addWidget(log_group)
        log_layout_inner = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 10))
        log_layout_inner.addWidget(self.log_text)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 800])

    def log(self, message):
        self.log_text.append(message.strip())
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    # --- 工作目录与CSV文件操作 ---

    def select_job_dir(self):
        """选择或创建工作目录，并加载关联的CSV文件。"""
        sim_jobs_root = PROJECT_ROOT / "sim_jobs"
        os.makedirs(sim_jobs_root, exist_ok=True)

        dir_path = QFileDialog.getExistingDirectory(self, "选择或创建工作目录", str(sim_jobs_root))
        if not dir_path:
            return

        self.current_job_dir = Path(dir_path)
        self.job_dir_entry.setText(str(self.current_job_dir))
        self.current_csv_path = self.current_job_dir / FILENAME_TASKS_CSV
        self.log(f"日志: 已设置工作目录为 '{self.current_job_dir}'")

        # 同时启用所有依赖于工作目录的按钮
        self.open_dir_button.setEnabled(True)
        self.create_template_button.setEnabled(True)
        self.reload_list_button.setEnabled(True)

        self.load_csv_data()

    def open_current_job_dir(self):
        """在系统的文件浏览器中打开当前的工作目录。"""
        if not self.current_job_dir or not self.current_job_dir.exists():
            QMessageBox.warning(self, "错误", "当前工作目录无效或不存在。")
            return

        try:
            # os.startfile 是在 Windows 上打开文件或目录的最直接方式
            os.startfile(self.current_job_dir)
            self.log(f"日志: 已在文件浏览器中打开目录 '{self.current_job_dir}'。")
        except Exception as e:
            error_message = f"无法打开目录: {e}"
            self.log(f"错误: {error_message}")
            QMessageBox.critical(self, "错误", error_message)

    # 创建模板的槽函数
    def create_template_in_job_dir(self):
        """在当前工作目录中调用csv_tool.py来创建tasks.csv模板。"""
        if not self.current_csv_path:
            QMessageBox.warning(self, "错误", "请先设置工作目录。")
            return

        reply = QMessageBox.question(self, "确认",
                                     f"这将在 '{self.current_job_dir.name}' 目录中创建或覆盖 'tasks.csv'。\n确定要继续吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        self.log(f"日志: 正在为 '{self.current_job_dir.name}' 创建模板文件...")

        cmd = [sys.executable, self.csv_tool_script, 'generate-template', '-o', str(self.current_csv_path)]

        # 如果存在GUI的默认参数文件，则使用它来生成模板，保持一致性
        if os.path.exists(self.default_params_file):
            cmd.extend(['-d', self.default_params_file])

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        if result.returncode == 0:
            self.log("日志: 模板创建成功。")
            self.log(result.stdout)
            # 成功创建后，立即加载并显示新模板的内容
            self.reload_csv()
        else:
            self.log(f"--- 错误: 模板创建失败！ ---\n{result.stderr}")
            QMessageBox.critical(self, "错误", f"创建模板失败:\n{result.stderr}")

    # 刷新列表的槽函数
    def reload_csv(self):
        """手动从磁盘重新加载CSV和历史数据。"""
        if not self.current_job_dir:
            QMessageBox.warning(self, "错误", "请先设置工作目录。")
            return
        self.log("日志: 正在从磁盘刷新任务列表...")
        self.load_csv_data()

    def load_history_data(self):
        """从 history.jsonl 加载已完成任务的状态。"""
        self.history_hashes.clear()
        history_file = self.current_job_dir / FILENAME_HISTORY
        if not history_file.exists():
            return

        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    status = STATUS_COMPLETED if entry.get('status') == 'success' else STATUS_FAILED
                    self.history_hashes.add((entry['hash'], status))
                except json.JSONDecodeError:
                    continue

    def load_csv_data(self):
        """从CSV文件加载任务数据到内存并更新UI。"""
        self.tasks_from_csv.clear()
        if self.current_csv_path and self.current_csv_path.exists():
            try:
                with open(self.current_csv_path, 'r', newline='', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.tasks_from_csv.append(row)
                self.log(f"日志: 已从 '{self.current_csv_path.name}' 加载 {len(self.tasks_from_csv)} 个任务。")
            except Exception as e:
                self.log(f"错误: 加载CSV文件失败: {e}")
        else:
            self.log("信息: 未找到 'tasks.csv'，列表为空。")
        self.load_history_data()
        self.update_task_list_widget()

    def save_csv_data(self):
        """将内存中的任务数据写回CSV文件。"""
        if not self.current_csv_path:
            QMessageBox.warning(self, "错误", "请先设置工作目录。")
            return

        if not self.tasks_from_csv:  # 如果任务列表为空，则直接写入空文件或删除
            if self.current_csv_path.exists():
                self.current_csv_path.unlink()  # 删除文件
            return

        try:
            headers = [COLUMN_TASK_NAME] + self.attributes_order
            with open(self.current_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.tasks_from_csv)
            self.log(f"日志: 任务列表已保存到 '{self.current_csv_path.name}'。")
        except Exception as e:
            self.log(f"错误: 保存CSV文件失败: {e}")

    def update_task_list_widget(self):
        """刷新任务列表的显示。"""
        self.task_list_widget.clear()
        status_colors = {
            STATUS_COMPLETED: QColor("lightgreen"),
            STATUS_FAILED: QColor("#FF9999"),
            STATUS_PENDING: QColor("white")
        }

        # 将历史记录转换为字典以便快速查找
        history_map = {h[0]: h[1] for h in self.history_hashes}

        for task in self.tasks_from_csv:
            task_name = task.get(COLUMN_TASK_NAME, "未命名任务")

            # 计算参数哈希以确定状态
            params_only = {k: v for k, v in task.items() if k != COLUMN_TASK_NAME}
            param_str = json.dumps(params_only, sort_keys=True, separators=(',', ':'))
            task_hash = __import__('hashlib').sha256(param_str.encode('utf-8')).hexdigest()

            status = history_map.get(task_hash, STATUS_PENDING)

            item = QListWidgetItem(f"[{status}] {task_name}")
            item.setBackground(status_colors.get(status, QColor("white")))
            self.task_list_widget.addItem(item)

        self.update_button.setEnabled(False)

    def get_params_from_entries(self, include_task_name=True):
        """从UI控件获取参数字典。"""
        params = {}
        if include_task_name:
            task_name = self.task_name_entry.text().strip()
            if not task_name:
                QMessageBox.warning(self, "输入错误", f"'{COLUMN_TASK_NAME}' 不能为空。")
                return None
            params[COLUMN_TASK_NAME] = task_name

        for name, widget in self.param_entries.items():
            value = widget.isChecked() if isinstance(widget, QCheckBox) else widget.text()
            params[name] = str(value)
        return params

    def add_task_to_csv(self):
        if not self.current_job_dir:
            QMessageBox.warning(self, "提示", "请先选择一个工作目录。")
            return
        params = self.get_params_from_entries()
        if params is None: return

        self.tasks_from_csv.append(params)
        self.save_csv_data()
        self.update_task_list_widget()

    def update_selected_task_in_csv(self):
        selected_row = self.task_list_widget.currentRow()
        if selected_row < 0: return

        params = self.get_params_from_entries()
        if params is None: return

        self.tasks_from_csv[selected_row] = params
        self.save_csv_data()
        self.update_task_list_widget()
        self.task_list_widget.setCurrentRow(selected_row)

    def delete_selected_task_from_csv(self):
        selected_row = self.task_list_widget.currentRow()
        if selected_row < 0: return
        self.tasks_from_csv.pop(selected_row)
        self.save_csv_data()
        self.update_task_list_widget()

    def clear_csv(self):
        if not self.tasks_from_csv: return
        reply = QMessageBox.question(self, "确认", "确定要清空任务列表 (删除tasks.csv内容) 吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.tasks_from_csv.clear()
            self.save_csv_data()
            self.update_task_list_widget()

    def on_task_item_selected(self):
        selected_row = self.task_list_widget.currentRow()
        if 0 <= selected_row < len(self.tasks_from_csv):
            task_data = self.tasks_from_csv[selected_row]
            self.task_name_entry.setText(task_data.get(COLUMN_TASK_NAME, ""))
            for name, value in task_data.items():
                if name in self.param_entries:
                    widget = self.param_entries[name]
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(str(value).lower() in ['true', '1', 't', 'y', 'yes'])
                    else:
                        widget.setText(str(value))
            self.update_button.setEnabled(True)
        else:
            self.update_button.setEnabled(False)

    # --- 默认参数文件操作 (与之前相同) ---
    def save_default_parameters(self):
        params = self.get_params_from_entries(include_task_name=False)
        if params is None: return
        try:
            with open(self.default_params_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=4)
            QMessageBox.information(self, "成功", "当前参数已保存为默认值。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存默认参数失败:\n{e}")

    def load_default_parameters(self):
        if not os.path.exists(self.default_params_file): return
        try:
            with open(self.default_params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            for name, value in params.items():
                if name in self.param_entries:
                    widget = self.param_entries[name]
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(bool(value))
                    else:
                        widget.setText(str(value))
        except Exception as e:
            self.log(f"错误: 加载默认参数失败: {e}")

    # --- 批处理执行逻辑 ---

    def start_batch_run(self):
        if not self.current_job_dir:
            QMessageBox.warning(self, "错误", "请先选择工作目录。")
            return
        if self.batch_process and self.batch_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "提示", "批处理任务已在运行中。")
            return

        # 保存当前CSV的任何更改
        self.save_csv_data()

        # 路径转换 (调用 config 中的工具)
        wsl_csv = get_wsl_path(self.current_csv_path)
        wsl_job_dir = get_wsl_path(self.current_job_dir)

        # 脚本在 WSL 中的路径 (基于配置文件中的 PROJECT_ROOT_WSL)
        wsl_csv_tool = f"{PROJECT_ROOT_WSL.rstrip('/')}/batch/csv_tool.py"
        wsl_runner = f"{PROJECT_ROOT_WSL.rstrip('/')}/batch/batch_runner.py"

        # 构造命令 (使用 config 中的生成器)
        conda_cmd = get_conda_activation_command()

        # --- 步骤 1: 在 WSL 中进行转换 ---
        self.log(f"\n{'=' * 20} 步骤 1: 转换CSV (在WSL中) {'=' * 20}")

        # 构建 WSL 转换命令
        cmd_convert = (
            f"{conda_cmd} && "
            f"python {wsl_csv_tool} convert {wsl_csv}"
        )

        convert_result = subprocess.run(
            ["wsl.exe", "-e", "bash", "-c", cmd_convert],
            capture_output=True, text=True, encoding='utf-8'
        )

        self.log(convert_result.stdout)
        if convert_result.returncode != 0:
            self.log(f"--- 错误: CSV 转换失败！ ---\n{convert_result.stderr}")
            QMessageBox.critical(self, "错误", f"WSL 内部转换失败:\n{convert_result.stderr}")
            return

        # --- 步骤 2: 在 WSL 中启动运行器 ---
        self.log(f"\n{'=' * 20} 步骤 2: 启动批处理运行器 {'=' * 20}")

        self.wsl_pid = None
        self.pid_capture_buffer = ""

        cmd_runner = (
            f"export TERM=dumb && "
            f"{conda_cmd} &&"
            f"echo \"PID:$$\"; "
            f"exec python {wsl_runner} '{wsl_job_dir}'"
        )

        wsl_cmd = ["wsl.exe", "-e", "bash", "-c", cmd_runner]

        self.batch_process = QProcess()
        self.batch_process.setProcessChannelMode(QProcess.MergedChannels)
        self.batch_process.readyReadStandardOutput.connect(self.handle_process_output)
        self.batch_process.finished.connect(self.on_batch_finished)

        self.batch_process.start(wsl_cmd[0], wsl_cmd[1:])

        self.start_stop_button.setText("停止运行")
        self.start_stop_button.clicked.disconnect()
        self.start_stop_button.clicked.connect(self.stop_batch_run)

    def handle_process_output(self):
        if not self.batch_process: return
        data = self.batch_process.readAllStandardOutput().data().decode('utf-8', errors='ignore')

        if self.wsl_pid is None:
            self.pid_capture_buffer += data
            if '\n' in self.pid_capture_buffer:
                first_line, rest_of_data = self.pid_capture_buffer.split('\n', 1)
                match = __import__('re').match(r'^PID:(\d+)', first_line.strip())
                if match:
                    self.wsl_pid = int(match.group(1))
                    self.log(f"日志: 已捕获到 WSL 中的批处理进程 PID: {self.wsl_pid}")
                    if rest_of_data: self.log(rest_of_data)
                else:
                    self.log(self.pid_capture_buffer)
                self.pid_capture_buffer = ""
        else:
            self.log(data)

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

    def on_batch_finished(self):
        self.log(f"\n{'=' * 20} 批处理任务已结束 {'=' * 20}")
        self.start_stop_button.setText("开始运行")
        try:
            self.start_stop_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.start_stop_button.clicked.connect(self.start_batch_run)

        self.batch_process = None
        self.wsl_pid = None

        # 运行结束后，刷新列表状态
        self.load_history_data()
        self.update_task_list_widget()
