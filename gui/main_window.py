# gui/main_window.py

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import threading

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QLabel, QPushButton, QListWidget,
    QTextEdit, QSplitter, QGroupBox, QListWidgetItem, QMessageBox,
    QFileDialog
)

# 从项目根目录的 config.py 导入
from simulation.config import SimulationParameters
from .app_config import DATA_DIR
# 从同级目录的 simulation_runner.py 导入
from .simulation_runner import SimulationRunner, STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED


class SimulationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("等离子体模拟控制器 (PySide6)")
        self.setGeometry(100, 100, 1200, 800)

        self.simulation_queue = []
        self.param_entries = {}
        self.runner_thread = None
        self.runner = None

        # 确保数据目录存在
        os.makedirs(DATA_DIR, exist_ok=True)
        self.default_params_file = os.path.join(DATA_DIR, "default_params.json")

        self.setup_ui()
        self.load_default_parameters()  # 启动时加载默认参数
        self.apply_stylesheet()

    def apply_stylesheet(self):
        """应用一个主题样式表。"""
        style = """
        /* 项目被选中时的样式 */
        QListWidget::item:selected {
            background-color: #0078d7; /* 醒目的蓝色，Windows常用选择色 */
            color: white;              /* 白色文字，保证对比度 */
            border-radius: 3px;
        }
        """
        self.setStyleSheet(style)

    def setup_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- 左侧面板 ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        # 1. 参数配置
        param_group = QGroupBox("1. 参数配置")
        param_layout = QFormLayout()

        # 加载默认参数以确定字段和类型
        default_params_instance = SimulationParameters()
        self.attributes_order = [a for a in dir(default_params_instance) if not a.startswith('__') and not callable(getattr(default_params_instance, a))]

        for attr_name in self.attributes_order:
            default_value = getattr(default_params_instance, attr_name)
            entry = QLineEdit(str(default_value))
            param_layout.addRow(QLabel(f"{attr_name}:"), entry)
            self.param_entries[attr_name] = entry

        # --- 参数操作按钮 ---
        param_button_layout = QHBoxLayout()
        add_button = QPushButton("添加到队列")
        add_button.clicked.connect(self.add_to_queue)
        self.update_button = QPushButton("更新选中项")
        self.update_button.clicked.connect(self.update_selected_task)
        self.update_button.setEnabled(False)  # 默认禁用
        save_defaults_button = QPushButton("保存为默认值")
        save_defaults_button.clicked.connect(self.save_default_parameters)

        param_button_layout.addWidget(add_button)
        param_button_layout.addWidget(self.update_button)
        param_button_layout.addWidget(save_defaults_button)

        param_layout.addRow(param_button_layout)
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 2. 运行队列
        queue_group = QGroupBox("2. 运行队列")
        queue_layout = QVBoxLayout()
        self.queue_list_widget = QListWidget()
        # 连接列表项选择变化信号到槽函数
        self.queue_list_widget.itemSelectionChanged.connect(self.on_queue_item_selected)
        queue_layout.addWidget(self.queue_list_widget)

        # --- 队列文件操作按钮 ---
        queue_file_layout = QHBoxLayout()
        save_queue_button = QPushButton("保存队列...")
        save_queue_button.clicked.connect(self.save_queue)
        load_queue_button = QPushButton("加载队列...")
        load_queue_button.clicked.connect(self.load_queue)
        queue_file_layout.addWidget(save_queue_button)
        queue_file_layout.addWidget(load_queue_button)
        queue_layout.addLayout(queue_file_layout)

        # --- 队列执行操作按钮 ---
        button_layout = QHBoxLayout()
        self.start_stop_button = QPushButton("开始运行队列")
        self.start_stop_button.clicked.connect(self.start_simulation_queue)
        delete_button = QPushButton("删除选中")
        delete_button.clicked.connect(self.delete_from_queue)
        clear_button = QPushButton("清空队列")
        clear_button.clicked.connect(self.clear_queue)

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
        splitter.setSizes([500, 700])

    def log(self, message):
        """向日志窗口追加消息。"""
        self.log_text.append(message.strip())

    def get_params_from_entries(self):
        """从界面上的输入框读取参数并进行类型转换。"""
        params = {}
        default_instance = SimulationParameters()
        for name, entry in self.param_entries.items():
            value_str = entry.text()
            # 获取原始数据类型（int, float, str等）
            original_type = type(getattr(default_instance, name))
            try:
                # 尝试转换为原始类型
                params[name] = original_type(value_str)
            except ValueError:
                self.log(f"错误: 参数 '{name}' 的值 '{value_str}' 无法转换为 {original_type.__name__} 类型。\n")
                QMessageBox.warning(self, "参数错误", f"参数 '{name}' 的值 '{value_str}' 格式不正确。")
                return None
        return params

    def add_to_queue(self):
        """将当前配置的参数作为一个新任务添加到队列中。"""
        params = self.get_params_from_entries()
        if params is None:
            return

        task_name = f"任务{len(self.simulation_queue) + 1}: output_dir='{params.get('output_dir', 'N/A')}'"
        task = {"name": task_name, "params": params, "status": STATUS_PENDING}
        self.simulation_queue.append(task)
        self.update_queue_listbox()
        self.log(f"日志: 已添加任务 '{task_name}' 到队列。\n")

    def update_selected_task(self):
        """更新当前选中的任务的参数。"""
        selected_row = self.queue_list_widget.currentRow()
        if selected_row < 0:
            return

        params = self.get_params_from_entries()
        if params is None:
            return

        # 更新任务数据
        task = self.simulation_queue[selected_row]
        task['params'] = params
        task['name'] = f"任务{selected_row + 1}: output_dir='{params.get('output_dir', 'N/A')}'"  # 更新名称
        task['status'] = STATUS_PENDING  # 编辑后状态重置

        self.update_queue_listbox()
        self.log(f"日志: 已更新任务 '{task['name']}'。\n")
        self.queue_list_widget.setCurrentRow(selected_row)  # 保持选中状态

    def on_queue_item_selected(self):
        """当队列中的项目被选中时，将其参数加载到编辑区。"""
        selected_items = self.queue_list_widget.selectedItems()
        if not selected_items:
            self.update_button.setEnabled(False)
            return

        selected_row = self.queue_list_widget.currentRow()
        if 0 <= selected_row < len(self.simulation_queue):
            task_params = self.simulation_queue[selected_row]['params']
            for name, value in task_params.items():
                if name in self.param_entries:
                    self.param_entries[name].setText(str(value))
            self.update_button.setEnabled(True)
        else:
            self.update_button.setEnabled(False)

    def update_queue_listbox(self):
        """刷新队列列表的显示。"""
        self.queue_list_widget.clear()
        status_colors = {
            STATUS_RUNNING: QColor("lightblue"),
            STATUS_COMPLETED: QColor("lightgreen"),
            STATUS_FAILED: QColor("#FF9999"),  # 浅红色
            STATUS_PENDING: QColor("white")
        }
        for task in self.simulation_queue:
            item = QListWidgetItem(f"[{task['status']}] {task['name']}")
            item.setBackground(status_colors.get(task['status'], QColor("white")))
            self.queue_list_widget.addItem(item)

        # 如果没有选中项，禁用更新按钮
        if self.queue_list_widget.currentRow() == -1:
            self.update_button.setEnabled(False)

    def delete_from_queue(self):
        """从队列中删除选中的任务。"""
        selected_row = self.queue_list_widget.currentRow()
        if selected_row < 0:
            return

        task = self.simulation_queue.pop(selected_row)
        self.log(f"日志: 已删除任务 '{task['name']}'。\n")
        self.update_queue_listbox()

    def clear_queue(self):
        """清空整个任务队列。"""
        if not self.simulation_queue: return
        reply = QMessageBox.question(self, "确认", "确定要清空整个任务队列吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.simulation_queue.clear()
            self.update_queue_listbox()
            self.log("日志: 任务队列已清空。\n")

    # --- 文件操作 ---
    def save_queue(self):
        """保存当前队列到 JSON 文件。"""
        if not self.simulation_queue:
            QMessageBox.information(self, "提示", "队列为空，无需保存。")
            return

        filePath, _ = QFileDialog.getSaveFileName(self, "保存队列", DATA_DIR, "JSON Files (*.json);;All Files (*)")
        if filePath:
            try:
                with open(filePath, 'w', encoding='utf-8') as f:
                    json.dump(self.simulation_queue, f, indent=4, ensure_ascii=False)
                self.log(f"日志: 队列已成功保存到 {filePath}\n")
            except Exception as e:
                self.log(f"错误: 保存队列失败: {e}\n")
                QMessageBox.critical(self, "错误", f"保存文件失败:\n{e}")

    def load_queue(self):
        """从 JSON 文件加载队列。"""
        filePath, _ = QFileDialog.getOpenFileName(self, "加载队列", DATA_DIR, "JSON Files (*.json);;All Files (*)")
        if filePath:
            try:
                with open(filePath, 'r', encoding='utf-8') as f:
                    self.simulation_queue = json.load(f)
                self.update_queue_listbox()
                self.log(f"日志: 队列已从 {filePath} 成功加载。\n")
            except Exception as e:
                self.log(f"错误: 加载队列失败: {e}\n")
                QMessageBox.critical(self, "错误", f"加载文件失败:\n{e}")

    def save_default_parameters(self):
        """将当前参数配置区的参数保存为默认值。"""
        params = self.get_params_from_entries()
        if params is None:
            return

        try:
            with open(self.default_params_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=4)
            self.log(f"日志: 默认参数已保存到 {self.default_params_file}\n")
            QMessageBox.information(self, "成功", "当前参数已保存为默认值。")
        except Exception as e:
            self.log(f"错误: 保存默认参数失败: {e}\n")
            QMessageBox.critical(self, "错误", f"保存默认参数失败:\n{e}")

    def load_default_parameters(self):
        """从文件加载默认参数并填充到UI。"""
        if not os.path.exists(self.default_params_file):
            self.log("日志: 未找到默认参数文件，使用类内置默认值。\n")
            return

        try:
            with open(self.default_params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)

            for name, value in params.items():
                if name in self.param_entries:
                    self.param_entries[name].setText(str(value))
            self.log(f"日志: 已从 {self.default_params_file} 加载默认参数。\n")
        except Exception as e:
            self.log(f"错误: 加载默认参数失败: {e}\n")

    # --- 任务执行逻辑 ---
    def on_queue_finished(self):
        """当后台任务队列全部完成或被中止时调用的槽函数。"""
        # 恢复按钮状态
        self.start_stop_button.setText("开始运行队列")
        self.start_stop_button.setEnabled(True)
        # 解除旧的连接（如果有），重新连接到 start 函数
        try:
            self.start_stop_button.clicked.disconnect(self.stop_simulation_queue)
        except RuntimeError:  # 如果连接不存在会抛出异常
            pass
        self.start_stop_button.clicked.connect(self.start_simulation_queue)

        if self.runner_thread:
            self.runner_thread.join()
        self.runner_thread = None
        self.runner = None

    def start_simulation_queue(self):
        """开始执行模拟任务队列。"""
        if self.runner_thread is not None and self.runner_thread.is_alive():
            self.log("日志: 任务队列已在运行中。\n")
            return
        if not self.simulation_queue:
            self.log("日志: 任务队列为空。\n")
            return

        # 切换按钮为“停止运行”状态
        self.start_stop_button.setText("停止运行")
        # 解除 start 连接，连接到 stop 函数
        self.start_stop_button.clicked.disconnect(self.start_simulation_queue)
        self.start_stop_button.clicked.connect(self.stop_simulation_queue)

        # 创建并启动后台任务线程
        self.runner = SimulationRunner(self.simulation_queue)

        # 将 run 方法放在一个标准 threading.Thread 中
        # 这是因为 run 方法是阻塞的 (waitForFinished)，直接在 QThread.run 中调用会阻塞事件循环
        self.runner_thread = threading.Thread(target=self.runner.run, daemon=True)

        # 连接信号到主线程的槽函数
        self.runner.signals.log_message.connect(self.log)
        self.runner.signals.task_update.connect(self.update_queue_listbox)
        self.runner.signals.queue_finished.connect(self.on_queue_finished)

        self.runner_thread.start()

    def stop_simulation_queue(self):
        """请求停止正在运行的任务队列。"""
        if self.runner:
            self.log("日志: 正在发送停止信号...\n")
            # 禁用按钮防止重复点击
            self.start_stop_button.setEnabled(False)
            self.runner.stop()
        else:
            self.log("日志: 没有正在运行的任务。\n")
