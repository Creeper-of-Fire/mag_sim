# analysis/core/parameter_selector.py
import hashlib
import json
from pathlib import Path
from typing import List, Any, Dict, Tuple, Optional

from rich.prompt import Prompt, Confirm
from rich.table import Table

from utils.project_config import FILENAME_HISTORY
from .selector import SimpleTableSelector
from .simulation import SimulationRun
from .simulationGroup import SimulationRunGroup
from .utils import console


class ParameterSelector:
    """
    负责从一组模拟运行中提取输入参数，并交互式地协助用户：
    1. 识别变化的参数 (Variable Detection)
    2. 过滤不需要的模拟 (Filtering)
    3. 选择用于绘图的 X 轴 (X-Axis Selection)
    4. 对结果进行排序 (Sorting)
    """

    def __init__(self, runs: List[SimulationRun]):
        self.initial_runs = runs
        # 数据结构: [{'run': run_obj, 'params': dict}, ...]
        self.raw_data_items = self._load_all_params(runs)
        # 分组后的数据
        self.grouped_items = self._group_runs()
        # 后续流程使用 self.data_items，它可能是分组后的也可能是原始的
        self.data_items = self.raw_data_items

    def _group_runs(self) -> Dict[str, List[Dict[str, Any]]]:
        """根据除 run_id 外的所有参数对 runs 进行分组"""
        groups = {}
        for item in self.raw_data_items:
            params = item['params'].copy()
            params.pop('run_id', None) # 忽略 run_id
            # 使用参数字典的 JSON 字符串作为 key，确保可哈希且唯一
            key = json.dumps(params, sort_keys=True)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        return groups

    def select(self) -> Tuple[str, List[Any], List[SimulationRun]]:
        """
        主入口。执行交互式流程。
        :return: (x_key_name, sorted_x_values, sorted_run_objects)
        """
        # --- 交互式分组确认 ---
        statistical_groups = {k: v for k, v in self.grouped_items.items() if len(v) > 1}
        if statistical_groups:
            console.print(f"\n[bold yellow]检测到 {len(statistical_groups)} 组统计性重复模拟。[/bold yellow]")
            if Confirm.ask("是否要将这些重复模拟的结果进行平均（推荐）?", default=True):
                # 用户同意平均，重构 self.data_items
                new_items = []
                # 添加未分组的单个模拟
                for k, v in self.grouped_items.items():
                    if len(v) == 1:
                        new_items.append(v[0])
                # 添加合并后的 Group 对象
                for k, v in statistical_groups.items():
                    group_runs = [item['run'] for item in v]
                    group_params = v[0]['params'].copy()
                    group_params['run_id'] = f"avg of {len(v)}"

                    # 创建 SimulationRunGroup 对象
                    group_obj = SimulationRunGroup(group_runs)
                    new_items.append({'run': group_obj, 'params': group_params})

                self.data_items = new_items
                console.print(f"[green]✔ 已合并。当前处理 {len(self.data_items)} 个独立实体。[/green]")
            else:
                # 用户拒绝，保持原样
                self.data_items = self.raw_data_items

        if not self.data_items:
            console.print("[red]错误: 传入的模拟列表为空。[/red]")
            return "Unknown", [], []

        x_key = "Run Name"  # Default

        while True:
            # 1. 动态分析当前剩余数据的变化量
            if not self.data_items:
                console.print("[red]错误：所有模拟数据都被过滤掉了！[/red]")
                return "Unknown", [], []

            varying_keys, varying_details = self._analyze_varying_params()

            # 2. 决策分支

            # 情况 A: 没有变量了
            if not varying_keys:
                console.print("[yellow]警告: 当前剩余的模拟参数完全一致。将使用 Run Name 作为 X 轴。[/yellow]")
                x_key = "Run Name"
                break

            # 情况 B: 只有一个变量 -> 自动锁定 (除非用户想看)
            if len(varying_keys) == 1:
                x_key = varying_keys[0]
                console.print(f"[green]✔ 锁定单一扫描变量: [bold]{x_key}[/bold] (共 {len(self.data_items)} 个模拟)[/green]")
                break

            # 情况 C: 多个变量 -> 交互菜单
            console.print(f"\n[bold cyan]检测到 {len(varying_keys)} 个变化的参数 (当前剩余 {len(self.data_items)} 个模拟)[/bold cyan]")

            # 这里我们强制单选，因为一次只能处理一个参数
            target_key = self._prompt_select_parameter(varying_keys, varying_details)
            if not target_key:
                continue # 如果没选或取消

            console.print(f"\n你选择了参数: [bold magenta]{target_key}[/bold magenta]")

            # 询问操作
            action = Prompt.ask(
                "请选择操作 ([bold green]x[/]: 设为绘图 X 轴 / [bold red]f[/]: 筛选/剔除数据)",
                choices=["x", "f"], default="x", show_choices=False, case_sensitive=False
            )

            if action == "x":
                x_key = target_key
                break
            else:
                self._filter_data(target_key)

        # 3. 排序并返回
        return self._sort_and_export(x_key)

    # --- 辅助UI方法 ---
    def _prompt_select_parameter(self, keys: List[str], details: Dict[str, List[str]]) -> Optional[str]:
        """使用通用选择器让用户从变化参数列表中选一个"""

        # 定义转换函数：参数名 -> [参数名, 示例值]
        def row_converter(k):
            vals = details[k]
            # 截取前3个值作为示例
            val_str = ", ".join(map(str, vals[:3])) + ("..." if len(vals) > 3 else "")
            return [f"[magenta]{k}[/magenta]", f"[dim]{val_str}[/dim]"]

        selector = SimpleTableSelector(
            items=keys,
            columns=["参数名", "当前包含的值 (示例)"],
            row_converter=row_converter,
            title="变化参数列表"
        )

        selected = selector.select(single=True, default="0")

        if not selected:
            # 用户可能按 Ctrl+C 中断了
            return None

        # 因为是 single 模式，返回的列表最多只有一个元素
        return selected[0]

    def _filter_data(self, key: str):
        """让用户选择保留哪些值（支持多选）"""
        # 获取所有唯一值
        all_vals = [item['params'].get(key) for item in self.data_items]
        # 转为字符串用于显示和比较，去重并排序
        unique_vals = sorted(list(set(map(str, all_vals))))

        selector = SimpleTableSelector(
            items=unique_vals,
            columns=["值"],
            row_converter=lambda v: [f"[cyan]{v}[/cyan]"],
            title=f"参数 [{key}] 的分布情况"
        )

        console.print("[dim]提示: 选中的值将被【保留】，未选中的将被剔除。[/dim]")

        # 这里利用了 Selector 的多选能力
        values_to_keep = selector.select(prompt_text="请选择要【保留】的数据值", default="all")

        if not values_to_keep:
            console.print("[yellow]未选择任何值，操作取消。[/yellow]")
            return

        # 执行过滤
        before_count = len(self.data_items)
        self.data_items = [
            item for item in self.data_items
            if str(item['params'].get(key)) in values_to_keep
        ]
        after_count = len(self.data_items)

        console.print(f"[green]已保留 {len(values_to_keep)} 组参数值。[/green]")
        console.print(f"[dim]模拟数量从 {before_count} 减少到 {after_count}。[/dim]\n")


    # --- 辅助功能 ---

    @staticmethod
    def generate_filename(x_label: str, runs: List[SimulationRun], prefix: str = "scan") -> str:
        """
        生成带有哈希后缀的文件名，并允许用户交互式修改后缀。
        保证只要参与的 runs 集合不变，默认哈希就不变。
        """
        # 1. 计算哈希：基于所有 run 的名字排序拼接
        # 这样即使 runs 列表顺序不同，只要内容一样，哈希就一样
        run_names_concat = "".join(sorted([r.name for r in runs]))
        short_hash = hashlib.md5(run_names_concat.encode('utf-8')).hexdigest()[:6]

        default_filename = f"{prefix}_{x_label}_{short_hash}"
        console.print(f"\n[cyan]默认输出文件名: {default_filename}.png[/cyan]")

        user_suffix = Prompt.ask(
            "请输入文件名后缀 (用于区分实验批次)",
            default=short_hash,
            show_default=True
        )

        # 即使这里 x_label 可能包含非法字符，但在 create_analysis_figure 里通常会处理
        # 这里简单替换一下空格
        clean_label = x_label.replace(" ", "_")
        return f"{prefix}_{clean_label}_{user_suffix}"

    # ================= Internal Helpers =================

    def _load_all_params(self, runs: List[SimulationRun]) -> List[Dict[str, Any]]:
        """
        加载所有 run 的参数。
        使用内存字典缓存 history 文件，避免对同一个文件进行几十次重复的磁盘 I/O 和 JSON 解析。
        优先级: 本地 custom_params.json > 全局 history.jsonl > run.sim 默认属性
        """
        # 1. 预解析 History 文件，建立全局缓存 { history_path: { run_name: params } }
        history_cache: Dict[Path, Dict[str, Any]] = {}

        for run in runs:
            run_path = Path(run.path).resolve()
            # 假设结构: job_dir / sim_results / run_dir -> history 在 job_dir 下
            history_path = run_path.parent.parent / FILENAME_HISTORY

            # 如果这个文件还没读过，且真实存在，就全量读入并解析一次
            if history_path not in history_cache and history_path.exists():
                history_cache[history_path] = {}
                try:
                    with open(history_path, 'r', encoding='utf-8') as f:
                        # 顺序读取，后出现的同名记录会覆盖前面的记录（完美等价于反向查找）
                        for line in f:
                            try:
                                record = json.loads(line)
                                out_dir = record.get('output_dir', '')
                                if out_dir:
                                    # 仅提取目录名作为 Key，避免极度耗时的路径 resolve
                                    r_name = Path(out_dir).name
                                    history_cache[history_path][r_name] = record.get('params', {})
                            except Exception:
                                continue
                except Exception:
                    pass

        # 2. 快速组装数据 (O(1) 内存查询)
        items = []
        for run in runs:
            run_path = Path(run.path).resolve()
            history_path = run_path.parent.parent / FILENAME_HISTORY

            params = {}

            # 层级 1: 从 run.sim 提取基础参数
            for k, v in vars(run.sim).items():
                if isinstance(v, (int, float, str, bool)) and not k.startswith('_'):
                    params[k] = v

            # 层级 2: 全局 history 文件覆盖
            if history_path in history_cache and run_path.name in history_cache[history_path]:
                params.update(history_cache[history_path][run_path.name])

            # 层级 3: 本地 custom_params.json 覆盖 (最高优先级)
            # 这允许 Slicer 工具注入自定义的虚拟参数
            local_param_file = run_path / "custom_params.json"
            if local_param_file.exists():
                try:
                    with open(local_param_file, 'r', encoding='utf-8') as f:
                        local_params = json.load(f)
                        params.update(local_params)
                except Exception as e:
                    console.print(f"[yellow]⚠ 读取 {local_param_file} 失败: {e}[/yellow]")

            items.append({'run': run, 'params': params})

        return items

    def _analyze_varying_params(self) -> Tuple[List[str], Dict[str, List[str]]]:
        all_keys = set()
        for item in self.data_items:
            all_keys.update(item['params'].keys())

        varying_keys = []
        varying_details = {}

        for k in all_keys:
            values = set()
            for item in self.data_items:
                val = item['params'].get(k, None)
                values.add(str(val))

            if len(values) > 1:
                varying_keys.append(k)
                varying_details[k] = list(values)

        return varying_keys, varying_details

    def _print_param_table(self, keys: List[str], details: Dict[str, List[str]]):
        table = Table(title="变化参数列表")
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("参数名", style="magenta")
        table.add_column("当前包含的值 (示例)", style="green")

        for i, key in enumerate(keys):
            vals = details[key]
            val_str = ", ".join(vals[:3]) + ("..." if len(vals) > 3 else "")
            table.add_row(str(i + 1), key, val_str)
        console.print(table)
        console.print("[dim]提示: 选择参数后，你可以将其设定为 X 轴，或者根据其值过滤掉不需要的模拟。[/dim]")

    def _sort_and_export(self, x_key: str) -> Tuple[str, List[Any], List[SimulationRun]]:
        def sort_key(item):
            if x_key == "Run Name":
                return item['run'].name
            val = item['params'].get(x_key, 0)
            try:
                return float(val)
            except:
                return str(val)

        self.data_items.sort(key=sort_key)

        sorted_runs = [item['run'] for item in self.data_items]
        sorted_values = [item['params'].get(x_key, item['run'].name if x_key == "Run Name" else "N/A") for item in self.data_items]

        return x_key, sorted_values, sorted_runs
