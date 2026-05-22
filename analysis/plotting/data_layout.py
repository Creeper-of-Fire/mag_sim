# plotting/data_layout.py
"""
数据捕获布局层：在绘图时透明记录数据，自动导出 CSV。

继承链: AnalysisLayout → DataLayout → ComparisonLayout
"""

import re
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .layout import AnalysisLayout
from ..core.utils import _determine_output_base_path, console
from utils.project_config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# 数据记录
# ---------------------------------------------------------------------------

class SeriesRecord(NamedTuple):
    method: str                       # 'plot', 'fill_between', 'errorbar', 'loglog', 'semilogy'
    x: np.ndarray
    y: Optional[np.ndarray] = None
    y2: Optional[np.ndarray] = None  # fill_between 上界
    yerr: Optional[np.ndarray] = None
    xerr: Optional[np.ndarray] = None
    label: Optional[str] = None


# ---------------------------------------------------------------------------
# DataAxis — 通过 __class__ 替换将真实 Axes 提升为数据捕获轴
# ---------------------------------------------------------------------------

_DAX_ATTRS = ('_daxis_records', '_daxis_ylabel', '_daxis_xlabel')


def _make_data_axis(ax: Axes) -> 'DataAxis':
    ax.__class__ = DataAxis
    ax._daxis_records: list[SeriesRecord] = []
    ax._daxis_ylabel: str | None = None
    ax._daxis_xlabel: str | None = None
    return ax                                     # type: ignore[return-value]


class DataAxis(Axes):
    """继承 Axes，覆盖数据方法以捕获绘图数据。

    不要直接构造；使用 _make_data_axis(ax) 将已有 Axes 提升为 DataAxis。
    """

    # ---- 内部属性访问 (避免触发 Axes 属性查找) ----

    @property
    def daxis_records(self) -> list[SeriesRecord]:
        return self._daxis_records                # type: ignore[attr-defined]

    @property
    def daxis_has_data(self) -> bool:
        return len(self._daxis_records) > 0       # type: ignore[attr-defined]

    @property
    def daxis_ylabel(self) -> str | None:
        return self._daxis_ylabel                 # type: ignore[attr-defined]

    @property
    def daxis_xlabel(self) -> str | None:
        return self._daxis_xlabel                 # type: ignore[attr-defined]

    # ---- 参数解析 ----

    @staticmethod
    def _parse_plot_args(*args, **kwargs):
        label = kwargs.get('label', None)
        if len(args) == 1:
            y = np.asarray(args[0])
            x = np.arange(len(y), dtype=float)
        elif len(args) >= 2:
            x, y = np.asarray(args[0]), np.asarray(args[1])
        else:
            x, y = np.array([]), np.array([])
        return x, y, label

    # ---- 覆盖的数据方法 ----

    def plot(self, *args, **kwargs):
        x, y, label = self._parse_plot_args(*args, **kwargs)
        self._daxis_records.append(SeriesRecord('plot', x, y, label=label))  # type: ignore[attr-defined]
        return Axes.plot(self, *args, **kwargs)

    def loglog(self, *args, **kwargs):
        x, y, label = self._parse_plot_args(*args, **kwargs)
        self._daxis_records.append(SeriesRecord('loglog', x, y, label=label))  # type: ignore[attr-defined]
        return Axes.loglog(self, *args, **kwargs)

    def semilogy(self, *args, **kwargs):
        x, y, label = self._parse_plot_args(*args, **kwargs)
        self._daxis_records.append(SeriesRecord('semilogy', x, y, label=label))  # type: ignore[attr-defined]
        return Axes.semilogy(self, *args, **kwargs)

    def fill_between(self, x, y1, y2=0, **kwargs):
        label = kwargs.get('label', None)
        self._daxis_records.append(SeriesRecord(  # type: ignore[attr-defined]
            'fill_between', np.asarray(x), np.asarray(y1), np.asarray(y2), label=label,
        ))
        return Axes.fill_between(self, x, y1, y2, **kwargs)

    def errorbar(self, x, y, yerr=None, xerr=None, **kwargs):
        label = kwargs.get('label', None)
        self._daxis_records.append(SeriesRecord(  # type: ignore[attr-defined]
            'errorbar', np.asarray(x), np.asarray(y),
            yerr=np.asarray(yerr) if yerr is not None else None,
            xerr=np.asarray(xerr) if xerr is not None else None,
            label=label,
        ))
        return Axes.errorbar(self, x, y, yerr=yerr, xerr=xerr, **kwargs)

    # ---- 覆盖的元数据方法 ----

    def set_ylabel(self, ylabel, *args, **kwargs):
        self._daxis_ylabel = ylabel               # type: ignore[attr-defined]
        return Axes.set_ylabel(self, ylabel, *args, **kwargs)

    def set_xlabel(self, xlabel, *args, **kwargs):
        self._daxis_xlabel = xlabel               # type: ignore[attr-defined]
        return Axes.set_xlabel(self, xlabel, *args, **kwargs)


# ---------------------------------------------------------------------------
# CSV 构建辅助
# ---------------------------------------------------------------------------

def _sanitize_label(label: str | None) -> str:
    if not label:
        return 'series'
    s = label.strip('$')
    for latex, ascii_name in [
        (r'\langle', ''), (r'\rangle', ''),
        (r'\epsilon', 'eps'), (r'\lambda', 'lam'),
        (r'\sigma', 'sigma'), (r'\theta', 'theta'),
        (r'\alpha', 'alpha'), (r'\beta', 'beta'),
    ]:
        s = s.replace(latex, ascii_name)
    s = re.sub(r'[^A-Za-z0-9_一-鿿-]', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'series'


def _deduplicate(name: str, seen: dict[str, int]) -> str:
    if name not in seen:
        seen[name] = 1
        return name
    seen[name] += 1
    return f'{name}_{seen[name]}'


def _build_dataframe(dax: DataAxis, x_name: str = 'x') -> pd.DataFrame:
    records = dax.daxis_records
    if not records:
        return pd.DataFrame()

    columns: dict[str, np.ndarray] = {x_name: records[0].x}
    seen: dict[str, int] = {}

    for rec in records:
        if rec.method in ('plot', 'loglog', 'semilogy'):
            col = _deduplicate(_sanitize_label(rec.label), seen)
            columns[col] = rec.y

        elif rec.method == 'fill_between':
            col = _deduplicate(_sanitize_label(rec.label), seen)
            columns[f'{col}_lower'] = rec.y
            columns[f'{col}_upper'] = rec.y2

        elif rec.method == 'errorbar':
            col = _deduplicate(_sanitize_label(rec.label), seen)
            columns[col] = rec.y
            if rec.yerr is not None:
                columns[f'{col}_yerr'] = rec.yerr
            if rec.xerr is not None:
                columns[f'{col}_xerr'] = rec.xerr

    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# DataLayout
# ---------------------------------------------------------------------------

class DataLayout(AnalysisLayout):
    """扩展 AnalysisLayout：request_axes 返回 DataAxis，退出时导出 CSV。"""

    def __init__(
        self,
        run_or_runs,
        base_filename: str,
        plot_ratio: Optional[tuple[float, float]] = None,
        override_filename: Optional[str] = None,
        ncols: int = 1,
        shared_xlabel: Optional[str] = None,
        xtick_labels: Optional[list[str]] = None,
    ):
        super().__init__(
            run_or_runs, base_filename,
            plot_ratio=plot_ratio,
            override_filename=override_filename,
            ncols=ncols,
        )
        self._shared_xlabel = shared_xlabel
        self._xtick_labels = xtick_labels

    def request_axes(self, ratio: float = 1.0) -> DataAxis:
        raw_ax = super().request_axes(ratio)
        return _make_data_axis(raw_ax)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return super().__exit__(exc_type, exc_val, exc_tb)

        # 1. 共享 xlabel
        if self._shared_xlabel is not None:
            self._apply_shared_xlabel()

        # 2. CSV 导出 (不阻断图片保存)
        try:
            self._export_csv()
        except Exception as e:
            console.print(f'  [yellow]CSV 导出跳过: {e}[/yellow]')

        # 3. 委托父类保存图片
        return super().__exit__(None, None, None)

    # ---- 内部方法 ----

    def _apply_shared_xlabel(self):
        bottom = self.bottom_row_axes
        bottom_ids = {id(ax) for ax in bottom}
        for ax in self.plot_axes:
            if id(ax) not in bottom_ids:
                ax.set_xticklabels([])
                ax.set_xlabel('')
        for ax in bottom:
            ax.set_xlabel(self._shared_xlabel)
            if self._xtick_labels is not None:
                ticks = np.arange(len(self._xtick_labels))
                ax.set_xticks(ticks)
                ax.set_xticklabels(self._xtick_labels, rotation=45, ha='right')

    def _export_csv(self):
        base_dir = _determine_output_base_path(self.run_or_runs)
        figure_stem = Path(self.output_name).stem
        csv_dir = base_dir / 'csv' / figure_stem
        csv_dir.mkdir(parents=True, exist_ok=True)

        x_name = _sanitize_label(self._shared_xlabel or 'x')
        exported = False
        for idx, dax in enumerate(self.plot_axes):
            if not isinstance(dax, DataAxis) or not dax.daxis_has_data:
                continue
            df = _build_dataframe(dax, x_name=x_name)
            if df.empty:
                continue
            df.to_csv(csv_dir / f'subplot_{idx}.csv', index=False)
            exported = True

        if exported:
            try:
                display = csv_dir.relative_to(PROJECT_ROOT)
            except ValueError:
                display = csv_dir
            console.print(f'  [green]CSV: {display}/[/green]')
