# Plasma Simulation 项目结构说明

基于 WarpX 的相对论电子-正电子对等离子体粒子模拟（PIC）框架。

## 目录概览

| 目录 | 用途 |
|------|------|
| simulation/ | 核心模拟引擎，配置并运行 WarpX PIC 模拟 |
| analysis/ | 模块化后处理分析框架（场演化、能谱、粒子追踪等） |
| analysis_cli/ | 分析 CLI 基础设施：模块发现、工具工作流、状态管理（AnalysisStore 单例） |
| batch/ | 批量作业系统：参数扫描、多后端提交（WSL/Gongji/Yingbo） |
| tui/ | Textual 终端 UI 应用，管理模拟任务 |
| tools/ | 独立工具脚本（公式渲染、物理量探索、HDF5 检查等） |
| utils/ | 共享工具：项目路径常量、CSV 文件定位 |
| data/ | 运行时状态文件（GUI 状态、模块选择缓存） |
| test/ | 测试脚本（WarpX 重联 benchmark、OpenMP 扩展性测试） |
| sim_jobs/ | **模拟数据输出目录，数据量极大，搜索时必须绕开** |

## 顶层入口文件

| 文件 | 用途 |
|------|------|
| main.py | 模拟入口，解析 CLI 参数并运行单次模拟 |
| analysis_console.py | 基于 Rich 的交互式分析控制台（REPL 风格） |
| tui_app.py | TUI 应用启动器 |
| config.py | 向后兼容层，重导出 simulation.config.SimulationParameters |
| pyproject.toml | 包元数据（plasma_simulation v0.1.0） |

## 工作流

1. **配置** — 通过 simulation/config.py、batch/csv_tool.py 或 TUI 定义模拟参数
2. **模拟** — 通过 main.py 本地运行（MPI）或 batch/ 系统批量提交到远程集群
3. **分析** — 通过 analysis_console.py 交互式加载 HDF5 数据，运行模块化分析
