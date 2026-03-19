#!/usr/bin/env python3

# 这是个兼容性文件，因为部分旧的dill会使用根目录的config，而且它也可以作为分析和模拟代码的桥接文件

import simulation.config as config
from simulation.config import SimulationParameters