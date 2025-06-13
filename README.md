# 基于RPIS的路径规划算法 - 模块化版本

## 项目结构

```
RPIS/
├── main.py                    # 主程序入口
├── config.py                  # 系统配置和参数
├── core_models.py             # 核心数学模型
├── rpis_computation.py        # 鲁棒安全集计算
├── astar_planner.py           # A*路径规划算法
├── visualization.py           # 可视化和图表
├── performance_analysis.py    # 性能分析和结果展示
├── A_star_rpis.py            # 原始完整版本（备份）
└── README.md                  # 项目说明文档
```

## 模块说明

### 1. `config.py` - 配置模块 (48行)
- 系统参数配置 (状态空间、动作空间、扰动参数)
- matplotlib中文字体设置
- 全局常量定义

### 2. `core_models.py` - 核心模型 (77行)
- 基础数学函数：状态验证、离散化、逆离散化
- Unicycle运动学模型
- 路径碰撞检测
- A*启发式函数

### 3. `rpis_computation.py` - RPIS计算 (118行)
- 鲁棒安全集计算的标准版本
- 优化的串行版本
- 支持扰动处理和迭代收敛

### 4. `astar_planner.py` - A*规划器 (224行)
- 三种A*算法实现：基线A*、鲁棒A*、图优化A*
- 安全状态图构建
- 算法性能对比框架

### 5. `visualization.py` - 可视化 (144行)
- 路径轨迹生成和处理
- 安全集可视化
- 路径规划结果展示

### 6. `performance_analysis.py` - 性能分析 (136行)
- 性能指标计算
- 学术论文格式的结果表格
- 算法对比分析报告

### 7. `main.py` - 主程序 (73行)
- 整合所有模块
- 系统运行流程控制
- 结果输出和总结

## 运行方式

### 运行完整系统
```bash
python main.py
```

### 单独测试模块
```python
# 测试RPIS计算
from rpis_computation import compute_robust_safe_set_optimized
# ... 其他测试代码

# 测试A*算法
from astar_planner import compare_astar_methods
# ... 其他测试代码
```