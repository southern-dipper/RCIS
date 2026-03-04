# RCIS  面向安全关键具身智能体的增强规划

基于 Unicycle 运动学模型，在二维障碍物环境中实现 A* 与 RRT 两种路径规划算法，并集成扰动鲁棒性分析。

## 项目结构

```
RCIS/
 A_star_rpis.py   # 基于 A* 的路径规划
 RRT_rpis.py      # 基于 RRT 的路径规划
 quick_test.py    # 快速验证脚本
 check_env.py     # 环境依赖检查
 requirements.txt # 依赖列表
```

## 环境配置

**Python 版本要求：** 3.8+

```bash
pip install -r requirements.txt
```

如需确认依赖是否安装正确：

```bash
python check_env.py
```

## 运行

运行 A* 路径规划：

```bash
python A_star_rpis.py
```

运行 RRT 路径规划：

```bash
python RRT_rpis.py
```

快速验证：

```bash
python quick_test.py
```

## 算法说明

| 算法 | 状态空间 | 动作空间 | 特点 |
|------|----------|----------|------|
| A*   | [x, y, θ] 离散网格 | 角速度 ω | 最优性保证，适合静态环境 |
| RRT  | [x, y, θ] 连续采样 | 角速度 ω | 概率完备，适合复杂障碍物场景 |

运动模型为 Unicycle（单轮车），线速度固定 V = 0.99，通过控制角速度实现转向。