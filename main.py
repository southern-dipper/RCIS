"""
主程序入口 - 整合所有模块
"""
import numpy as np
import time

# 导入所有模块
from config import *
from core_models import *
from rpis_computation import *
from astar_planner import *
from visualization import *
from performance_analysis import *

def main():
    # 定义问题
    start_continuous = np.array([0., 0., np.pi / 4])  
    goal_continuous = np.array([10., 10.])
    
    start_indices = discretize_state(*start_continuous)
    goal_xy_indices = discretize_state(goal_continuous[0], goal_continuous[1], 0)[:2]

    # 生成简单的障碍物
    obstacle_indices = set()
    
    # 添加一些简单的障碍物
    for i in range(4, 16):
        obstacle_indices.add((i, 18))  # 水平障碍
    for i in range(6, 16):
        obstacle_indices.add((i, 7)) 
    for j in range(8, 19):
        obstacle_indices.add((18, j))  # 垂直障碍
    obstacle_indices.add((4, 16))
    obstacle_indices.add((3,17))
    obstacle_indices.add((5,17))
    
    # 1. 计算鲁棒安全集
    start_time = time.time()
    S_infinity = compute_robust_safe_set_optimized(obstacle_indices, W)
    safe_set_time = time.time() - start_time

    if not S_infinity:
        print("鲁棒安全集为空，无法进行路径规划。")
        return
    
    # 2. A*算法对比实验
    comparison_results = compare_astar_methods(start_indices, goal_xy_indices, S_infinity, obstacle_indices)
    
    # 生成并打印性能指标
    metrics = generate_performance_metrics(comparison_results)
    print_academic_results_table(metrics)
    
    # 选择用于可视化的路径
    if comparison_results['graph_optimized']['path'] is not None:
        path_indices = comparison_results['graph_optimized']['path']
    elif comparison_results['baseline']['path'] is not None:
        path_indices = comparison_results['baseline']['path']
    else:
        path_indices = None

    # 3. 计算安全角度统计
    safe_angle_count = {}
    max_angles = len(theta_space)
    
    for ix, iy, itheta in S_infinity:
        key = (ix, iy)
        if key not in safe_angle_count:
            safe_angle_count[key] = 0
        safe_angle_count[key] += 1
    
    # 4. 生成路径规划可视化
    create_original_path_visualization(S_infinity, obstacle_indices, path_indices, 
                                       start_continuous, goal_continuous, safe_angle_count)

if __name__ == "__main__":
    main()
