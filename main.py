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
    # 打印系统信息
    print("鲁棒路径规划系统启动")
    print(f"状态空间: {len(x_space)}×{len(y_space)}×{len(theta_space)} = {len(x_space) * len(y_space) * len(theta_space)}个状态")
    print(f"动作空间: {len(omega_space)}个动作, 扰动集: {len(W)}个角点")
    
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
    print(f"障碍物数量: {len(obstacle_indices)}")
    
    # 1. 计算鲁棒安全集（使用优化的串行版本）
    start_time = time.time()
    S_infinity = compute_robust_safe_set_optimized(obstacle_indices, W)
    print(f"鲁棒安全集计算耗时: {time.time() - start_time:.2f} 秒")

    if not S_infinity:
        print("鲁棒安全集为空，无法进行路径规划。")
        return
    
    # 2. A*算法对比实验
    comparison_results = compare_astar_methods(start_indices, goal_xy_indices, S_infinity, obstacle_indices)
    
    # 生成并打印专业性能指标
    metrics = generate_performance_metrics(comparison_results)
    print_academic_results_table(metrics)
    
    # 选择用于可视化的路径（优先选择鲁棒A*的结果）
    if comparison_results['robust']['path'] is not None:
        path_indices = comparison_results['robust']['path']
    elif comparison_results['graph_optimized']['path'] is not None:
        path_indices = comparison_results['graph_optimized']['path']
    elif comparison_results['baseline']['path'] is not None:
        path_indices = comparison_results['baseline']['path']
    else:
        path_indices = None
        print("未找到路径")

    # 3. 计算安全角度统计
    safe_angle_count = {}
    max_angles = len(theta_space)
    
    for ix, iy, itheta in S_infinity:
        key = (ix, iy)
        if key not in safe_angle_count:
            safe_angle_count[key] = 0
        safe_angle_count[key] += 1
    
    # 4. 生成路径规划可视化
    print("生成路径可视化...")
    create_original_path_visualization(S_infinity, obstacle_indices, path_indices, 
                                       start_continuous, goal_continuous, safe_angle_count)
    
    # 5. 打印关键统计指标
    print(f"\n系统性能总结:")
    print(f"安全集收敛: {len(S_infinity):,} 个状态")
    print(f"平均鲁棒性: {np.mean(list(safe_angle_count.values())):.2f}/{max_angles} 角度" if safe_angle_count else "N/A")
    print(f"状态空间缩减: {len(S_infinity)/(len(x_space)*len(y_space)*len(theta_space))*100:.1f}% 保留")
    
    if comparison_results['robust']['stats']['success'] and comparison_results['graph_optimized']['stats']['success']:
        robust_time = comparison_results['robust']['stats']['computation_time']
        graph_time = comparison_results['graph_optimized']['stats']['computation_time']
        speedup = robust_time / graph_time if graph_time > 0 else float('inf')
        print(f"图优化搜索提速: {speedup:.1f}x")    
    print("分析完成")

if __name__ == "__main__":
    main()
