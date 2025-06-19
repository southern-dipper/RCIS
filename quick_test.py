"""
基于RPIS的安全RRT算法 - 快速测试脚本
用于参数调优和性能评估
"""

import numpy as np
import time
from RRT_rpis import *

def quick_performance_test(num_runs=3, max_iterations=500):
    """
    快速性能测试，多次运行获取统计数据
    """
    print("=" * 60)
    print("基于RPIS的安全RRT vs 基线RRT - 性能对比测试")
    print("=" * 60)
    
    # 测试参数
    start_continuous = np.array([0., 0., np.pi / 4])  
    goal_continuous = np.array([10., 10.])
    
    start_indices = discretize_state(*start_continuous)
    goal_xy_indices = discretize_state(goal_continuous[0], goal_continuous[1], 0)[:2]

    # 生成障碍物
    obstacle_indices = set()
    for i in range(8, 22):
        obstacle_indices.add((i, 15))  # 水平障碍
    for i in range(8, 18):
        obstacle_indices.add((i, 11)) 
    for i in range(11, 19):
        obstacle_indices.add((i, 7))
    for i in range(11, 22):
        obstacle_indices.add((i, 3))  
    
    for j in range(12, 15):
        obstacle_indices.add((7, j))  # 垂直障碍
    for j in range(4, 15):
        obstacle_indices.add((22, j))  # 垂直障碍    
    for j in range(4, 7):
        obstacle_indices.add((11, j))  # 垂直障碍 
    
    print(f"测试设置:")
    print(f"  起点: {start_continuous}")
    print(f"  目标: {goal_continuous}")
    print(f"  障碍物数量: {len(obstacle_indices)}")
    print(f"  最大迭代次数: {max_iterations}")
    print(f"  测试轮数: {num_runs}")
    print()
    
    # 计算安全集（只需计算一次）
    print("计算鲁棒安全集...")
    safe_set_start_time = time.time()
    S_infinity = compute_robust_safe_set_optimized(obstacle_indices, W)
    safe_set_time = time.time() - safe_set_start_time
    print(f"安全集计算完成: {len(S_infinity)} 个安全状态, 用时: {safe_set_time:.2f}秒")
    print()
      # 存储测试结果
    baseline_results = []
    safe_results = []
    
    print("开始多轮测试...")
    print("-" * 60)
    
    for run in range(num_runs):
        print(f"第 {run + 1} 轮测试:")
        
        # 测试基线RRT
        baseline_path, baseline_tree, baseline_time = rrt_search(
            start_indices, goal_xy_indices, S_infinity, obstacle_indices, 
            max_iterations=max_iterations, goal_tolerance=1.0
        )
        
        baseline_success = baseline_path is not None
        baseline_path_length = len(baseline_path) if baseline_success else 0
        baseline_tree_nodes = len(baseline_tree)
        baseline_tree_branches = count_tree_branches(baseline_tree) if baseline_success else 0
        
        baseline_results.append({
            'success': baseline_success,
            'time': baseline_time,
            'path_length': baseline_path_length,
            'tree_nodes': baseline_tree_nodes,
            'tree_branches': baseline_tree_branches
        })
          # 测试安全RRT
        safe_path, safe_tree, safe_time = safe_rrt_search(
            start_indices, goal_xy_indices, S_infinity, obstacle_indices,
            max_iterations=max_iterations, goal_tolerance=1.0
        )
        
        safe_success = safe_path is not None
        safe_path_length = len(safe_path) if safe_success else 0
        safe_tree_nodes = len(safe_tree)
        safe_tree_branches = count_tree_branches(safe_tree) if safe_success else 0
        
        safe_results.append({
            'success': safe_success,
            'time': safe_time,
            'path_length': safe_path_length,
            'tree_nodes': safe_tree_nodes,
            'tree_branches': safe_tree_branches
        })
          # 输出本轮结果
        print(f"    基线RRT: {'成功' if baseline_success else '失败'}, "
              f"时间: {baseline_time:.3f}s, 路径节点数: {baseline_path_length}, "
              f"树节点: {baseline_tree_nodes}, 分支数: {baseline_tree_branches}")
        print(f"    安全RRT: {'成功' if safe_success else '失败'}, "
              f"时间: {safe_time:.3f}s, 路径节点数: {safe_path_length}, "
              f"树节点: {safe_tree_nodes}, 分支数: {safe_tree_branches}")
        print()
    
    # 统计分析
    print("=" * 60)
    print("统计分析结果:")
    print("=" * 60)
    
    # 基线RRT统计
    baseline_success_rate = sum(1 for r in baseline_results if r['success']) / num_runs
    baseline_avg_time = np.mean([r['time'] for r in baseline_results])
    baseline_std_time = np.std([r['time'] for r in baseline_results])
    baseline_avg_path_length = np.mean([r['path_length'] for r in baseline_results if r['success']]) if baseline_success_rate > 0 else 0
    baseline_avg_tree_nodes = np.mean([r['tree_nodes'] for r in baseline_results])
    baseline_avg_tree_branches = np.mean([r['tree_branches'] for r in baseline_results if r['success']]) if baseline_success_rate > 0 else 0
    
    # 安全RRT统计
    safe_success_rate = sum(1 for r in safe_results if r['success']) / num_runs
    safe_avg_time = np.mean([r['time'] for r in safe_results])
    safe_std_time = np.std([r['time'] for r in safe_results])
    safe_avg_path_length = np.mean([r['path_length'] for r in safe_results if r['success']]) if safe_success_rate > 0 else 0
    safe_avg_tree_nodes = np.mean([r['tree_nodes'] for r in safe_results])
    safe_avg_tree_branches = np.mean([r['tree_branches'] for r in safe_results if r['success']]) if safe_success_rate > 0 else 0
    
    print(f"基线RRT:")
    print(f"  成功率: {baseline_success_rate:.1%} ({sum(1 for r in baseline_results if r['success'])}/{num_runs})")
    print(f"  平均搜索时间: {baseline_avg_time:.3f}±{baseline_std_time:.3f}秒")
    print(f"  平均路径节点数: {baseline_avg_path_length:.1f} 个节点")
    print(f"  平均树节点数: {baseline_avg_tree_nodes:.1f}")
    print(f"  平均树分支数: {baseline_avg_tree_branches:.1f}")
    print()
    
    print(f"安全RRT:")
    print(f"  成功率: {safe_success_rate:.1%} ({sum(1 for r in safe_results if r['success'])}/{num_runs})")
    print(f"  平均搜索时间: {safe_avg_time:.3f}±{safe_std_time:.3f}秒")
    print(f"  平均路径节点数: {safe_avg_path_length:.1f} 个节点")
    print(f"  平均树节点数: {safe_avg_tree_nodes:.1f}")
    print(f"  平均树分支数: {safe_avg_tree_branches:.1f}")
    print()
    
    # 对比分析
    print("对比分析:")
    if baseline_success_rate > 0 and safe_success_rate > 0:
        time_ratio = safe_avg_time / baseline_avg_time
        path_ratio = safe_avg_path_length / baseline_avg_path_length if baseline_avg_path_length > 0 else float('inf')
        tree_ratio = safe_avg_tree_nodes / baseline_avg_tree_nodes if baseline_avg_tree_nodes > 0 else float('inf')
        
        print(f"  搜索时间比值 (安全RRT/基线RRT): {time_ratio:.2f}x")
        print(f"  路径节点数比值 (安全RRT/基线RRT): {path_ratio:.2f}x")
        print(f"  树节点比值 (安全RRT/基线RRT): {tree_ratio:.2f}x")
    else:
        print("  无法进行对比分析（某个算法成功率为0）")
    
    print(f"  安全集计算时间: {safe_set_time:.2f}秒")
    print()
    
    return baseline_results, safe_results, S_infinity

def parameter_sensitivity_test(S_infinity=None):  
    """
    参数敏感性测试
    """
    print("=" * 60)
    print("参数敏感性测试")
    print("=" * 60)
    
    # 测试不同的目标偏向概率
    goal_bias_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # 测试参数
    start_continuous = np.array([0., 0., np.pi / 4])  
    goal_continuous = np.array([10., 10.])
    
    start_indices = discretize_state(*start_continuous)
    goal_xy_indices = discretize_state(goal_continuous[0], goal_continuous[1], 0)[:2]    # 生成障碍物
    obstacle_indices = set()
    for i in range(4, 16):
        obstacle_indices.add((i, 18))
    for i in range(6, 16):
        obstacle_indices.add((i, 7)) 
    for j in range(8, 19):
        obstacle_indices.add((18, j))
    
    # 计算或复用安全集
    if S_infinity is None:
        print("计算安全集...")
        S_infinity = compute_robust_safe_set_optimized(obstacle_indices, W)
    else:
        print("复用已计算的安全集...")
    
    print(f"测试不同的目标偏向概率: {goal_bias_values}")
    print()
    
    for goal_bias in goal_bias_values:
        print(f"测试目标偏向概率 = {goal_bias}")
          # 进行n轮测试
        success_count = 0
        total_time = 0
        n=10
        for run in range(n):
            safe_path, safe_tree, safe_time = safe_rrt_search(
                start_indices, goal_xy_indices, S_infinity, obstacle_indices,
                max_iterations=500, goal_tolerance=1.0, goal_bias_prob=goal_bias
            )
            
            if safe_path is not None:
                success_count += 1
            total_time += safe_time
        
        success_rate = success_count / n
        avg_time = total_time / n
        
        print(f"  成功率: {success_rate:.1%}, 平均时间: {avg_time:.3f}秒")
        print()

if __name__ == "__main__":
    # 快速性能测试
    baseline_results, safe_results, S_infinity = quick_performance_test(num_runs=500, max_iterations=1000)
    
    # 参数敏感性测试（复用安全集）
    parameter_sensitivity_test(S_infinity)