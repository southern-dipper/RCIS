"""
基于RPIS的安全RRT算法 - 快速测试脚本
用于参数调优和性能评估
"""

import numpy as np
import time
from RRT_rpis import *

def generate_random_start_goal(obstacle_indices, min_distance=1.5, S_infinity=None):
    """
    生成随机的起点和终点，确保它们：
    1. 不在障碍物上
    2. 在网格边界内
    3. 彼此间距离足够远
    4. 起点在安全集内（如果提供了安全集）
    
    Args:
        obstacle_indices: 障碍物位置集合
        min_distance: 起点和终点之间的最小距离
        S_infinity: 安全集（可选）
    
    Returns:
        start_continuous: 起点连续坐标 [x, y, theta]
        goal_continuous: 终点连续坐标 [x, y]
    """
    max_attempts = 1000
    
    for attempt in range(max_attempts):
        # 生成随机起点
        start_x = np.random.uniform(X_MIN + 0.5, X_MAX - 0.5)
        start_y = np.random.uniform(Y_MIN + 0.5, Y_MAX - 0.5)
        start_theta = np.random.uniform(-np.pi, np.pi)
        start_continuous = np.array([start_x, start_y, start_theta])
        
        # 生成随机终点
        goal_x = np.random.uniform(X_MIN + 0.5, X_MAX - 0.5)
        goal_y = np.random.uniform(Y_MIN + 0.5, Y_MAX - 0.5)
        goal_continuous = np.array([goal_x, goal_y])
        
        # 检查起点和终点是否在障碍物上
        start_indices = discretize_state(*start_continuous)
        goal_xy_indices = discretize_state(goal_continuous[0], goal_continuous[1], 0)[:2]
        
        start_pos = (start_indices[0], start_indices[1])
        goal_pos = (goal_xy_indices[0], goal_xy_indices[1])
        
        # 检查是否在障碍物上
        if start_pos in obstacle_indices or goal_pos in obstacle_indices:
            continue
            
        # 检查起点是否在安全集内（如果提供了安全集）
        if S_infinity is not None:
            if not is_state_in_safe_set(start_continuous, S_infinity):
                continue
            
        # 检查距离是否足够远
        distance = np.linalg.norm(np.array([start_x, start_y]) - np.array([goal_x, goal_y]))
        if distance >= min_distance:
            return start_continuous,np.array([5, 6.5])
    
    # 如果无法生成合适的起点和终点，使用默认值
    print("警告：无法生成随机起点和终点，使用默认值")
    return np.array([0., 0., np.pi / 4]), np.array([5, 6.5])

def quick_performance_test(num_runs=3, max_iterations=3000):
    """
    快速性能测试，多次运行获取统计数据
    """
    print("=" * 60)
    print("基于RPIS的安全RRT vs 基线RRT - 性能对比测试")
    print("=" * 60)
    bias=0.2 
    
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
    for i in range(0,4):
        obstacle_indices.add((i, 23))
    for i in range(1,4):
        obstacle_indices.add((i, 19))  
    
    for j in range(11, 16):
        obstacle_indices.add((7, j))  # 垂直障碍
    for j in range(3, 16):
        obstacle_indices.add((22, j))  # 垂直障碍    
    for j in range(4, 7):
        obstacle_indices.add((11, j))  # 垂直障碍 
    for j in range(19,24):
        obstacle_indices.add((4, j)) 
    
    print(f"测试设置:")
    print(f"  障碍物数量: {len(obstacle_indices)}")
    print(f"  最大迭代次数: {max_iterations}")
    print(f"  测试轮数: {num_runs}")
    print()
    
    # 计算安全集
    safe_set_start_time = time.time()
    goal_continuous = np.array([5, 6.5])
    S_infinity = compute_robust_safe_set_optimized(obstacle_indices, W)
    safe_set_time = time.time() - safe_set_start_time
    print(f"安全集计算完成: {len(S_infinity)} 个安全状态, 用时: {safe_set_time:.2f}秒")
    print()
      # 存储测试结果
    baseline_results = []
    safe_results = []
    
    print("开始多轮测试...")
    print("-" * 60)
    start_continuous = np.array([0., 0., np.pi / 4])  
    goal_continuous = np.array([10., 10.])
    start_indices = discretize_state(*start_continuous)
    goal_xy_indices = discretize_state(goal_continuous[0], goal_continuous[1], 0)[:2]
    for run in range(num_runs):        
        print(f"第 {run + 1} 轮测试:")
        # 测试基线RRT
        baseline_path, baseline_tree, baseline_time = rrt_search(
            start_indices, goal_xy_indices, S_infinity, obstacle_indices, 
            max_iterations=max_iterations, goal_tolerance=1.0,goal_bias_prob=bias
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
            max_iterations=max_iterations, goal_tolerance=1.0, goal_bias_prob=bias
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
    baseline_avg_tree_branches = np.mean([r['tree_branches'] for r in baseline_results if r['success']]) 
    
    # 安全RRT统计
    safe_success_rate = sum(1 for r in safe_results if r['success']) / num_runs
    safe_avg_time = np.mean([r['time'] for r in safe_results])
    safe_std_time = np.std([r['time'] for r in safe_results])
    safe_avg_path_length = np.mean([r['path_length'] for r in safe_results if r['success']]) if safe_success_rate > 0 else 0
    safe_avg_tree_nodes = np.mean([r['tree_nodes'] for r in safe_results])
    safe_avg_tree_branches = np.mean([r['tree_branches'] for r in safe_results if r['success']]) 
    
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
    for i in range(0,4):
        obstacle_indices.add((i, 23))
    for i in range(1,4):
        obstacle_indices.add((i, 19))  
    
    for j in range(11, 16):
        obstacle_indices.add((7, j))  # 垂直障碍
    for j in range(3, 16):
        obstacle_indices.add((22, j))  # 垂直障碍    
    for j in range(4, 7):
        obstacle_indices.add((11, j))  # 垂直障碍    for j in range(19,24):
        obstacle_indices.add((4, j))
    
    # 计算或复用安全集
    if S_infinity is None:
        print("计算安全集...")
        S_infinity = compute_robust_safe_set_optimized(obstacle_indices, W)
    else:
        print("复用已计算的安全集...")
    
    print(f"测试不同的目标偏向概率: {goal_bias_values}")
    print(f"每个参数测试使用随机生成的起点和终点")
    print()
    
    for goal_bias in goal_bias_values:
        print(f"测试目标偏向概率 = {goal_bias}")
        # 进行n轮测试
        success_count = 0
        total_time = 0
        n=10
        for run in range(n):
            # 为每轮测试生成随机的起点和终点
            start_continuous, goal_continuous = generate_random_start_goal(obstacle_indices, S_infinity=S_infinity)
            start_indices = discretize_state(*start_continuous)
            goal_xy_indices = discretize_state(goal_continuous[0], goal_continuous[1], 0)[:2]
            
            safe_path, safe_tree, safe_time = safe_rrt_search(
                start_indices, goal_xy_indices, S_infinity, obstacle_indices,
                max_iterations=5000, goal_tolerance=1.0, goal_bias_prob=goal_bias
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
    baseline_results, safe_results, S_infinity = quick_performance_test(num_runs=1000, max_iterations=2000)
    
    # 参数敏感性测试（复用安全集）
    #parameter_sensitivity_test(S_infinity)