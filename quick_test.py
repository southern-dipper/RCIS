#!/usr/bin/env python3
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
    for i in range(4, 16):
        obstacle_indices.add((i, 18))
    for i in range(6, 16):
        obstacle_indices.add((i, 7)) 
    for j in range(8, 19):
        obstacle_indices.add((18, j))
    obstacle_indices.add((4, 16))
    obstacle_indices.add((3, 17))
    obstacle_indices.add((5, 17))
    
    print(f"环境设置: 20x25网格, {len(obstacle_indices)}个障碍物格子")
    
    # 计算安全集
    print("计算RPIS安全集...")
    start_time = time.time()
    S_infinity, safe_actions_map = compute_robust_safe_set_optimized(obstacle_indices, W)
    rpis_time = time.time() - start_time
    
    if not S_infinity:
        print("❌ 鲁棒安全集为空，测试终止")
        return
    
    print(f"✅ 安全集计算完成: {len(S_infinity)}个状态, 耗时: {rpis_time:.2f}秒")
    
    # 存储测试结果
    baseline_results = []
    safe_results = []
    
    print(f"\n开始 {num_runs} 轮测试 (最大迭代数: {max_iterations})...")
    
    for run in range(num_runs):
        print(f"\n--- 第 {run+1}/{num_runs} 轮测试 ---")
        
        # 基线RRT测试
        print("🔄 基线RRT...")
        start_time = time.time()
        rrt_result = rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, 
                               safe_actions_map, max_iterations=max_iterations, goal_tolerance=1.0)
        baseline_time = time.time() - start_time
        
        rrt_path, rrt_tree = rrt_result
        if rrt_path:
            baseline_results.append({
                'success': True,
                'time': baseline_time,
                'path_length': len(rrt_path),
                'tree_nodes': len(rrt_tree),
                'tree_branches': count_tree_branches(rrt_tree)
            })
            print(f"✅ 成功 - 时间: {baseline_time:.2f}s, 路径长度: {len(rrt_path)}, 树节点: {len(rrt_tree)}")
        else:
            baseline_results.append({'success': False, 'time': baseline_time})
            print(f"❌ 失败 - 时间: {baseline_time:.2f}s")
        
        # 安全RRT测试
        print("🔄 安全RRT...")
        start_time = time.time()
        safe_rrt_result = safe_rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices,
                                         safe_actions_map, max_iterations=max_iterations, goal_tolerance=1.0)
        safe_time = time.time() - start_time
        
        safe_path, safe_tree = safe_rrt_result
        if safe_path:
            safe_results.append({
                'success': True,
                'time': safe_time,
                'path_length': len(safe_path),
                'tree_nodes': len(safe_tree),
                'tree_branches': count_tree_branches(safe_tree)
            })
            print(f"✅ 成功 - 时间: {safe_time:.2f}s, 路径长度: {len(safe_path)}, 树节点: {len(safe_tree)}")
        else:
            safe_results.append({'success': False, 'time': safe_time})
            print(f"❌ 失败 - 时间: {safe_time:.2f}s")
    
    # 统计分析
    print("\n" + "=" * 60)
    print("测试结果统计")
    print("=" * 60)
    
    # 基线RRT统计
    baseline_success = [r for r in baseline_results if r['success']]
    safe_success = [r for r in safe_results if r['success']]
    
    print(f"基线RRT成功率: {len(baseline_success)}/{num_runs} ({len(baseline_success)/num_runs*100:.1f}%)")
    print(f"安全RRT成功率: {len(safe_success)}/{num_runs} ({len(safe_success)/num_runs*100:.1f}%)")
    
    if baseline_success and safe_success:
        print("\n平均性能对比:")
        
        baseline_avg_time = np.mean([r['time'] for r in baseline_success])
        safe_avg_time = np.mean([r['time'] for r in safe_success])
        
        baseline_avg_path = np.mean([r['path_length'] for r in baseline_success])
        safe_avg_path = np.mean([r['path_length'] for r in safe_success])
        
        baseline_avg_nodes = np.mean([r['tree_nodes'] for r in baseline_success])
        safe_avg_nodes = np.mean([r['tree_nodes'] for r in safe_success])
        
        print(f"平均运行时间 - 基线: {baseline_avg_time:.2f}s, 安全: {safe_avg_time:.2f}s ({safe_avg_time/baseline_avg_time:.2f}x)")
        print(f"平均路径长度 - 基线: {baseline_avg_path:.1f}, 安全: {safe_avg_path:.1f} ({safe_avg_path/baseline_avg_path:.2f}x)")
        print(f"平均树节点数 - 基线: {baseline_avg_nodes:.1f}, 安全: {safe_avg_nodes:.1f} ({safe_avg_nodes/baseline_avg_nodes:.2f}x)")
        
        # 计算性能比值
        time_ratio = safe_avg_time / baseline_avg_time
        if time_ratio <= 1.2:
            print("🎉 安全RRT性能优秀 (时间开销 <= 20%)")
        elif time_ratio <= 2.0:
            print("✅ 安全RRT性能可接受 (时间开销 <= 100%)")
        else:
            print("⚠️  安全RRT性能需要优化 (时间开销 > 100%)")
    
    print("\n测试完成！")

def parameter_sensitivity_test():
    """
    参数敏感性测试 - 修正距离阈值
    """
    print("\n" + "=" * 60)
    print("安全RRT参数敏感性测试")
    print("=" * 60)
    
    # 测试不同的修正距离阈值
    thresholds = [1.0, 1.2, 1.5, 2.0, 2.5]
    
    # 这里可以实现参数调优逻辑
    print("参数敏感性测试功能待实现...")
    print("计划测试的修正距离阈值:", thresholds)

if __name__ == "__main__":
    # 快速性能测试
    quick_performance_test(num_runs=3, max_iterations=800)
    
    # 参数敏感性测试
    # parameter_sensitivity_test()
