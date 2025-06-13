"""
性能分析和结果展示模块
"""
import numpy as np
import time
from config import *

def generate_performance_metrics(results):
    """生成性能指标"""
    metrics = {}
    
    # 计算效率指标
    for method_name, result in results.items():
        # 跳过非算法结果的键
        if method_name in ['safe_set', 'graph_edges_count']:
            continue
            
        stats = result['stats']
          # 包含搜索时间、路径长度和分支数指标
        metrics[method_name] = {
            'search_time_ms': stats['computation_time'] * 1000,
            'path_length': stats.get('path_length', 0) if stats['success'] else float('inf'),
            'avg_branches_per_node': stats.get('avg_branches_per_node', 0),
            'total_branches_explored': stats.get('total_branches_explored', 0),
            'nodes_expanded': stats.get('nodes_expanded', 0)
        }
    
    return metrics

def print_academic_results_table(metrics):
    """打印结果表格"""
    print("\n" + "="*85)
    print("标准A* vs 安全A* 性能对比")
    print("="*85)
    
    print(f"{'算法':<12} {'搜索时间(ms)':<15} {'平均分支数':<12} {'扩展节点数':<12} {'路径长度':<10}")
    print("-" * 85)
    
    for method in ['baseline', 'graph_optimized']:
        if method in metrics:
            data = metrics[method]
            method_name = {'baseline': '标准A*', 'graph_optimized': '安全A*'}[method]
            path_len = data['path_length'] if data['path_length'] != float('inf') else 'N/A'
            print(f"{method_name:<12} {data['search_time_ms']:<15.2f} {data['avg_branches_per_node']:<12.2f} {data['nodes_expanded']:<12} {path_len:<10}")
    
    # 计算提升倍数
    if 'baseline' in metrics and 'graph_optimized' in metrics:
        baseline_time = metrics['baseline']['search_time_ms']
        graph_time = metrics['graph_optimized']['search_time_ms']
        baseline_branches = metrics['baseline']['avg_branches_per_node']
        graph_branches = metrics['graph_optimized']['avg_branches_per_node']
        
        print("\n" + "="*85)
        print("性能提升分析:")
        if graph_time > 0:
            speedup = baseline_time / graph_time
            print(f"• 搜索速度提升: 安全A*比标准A*快 {speedup:.1f} 倍")
        
        if baseline_branches > 0 and graph_branches > 0:
            branch_reduction = (baseline_branches - graph_branches) / baseline_branches * 100
            print(f"• 分支数减少: {branch_reduction:.1f}% (从 {baseline_branches:.2f} 减少到 {graph_branches:.2f})")
            print(f"• 分支效率提升: {baseline_branches/graph_branches:.1f} 倍")
        
        baseline_total_branches = metrics['baseline']['total_branches_explored']
        graph_total_branches = metrics['graph_optimized']['total_branches_explored']
        if baseline_total_branches > 0:
            total_branch_reduction = (baseline_total_branches - graph_total_branches) / baseline_total_branches * 100
            print(f"• 总分支探索减少: {total_branch_reduction:.1f}% ({baseline_total_branches} → {graph_total_branches})")
        
        print("="*85)


