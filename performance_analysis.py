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
        
        # 只保留搜索时间指标
        metrics[method_name] = {
            'search_time_ms': stats['computation_time'] * 1000,
            'path_length': stats.get('path_length', 0) if stats['success'] else float('inf')
        }
    
    return metrics

def print_academic_results_table(metrics):
    """打印结果表格"""
    print("\n" + "="*60)
    print("标准A* vs 安全图A* 性能对比")
    print("="*60)
    
    print(f"{'算法':<12} {'搜索时间(ms)':<15} {'路径长度':<10}")
    print("-" * 60)
    
    for method in ['baseline', 'graph_optimized']:
        if method in metrics:
            data = metrics[method]
            method_name = {'baseline': '标准A*', 'graph_optimized': '安全图A*'}[method]
            path_len = data['path_length'] if data['path_length'] != float('inf') else 'N/A'
            print(f"{method_name:<12} {data['search_time_ms']:<15.2f} {path_len:<10}")
    
    # 计算提升倍数
    if 'baseline' in metrics and 'graph_optimized' in metrics:
        baseline_time = metrics['baseline']['search_time_ms']
        graph_time = metrics['graph_optimized']['search_time_ms']
        if graph_time > 0:
            speedup = baseline_time / graph_time
            print(f"\n性能提升: 安全图A*比标准A*快 {speedup:.1f} 倍")


