"""
A*路径规划算法模块
"""
import heapq
import time
from tqdm import tqdm
from config import *
from core_models import *

def build_safe_state_graph(S_infinity, obstacle_indices):
    """
    预构建安全状态图：一次构建，多次查询
    
    Args:
        S_infinity: 鲁棒安全集
        obstacle_indices: 障碍物索引集合
    
    Returns:
        dict: {state_indices: [(neighbor_indices, omega), ...]}
              每个安全状态映射到其所有安全邻居及对应动作
    """
    print("构建安全状态图...")
    
    safe_graph = {}
    total_edges = 0
    
    for state_indices in tqdm(S_infinity, desc="构建图"):
        neighbors = []
        current_state = indices_to_state(*state_indices)
        
        for omega in omega_space:
            next_state = unicycle_model(current_state, omega)
            
            # 边界检查
            if not is_state_valid(next_state[0], next_state[1]):
                continue
                
            next_indices = discretize_state(*next_state)
            
            # 关键：只有当下一状态也在安全集内时才添加连接
            if next_indices in S_infinity:
                neighbors.append((next_indices, omega))
                total_edges += 1
        
        safe_graph[state_indices] = neighbors
    
    # 统计图的性质
    avg_degree = total_edges / len(safe_graph) if safe_graph else 0
    print(f"图构建完成: {len(safe_graph)}节点, {total_edges}边")
    
    return safe_graph

def a_star_on_safe_graph(start_indices, goal_xy_indices, safe_graph):
    """
    在预构建的安全图上进行超高速A*搜索 - 优化版本
    
    Args:
        start_indices: 起点状态索引
        goal_xy_indices: 目标xy位置索引
        safe_graph: 预构建的安全状态图
    
    Returns:
        tuple: (path, search_stats)
    """
    # 初始化统计信息
    search_stats = {
        'mode': '图优化A*',
        'nodes_expanded': 0,
        'nodes_in_open_set': 1,  # 起点已加入
        'nodes_rejected_by_bounds': 0,
        'nodes_rejected_by_safety': 0,  # 预构建图中此项为0
        'nodes_rejected_by_collision': 0,  # 预构建图中此项为0
        'path_length': 0,
        'success': False
    }
    
    if start_indices not in safe_graph:
        print("错误：起点不在安全图中！")
        return None, search_stats
    
    open_set = []
    counter = 0  # 用于打破堆中的平局
    heapq.heappush(open_set, (0, counter, start_indices))
    counter += 1
    
    came_from = {}
    g_score = {start_indices: 0}
    f_score = {start_indices: heuristic(start_indices, (*goal_xy_indices, 0))}
    closed_set = set()
    
    while open_set:
        _, _, current_indices = heapq.heappop(open_set)
        
        # 重要：如果已经处理过这个节点，跳过
        if current_indices in closed_set:
            continue
        
        # 标记为已处理
        closed_set.add(current_indices)
        search_stats['nodes_expanded'] += 1
        
        # 目标检测
        if (current_indices[0], current_indices[1]) == goal_xy_indices:
            print("成功找到路径！")
            path = []
            while current_indices in came_from:
                path.append(current_indices)
                current_indices = came_from[current_indices]
            path.append(start_indices)
            search_stats['path_length'] = len(path)
            search_stats['success'] = True
            return path[::-1], search_stats
        
        # 核心优化：只处理预构建图中的直接邻居
        for neighbor_indices, omega in safe_graph[current_indices]:
            # 跳过已经完全处理过的邻居
            if neighbor_indices in closed_set:
                continue
                
            tentative_g_score = g_score[current_indices] + 1
            
            # 只有在找到更好路径时才更新
            if neighbor_indices not in g_score or tentative_g_score < g_score[neighbor_indices]:
                came_from[neighbor_indices] = current_indices
                g_score[neighbor_indices] = tentative_g_score
                f_score[neighbor_indices] = tentative_g_score + heuristic(neighbor_indices, (*goal_xy_indices, 0))
                heapq.heappush(open_set, (f_score[neighbor_indices], counter, neighbor_indices))
                counter += 1
                search_stats['nodes_in_open_set'] += 1
    
    print("未能找到路径。")
    return None, search_stats

def get_next_state_indices_for_astar(state_indices, omega):
    """A*使用确定性模型（无扰动）进行路径搜索"""
    current_state = indices_to_state(*state_indices)
    next_continuous_state = unicycle_model(current_state, omega)
    
    # 检查边界
    if not is_state_valid(next_continuous_state[0], next_continuous_state[1]):
        return None
    
    next_indices = discretize_state(*next_continuous_state)
    return next_indices

def a_star_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, use_robust_constraints=True):
    """
    A*搜索算法，支持鲁棒约束和基线模式对比
    
    Args:
        start_indices: 起点离散索引
        goal_xy_indices: 目标点xy索引
        S_infinity: 鲁棒安全集
        obstacle_indices: 障碍物索引
        use_robust_constraints: 是否使用鲁棒安全集约束 (True=鲁棒A*, False=基线A*)
    
    Returns:
        tuple: (path, search_stats) 包含路径和搜索统计信息
    """
    mode_name = "鲁棒A*" if use_robust_constraints else "基线A*"
    
    # 初始化统计信息
    search_stats = {
        'mode': mode_name,
        'nodes_expanded': 0,
        'nodes_in_open_set': 0,
        'nodes_rejected_by_bounds': 0,
        'nodes_rejected_by_safety': 0,
        'nodes_rejected_by_collision': 0,
        'path_length': 0,
        'success': False
    }
    
    # 鲁棒模式下检查起点安全性
    if use_robust_constraints and start_indices not in S_infinity:
        print("错误：起点不在鲁棒安全集内！")
        return None, search_stats

    open_set = []
    heapq.heappush(open_set, (0, start_indices))
    
    came_from = {}
    g_score = {start_indices: 0}
    f_score = {start_indices: heuristic(start_indices, (*goal_xy_indices, 0))}

    while open_set:
        _, current_indices = heapq.heappop(open_set)
        search_stats['nodes_expanded'] += 1
        
        if (current_indices[0], current_indices[1]) == goal_xy_indices:
            print("成功找到路径！")
            path = []
            while current_indices in came_from:
                path.append(current_indices)
                current_indices = came_from[current_indices]
            path.append(start_indices)
            search_stats['path_length'] = len(path)
            search_stats['success'] = True
            return path[::-1], search_stats

        for omega in omega_space:
            neighbor_indices = get_next_state_indices_for_astar(current_indices, omega)
            
            # 边界检查（两种模式都需要）
            if neighbor_indices is None:
                search_stats['nodes_rejected_by_bounds'] += 1
                continue
            
            # 安全集约束检查（仅鲁棒模式）
            if use_robust_constraints:
                if neighbor_indices not in S_infinity:
                    search_stats['nodes_rejected_by_safety'] += 1
                    continue
            
            # 路径碰撞检查（两种模式都需要）
            current_state = indices_to_state(*current_indices)
            neighbor_state = indices_to_state(*neighbor_indices)
            if check_path_collision(current_state, neighbor_state, obstacle_indices):
                search_stats['nodes_rejected_by_collision'] += 1
                continue

            tentative_g_score = g_score[current_indices] + 1
            
            if neighbor_indices not in g_score or tentative_g_score < g_score[neighbor_indices]:
                came_from[neighbor_indices] = current_indices
                g_score[neighbor_indices] = tentative_g_score
                f_score[neighbor_indices] = tentative_g_score + heuristic(neighbor_indices, (*goal_xy_indices, 0))
                heapq.heappush(open_set, (f_score[neighbor_indices], neighbor_indices))
                search_stats['nodes_in_open_set'] += 1

    print("未能找到路径。")
    return None, search_stats

def compare_astar_methods(start_indices, goal_xy_indices, S_infinity, obstacle_indices):
    """
    对比三种A*方法的性能：基线A*、当前鲁棒A*、图优化A*
    
    Returns:
        dict: 包含三种方法的结果和统计对比
    """
    print("A*算法性能对比...")
    
    results = {}
    
    # 1. 运行基线A*
    start_time = time.time()
    baseline_path, baseline_stats = a_star_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, use_robust_constraints=False)
    baseline_time = time.time() - start_time
    baseline_stats['computation_time'] = baseline_time
    results['baseline'] = {'path': baseline_path, 'stats': baseline_stats}
    
    # 2. 运行当前鲁棒A*
    start_time = time.time()
    robust_path, robust_stats = a_star_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, use_robust_constraints=True)
    robust_time = time.time() - start_time
    robust_stats['computation_time'] = robust_time
    results['robust'] = {'path': robust_path, 'stats': robust_stats}
    
    # 3. 预构建安全图
    graph_build_start = time.time()
    safe_graph = build_safe_state_graph(S_infinity, obstacle_indices)
    graph_build_time = time.time() - graph_build_start
    
    # 计算图的边数
    graph_edges = sum(len(neighbors) for neighbors in safe_graph.values())
    
    # 4. 运行图优化A*
    start_time = time.time()
    graph_path, graph_stats = a_star_on_safe_graph(start_indices, goal_xy_indices, safe_graph)
    graph_search_time = time.time() - start_time
    graph_stats['computation_time'] = graph_search_time
    graph_stats['graph_build_time'] = graph_build_time
    graph_stats['total_time'] = graph_build_time + graph_search_time
    results['graph_optimized'] = {'path': graph_path, 'stats': graph_stats}
    
    # 添加内存计算所需的额外信息
    results['safe_set'] = S_infinity
    results['graph_edges_count'] = graph_edges
    
    return results
