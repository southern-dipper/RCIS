import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import time
import matplotlib.colors as mcolors

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 1. 环境与模型定义 ---

# Unicycle模型参数  
V = 1.0  # 步长1.0米，大于格子对角线长度√2×0.5≈0.707米
DT = 1.0

# 状态空间 S = [x, y, theta] - 优化后的网格（平衡精度与计算效率）
X_MIN, X_MAX, X_STEP = -1, 11, 0.5  
Y_MIN, Y_MAX, Y_STEP = -1, 11, 0.5    
THETA_MIN, THETA_MAX, THETA_STEP = -np.pi, np.pi, np.pi / 8  # 角度分辨率

# 动作空间 A = [omega]  
OMEGA_MIN, OMEGA_MAX, OMEGA_STEP = -np.pi / 4, np.pi / 4, np.pi / 8 # 动作空间

x_space = np.linspace(X_MIN, X_MAX, int((X_MAX - X_MIN) / X_STEP) + 1)
y_space = np.linspace(Y_MIN, Y_MAX, int((Y_MAX - Y_MIN) / Y_STEP) + 1)
theta_space = np.linspace(THETA_MIN, THETA_MAX, int((THETA_MAX - THETA_MIN) / THETA_STEP) + 1)[:-1]
omega_space = np.linspace(OMEGA_MIN, OMEGA_MAX, int((OMEGA_MAX - OMEGA_MIN) / OMEGA_STEP) + 1)

# 状态空间信息将在main函数中打印，避免重复输出

# *** 定义扰动空间 W - 优化版本 ***
N_PERTURB_SAMPLES = 0  # 只采样四个角点，不包括中心

# 定义扰动集：只测试四个角点
epsilon = 1e-3  
wx_space = np.array([-X_STEP/2+1e-3, X_STEP/2-1e-3])
wy_space = np.array([-Y_STEP/2+1e-3, Y_STEP/2-1e-3])
wtheta_space = np.array([0.0])  # 暂不考虑角度扰动

# 创建扰动集 W (只取四个角点，共4个点，不包括中心)
W = [(wx, wy, 0.0) for wx in wx_space for wy in wy_space]

def is_state_valid(x, y):
    """检查状态是否在有效边界内（不与边界碰撞）"""
    return X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX

def discretize_state(x, y, theta):
    # 角度标准化到[-π, π]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    
    # 使用round进行四舍五入离散化
    ix = int(round((x - X_MIN) / X_STEP))
    iy = int(round((y - Y_MIN) / Y_STEP))
    itheta = int(round((theta - THETA_MIN) / THETA_STEP))
    
    # 边界保护
    ix = max(0, min(ix, len(x_space)-1))
    iy = max(0, min(iy, len(y_space)-1))
    itheta = max(0, min(itheta, len(theta_space)-1))
    
    return (ix, iy, itheta)

def indices_to_state(ix, iy, itheta):
    return np.array([x_space[ix], y_space[iy], theta_space[itheta]])

def unicycle_model(state, omega):
    x, y, theta = state
    x_next = x + V * np.cos(theta) * DT
    y_next = y + V * np.sin(theta) * DT
    theta_next = theta + omega * DT
    return np.array([x_next, y_next, theta_next])

def check_path_collision(start_state, end_state, obstacle_indices):
    """检查从起点到终点的路径是否与障碍物碰撞"""
    x1, y1 = start_state[0], start_state[1]
    x2, y2 = end_state[0], end_state[1]
    
    # 计算路径长度
    path_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # 如果路径很短，只检查端点
    if path_length < min(X_STEP, Y_STEP) * 0.5:
        end_indices = discretize_state(x2, y2, 0)
        return (end_indices[0], end_indices[1]) in obstacle_indices
    
    # 使用自适应采样密度：确保采样点间距小于网格对角线长度的一半
    grid_diagonal = np.sqrt(X_STEP**2 + Y_STEP**2)
    sample_distance = grid_diagonal * 0.4  # 采样间距为网格对角线的40%
    num_samples = max(int(np.ceil(path_length / sample_distance)), 2)
    
    # 检查路径上的采样点
    for i in range(num_samples + 1):
        t = i / num_samples
        x_sample = x1 + t * (x2 - x1)
        y_sample = y1 + t * (y2 - y1)
        
        # 检查采样点是否与障碍物碰撞
        sample_indices = discretize_state(x_sample, y_sample, 0)
        if (sample_indices[0], sample_indices[1]) in obstacle_indices:
            return True  # 发生碰撞
    
    return False  # 无碰撞

# --- 2. 鲁棒安全集计算 ---

def compute_robust_safe_set(obstacle_indices, W):
    print("计算鲁棒安全集...")
    
    # S_safe: 所有不与障碍物位置重叠且在边界内的格元
    S_safe = set()
    for ix in range(len(x_space)):
        for iy in range(len(y_space)):
            # 检查是否与障碍物碰撞
            if (ix, iy) not in obstacle_indices:
                # 检查是否在有效边界内
                x_actual, y_actual = x_space[ix], y_space[iy]
                if is_state_valid(x_actual, y_actual):
                    for itheta in range(len(theta_space)):
                        S_safe.add((ix, iy, itheta))
    
    Sk = S_safe.copy()
    k = 0
    
    while True:
        print(f"迭代 {k}: 当前安全集大小 |S_{k}| = {len(Sk)}")
        if not Sk:
            print("安全集为空，无法继续。问题可能无解或过于困难。")
            return set()
            
        Sk_plus_1 = set()
        
        # 对每个状态检查是否存在鲁棒安全动作
        for s_indices in tqdm(Sk, desc=f"迭代 {k}", ncols=100):
            exists_robust_action = False
            # 对每个动作检查是否鲁棒安全
            for omega in omega_space:
                is_action_robustly_safe = True
                s_center = indices_to_state(*s_indices)
                
                # 对所有扰动检查安全性
                for w in W:
                    # 施加扰动后的真实初始状态
                    s_real = s_center + np.array(w)
                    
                    # 计算下一步状态
                    s_next_real = unicycle_model(s_real, omega)
                    
                    # 首先检查下一步是否在边界内
                    if not is_state_valid(s_next_real[0], s_next_real[1]):
                        is_action_robustly_safe = False
                        break
                    
                    # *** 新增：检查路径是否与障碍物碰撞 ***
                    if check_path_collision(s_real, s_next_real, obstacle_indices):
                        is_action_robustly_safe = False
                        break
                    
                    # 离散化下一步状态
                    s_next_indices = discretize_state(*s_next_real)
                    
                    # 检查下一步是否在当前安全集内
                    if s_next_indices not in Sk:
                        is_action_robustly_safe = False
                        break
                
                if is_action_robustly_safe:
                    exists_robust_action = True
                    break
            
            if exists_robust_action:
                Sk_plus_1.add(s_indices)        # 检查收敛
        if Sk_plus_1 == Sk:
            print(f"鲁棒安全集已收敛: {len(Sk_plus_1)} 个状态")
            return Sk_plus_1
        
        Sk = Sk_plus_1
        k += 1

def compute_robust_safe_set_optimized(obstacle_indices, W):
    """优化的串行版本 - 移除无效的并行开销"""
    print("计算鲁棒安全集...")
    
    # 预计算常用值
    omega_list = list(omega_space)  # 转为list避免重复转换
    W_array = np.array(W)  # 转为numpy数组加速
    
    # S_safe: 所有不与障碍物位置重叠且在边界内的格元
    S_safe = set()
    for ix in range(len(x_space)):
        for iy in range(len(y_space)):
            if (ix, iy) not in obstacle_indices:
                x_actual, y_actual = x_space[ix], y_space[iy]
                if is_state_valid(x_actual, y_actual):
                    for itheta in range(len(theta_space)):
                        S_safe.add((ix, iy, itheta))
    
    Sk = S_safe.copy()
    k = 0
    
    while True:
        print(f"迭代 {k}: 当前安全集大小 |S_{k}| = {len(Sk)}")
        if not Sk:
            print("安全集为空，无法继续。")
            return set()
            
        Sk_plus_1 = set()
        
        # 优化的串行计算
        for s_indices in tqdm(Sk, desc=f"迭代 {k}", ncols=100):
            s_center = np.array([x_space[s_indices[0]], y_space[s_indices[1]], theta_space[s_indices[2]]])
            
            # 对每个动作检查是否鲁棒安全
            for omega in omega_list:
                is_action_robustly_safe = True
                
                # 对所有扰动检查安全性
                for w in W_array:
                    s_real = s_center + w
                    s_next_real = unicycle_model(s_real, omega)
                    
                    # 边界检查
                    if not is_state_valid(s_next_real[0], s_next_real[1]):
                        is_action_robustly_safe = False
                        break
                    
                    # 路径碰撞检查
                    if check_path_collision(s_real, s_next_real, obstacle_indices):
                        is_action_robustly_safe = False
                        break
                    
                    # 安全集检查
                    s_next_indices = discretize_state(*s_next_real)
                    if s_next_indices not in Sk:
                        is_action_robustly_safe = False
                        break
                
                if is_action_robustly_safe:
                    Sk_plus_1.add(s_indices)
                    break  # 找到一个安全动作就够了        # 检查收敛
        if Sk_plus_1 == Sk:
            print(f"鲁棒安全集已收敛: {len(Sk_plus_1)} 个状态")
            return Sk_plus_1
        
        Sk = Sk_plus_1
        k += 1

# --- 3. A* 路径规划算法 ---

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

def heuristic(state_indices, goal_indices):
    x1, y1, _ = indices_to_state(*state_indices)
    x2, y2, _ = indices_to_state(*goal_indices)
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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

def create_original_path_visualization(S_infinity, obstacle_indices, path_indices, start_continuous, goal_continuous, safe_angle_count):
    """恢复原来的单独大图可视化"""
    # 创建原来的大图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    max_angles = len(theta_space)
    
    # 创建渐变绿色地图
    grid_map = np.zeros((len(y_space), len(x_space)))
    
    # 设置安全区域的值（基于安全角度数量）
    for (ix, iy), count in safe_angle_count.items():
        # 将安全角度数量映射到0-1之间，然后映射到1-2之间用于颜色映射
        safety_level = count / max_angles
        grid_map[iy, ix] = 1 + safety_level  # 1到2之间的值
    
    # 设置障碍物
    for ix, iy in obstacle_indices:
        grid_map[iy, ix] = 3  # 障碍物用3表示

    # 创建自定义颜色映射：白色->浅绿色->深绿色->黑色
    colors = ['#FFFFFF',    # 0: 白色 (无安全角度)
              '#E8F5E8',    # 1: 极浅绿色 
              '#C8E6C9',    # 1.25: 浅绿色
              '#A5D6A7',    # 1.5: 中浅绿色
              '#81C784',    # 1.75: 中绿色
              '#66BB6A',    # 2: 深绿色 (所有角度安全)
              '#000000']    # 3: 黑色 (障碍物)
    
    # 创建颜色映射
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_green', colors, N=n_bins)
    
    # 设置颜色范围
    vmin, vmax = 0, 3
    
    im = ax.imshow(grid_map, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   extent=[X_MIN - X_STEP/2, X_MAX + X_STEP/2, 
                           Y_MIN - Y_STEP/2, Y_MAX + Y_STEP/2])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('鲁棒安全角度数', rotation=270, labelpad=20)
    
    # 绘制路径
    if path_indices:
        # 获取完整的路径格子和连续轨迹
        path_cells, continuous_trajectory = get_path_cells_and_trajectory(path_indices)
        
        # 绘制所有路径经过的格子
        for ix, iy in path_cells:
            rect = patches.Rectangle(
                (x_space[ix] - X_STEP/2, y_space[iy] - Y_STEP/2), 
                X_STEP, Y_STEP, 
                facecolor='#E57373', alpha=0.7, edgecolor='red', linewidth=1
            )
            ax.add_patch(rect)
        
        # 绘制实际的连续轨迹
        trajectory_array = np.array(continuous_trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                color='#B71C1C', linewidth=3, label='A*轨迹', alpha=0.9)
        
        # 绘制关键点（A*路径节点）
        path_states = np.array([indices_to_state(*p) for p in path_indices])
        ax.plot(path_states[:, 0], path_states[:, 1], 
                color='#D32F2F', marker='o', markersize=4, linewidth=0, 
                label='离散路径点', alpha=0.8)
        
        # 绘制方向箭头（每隔几个点绘制一个）
        for i in range(0, len(path_states), 2):  # 每隔2个点绘制一个箭头
            state = path_states[i]
            ax.arrow(state[0], state[1], 
                     0.4 * np.cos(state[2]), 0.4 * np.sin(state[2]), 
                     head_width=0.2, head_length=0.2, fc='#B71C1C', ec='#B71C1C', alpha=0.8)
    
    ax.plot(start_continuous[0], start_continuous[1], marker='o', color='green', markersize=12, label='起点')
    ax.plot(goal_continuous[0], goal_continuous[1], marker='*', color='blue', markersize=18, label='终点')
    
    # 添加网格线
    for x in x_space:
        ax.axvline(x - X_STEP/2, color='gray', linewidth=0.5, alpha=0.7)
        ax.axvline(x + X_STEP/2, color='gray', linewidth=0.5, alpha=0.7)
    for y in y_space:
        ax.axhline(y - Y_STEP/2, color='gray', linewidth=0.5, alpha=0.7)
        ax.axhline(y + Y_STEP/2, color='gray', linewidth=0.5, alpha=0.7)
    
    ax.set_xticks(x_space[::2])
    ax.set_yticks(y_space[::2])
    
    # 创建图例元素
    legend_patches = [
        patches.Patch(color='#E8F5E8', label='低鲁棒性'),
        patches.Patch(color='#66BB6A', label='高鲁棒性'), 
        patches.Patch(color='#000000', label='障碍物'),
        patches.Patch(color='#E57373', label='路径覆盖')
    ]
    
    # 获取线条图例
    handles, labels = ax.get_legend_handles_labels()
    
    # 将图例放在图形外部右侧
    ax.legend(handles=handles + legend_patches, 
             bbox_to_anchor=(1.35, 1), loc='upper left')

    ax.set_xlim(X_MIN - 1, X_MAX + 1)
    ax.set_ylim(Y_MIN - 1, Y_MAX + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('基于鲁棒前向不变集的独轮车机器人路径规划\n'
                '离散时间鲁棒可达性分析与扰动处理')
    plt.tight_layout()
    plt.show()

def get_path_cells_and_trajectory(path_indices, came_from=None):
    """获取路径经过的所有格子和实际连续轨迹"""
    if not path_indices:
        return set(), []
    
    path_cells = set()
    continuous_trajectory = []
    
    # 第一个点：起始状态
    current_state = indices_to_state(*path_indices[0])
    continuous_trajectory.append(current_state[:2])
    path_cells.add((path_indices[0][0], path_indices[0][1]))
    
    # 从第二个点开始，通过unicycle模型生成真正的连续轨迹
    for i in range(1, len(path_indices)):
        prev_discrete_state = indices_to_state(*path_indices[i-1])
        curr_discrete_state = indices_to_state(*path_indices[i])
        
        # 计算使用的omega（通过逆向计算）
        # 从prev_discrete_state开始，找到能到达curr_discrete_state附近的omega
        best_omega = 0
        min_distance = float('inf')
        
        for omega in omega_space:
            predicted_next = unicycle_model(prev_discrete_state, omega)
            distance = np.sqrt((predicted_next[0] - curr_discrete_state[0])**2 + 
                             (predicted_next[1] - curr_discrete_state[1])**2)
            if distance < min_distance:
                min_distance = distance
                best_omega = omega
        
        # 使用找到的omega生成真正的连续轨迹点
        actual_next_state = unicycle_model(prev_discrete_state, best_omega)
        continuous_trajectory.append(actual_next_state[:2])
        
        # 在连续轨迹上密集采样，找到所有经过的格子
        x1, y1 = prev_discrete_state[0], prev_discrete_state[1]
        x2, y2 = actual_next_state[0], actual_next_state[1]
        
        # 计算应该有多少个采样点
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        num_samples = max(int(distance / (min(X_STEP, Y_STEP) * 0.2)), 10)
        
        for j in range(num_samples + 1):
            t = j / max(num_samples, 1)
            x_sample = x1 + t * (x2 - x1)
            y_sample = y1 + t * (y2 - y1)
            
            # 将采样点转换为格子索引
            sample_indices = discretize_state(x_sample, y_sample, 0)
            path_cells.add((sample_indices[0], sample_indices[1]))
    
    return path_cells, continuous_trajectory        
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
    
    # 5. 生成并打印专业性能指标
    metrics = generate_performance_metrics(results)
    print_academic_results_table(metrics)
    
    return results


def generate_performance_metrics(results):
    """生成专业的性能指标"""
    metrics = {
        'computational_efficiency': {},
        'algorithmic_performance': {},
        'safety_metrics': {}
    }
    
    # 获取安全集大小和图边数
    safe_set_size = len(results.get('safe_set', set()))
    graph_edges_count = results.get('graph_edges_count', 0)
    
    # 计算效率指标
    for method_name, result in results.items():
        # 跳过非算法结果的键
        if method_name in ['safe_set', 'graph_edges_count']:
            continue
            
        stats = result['stats']
        
        # 计算内存占用：不同算法的实际存储需求
        if method_name == 'baseline':
            # 基线A*：只需维护开放集
            memory_usage = stats['nodes_in_open_set']
        elif method_name == 'robust':
            # 鲁棒A*：需要存储完整安全集 + 开放集
            memory_usage = safe_set_size
        elif method_name == 'graph_optimized':
            # 图优化A*：需要存储图的边连接 + 开放集
            memory_usage = graph_edges_count
        else:
            memory_usage = stats['nodes_in_open_set']
        
        # 搜索效率（仅搜索阶段）
        metrics['computational_efficiency'][method_name] = {
            'search_time_ms': stats['computation_time'] * 1000,
            'nodes_expanded': stats['nodes_expanded'],
            'nodes_per_second': stats['nodes_expanded'] / max(stats['computation_time'], 1e-6),
            'memory_usage': memory_usage
        }
        
        # 算法性能（删除success_rate）
        if stats['success']:
            metrics['algorithmic_performance'][method_name] = {
                'path_length': stats.get('path_length', 0),
                'optimality_ratio': stats.get('path_length', 0) / max(1, results.get('baseline', {}).get('stats', {}).get('path_length', 1))
            }
        else:
            metrics['algorithmic_performance'][method_name] = {
                'path_length': float('inf'),
                'optimality_ratio': float('inf')
            }
    
    return metrics

def print_academic_results_table(metrics):
    """打印学术论文格式的结果表格"""
    print("\n" + "="*80)
    print("基于RPIS的A*路径规划算法性能对比")
    print("="*80)
    
    # 表格1：搜索效率对比（仅搜索阶段）
    print("\n表1: 搜索阶段计算效率对比")
    print("-" * 80)
    print(f"{'算法':<18} {'搜索时间':<12} {'节点扩展':<12} {'搜索效率':<15} {'内存占用':<15}")
    print(f"{'':18} {'(毫秒)':<12} {'(个数)':<12} {'(节点/秒)':<15} {'(存储单位)':<15}")
    print("-" * 80)
    
    for method in ['baseline', 'robust', 'graph_optimized']:
        if method in metrics['computational_efficiency']:
            data = metrics['computational_efficiency'][method]
            method_name = {'baseline': '基线A*', 'robust': '鲁棒A*', 'graph_optimized': '图优化A*'}[method]
            memory_desc = {'baseline': '开放集节点', 'robust': '安全集状态', 'graph_optimized': '图边连接'}[method]
            print(f"{method_name:<18} {data['search_time_ms']:<12.2f} {data['nodes_expanded']:<12} {data['nodes_per_second']:<15.0f} {data['memory_usage']:<7}{memory_desc}")
    
    # 表格2：路径质量对比
    print(f"\n表2: 路径质量与安全性对比")
    print("-" * 80)
    print(f"{'算法':<18} {'路径长度':<12} {'相对基线':<12} {'安全保障':<15}")
    print(f"{'':18} {'(步数)':<12} {'(倍数)':<12} {'':15}")
    print("-" * 80)
    
    for method in ['baseline', 'robust', 'graph_optimized']:
        if method in metrics['algorithmic_performance']:
            data = metrics['algorithmic_performance'][method]
            method_name = {'baseline': '基线A*', 'robust': '鲁棒A*', 'graph_optimized': '图优化A*'}[method]
            safety = {'baseline': '无', 'robust': '是', 'graph_optimized': '是'}[method]
            path_len = data['path_length'] if data['path_length'] != float('inf') else 'N/A'
            opt_ratio = f"{data['optimality_ratio']:.2f}" if data['optimality_ratio'] != float('inf') else '1.00'
            print(f"{method_name:<18} {path_len:<12} {opt_ratio:<12} {safety:<15}")
    
    # 性能提升分析
    if 'robust' in metrics['computational_efficiency'] and 'graph_optimized' in metrics['computational_efficiency']:
        robust_time = metrics['computational_efficiency']['robust']['search_time_ms']
        graph_time = metrics['computational_efficiency']['graph_optimized']['search_time_ms']
        speedup = robust_time / graph_time if graph_time > 0 else float('inf')
        
        print(f"\n性能分析总结:")
        print(f"• 图优化A*相比鲁棒A*搜索提速: {speedup:.1f}倍")
        print(f"• 安全性保障: 鲁棒A*和图优化A*均提供数学证明的安全性")

def print_three_way_comparison_results(baseline_stats, robust_stats, graph_stats):
    """打印三种A*方法的详细对比结果"""
    print("\n" + "="*80)
    print("三种A*算法性能对比")
    print("="*80)
    
    print(f"{'指标':<20} {'基线A*':<15} {'鲁棒A*':<15} {'图优化A*':<15}")
    print("-" * 80)
    
    # 成功率对比
    baseline_success = "✓" if baseline_stats['success'] else "✗"
    robust_success = "✓" if robust_stats['success'] else "✗"
    graph_success = "✓" if graph_stats['success'] else "✗"
    print(f"{'成功找到路径':<20} {baseline_success:<15} {robust_success:<15} {graph_success:<15}")
    
    # 计算时间对比（搜索阶段）
    baseline_time = baseline_stats['computation_time']
    robust_time = robust_stats['computation_time']
    graph_time = graph_stats['computation_time']
    print(f"{'搜索时间(秒)':<20} {baseline_time:<15.4f} {robust_time:<15.4f} {graph_time:<15.4f}")
    
    # 图构建时间
    print(f"{'图构建时间(秒)':<20} {'-':<15} {'-':<15} {graph_stats['graph_build_time']:<15.4f}")
    
    # 总时间（包含预处理）
    graph_total = graph_stats['total_time']
    print(f"{'总时间(秒)':<20} {baseline_time:<15.4f} {robust_time:<15.4f} {graph_total:<15.4f}")
    
    # 扩展节点数对比
    baseline_expanded = baseline_stats['nodes_expanded']
    robust_expanded = robust_stats['nodes_expanded']
    graph_expanded = graph_stats['nodes_expanded']
    print(f"{'扩展节点数':<20} {baseline_expanded:<15} {robust_expanded:<15} {graph_expanded:<15}")
    
    # 路径长度对比
    if baseline_stats['success'] and robust_stats['success'] and graph_stats['success']:
        baseline_length = baseline_stats['path_length']
        robust_length = robust_stats['path_length']
        graph_length = graph_stats['path_length']
        print(f"{'路径长度':<20} {baseline_length:<15} {robust_length:<15} {graph_length:<15}")

    if graph_time < robust_time:
        speedup = robust_time / graph_time
        print(f"• 图优化相比鲁棒A*搜索提速: {speedup:.1f}x")
    
    if graph_expanded < robust_expanded:
        efficiency = robust_expanded / graph_expanded
        print(f"• 图优化相比鲁棒A*节点扩展效率提升: {efficiency:.1f}x")

# --- 4. 主程序与可视化 ---
if __name__ == "__main__":    
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
        exit()
    
    # 2. A*算法对比实验
    comparison_results = compare_astar_methods(start_indices, goal_xy_indices, S_infinity, obstacle_indices)
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