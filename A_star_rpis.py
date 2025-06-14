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
    """Unicycle运动学模型 - 使用中点积分方法提高精度"""
    x, y, theta = state
    theta_next = theta + omega * DT
    theta_mid = theta + omega * DT / 2
    x_next = x + V * np.cos(theta_next) * DT
    y_next = y + V * np.sin(theta_next) * DT
    
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

# --- 3. 辅助函数定义 ---

def compare_astar_methods(start_indices, goal_xy_indices, S_infinity, obstacle_indices):
    """
    对比两种A*方法的性能：基线A*和鲁棒A*
    Returns:
        dict: 包含两种方法的结果和统计对比
    """
    print("A*算法性能对比...")
    results = {}
    # 1. 运行基线A*
    print("运行基线A*...")
    start_time = time.time()
    baseline_path, baseline_stats = a_star_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, use_robust_constraints=False)
    baseline_time = time.time() - start_time
    baseline_stats['computation_time'] = baseline_time
    results['baseline'] = {'path': baseline_path, 'stats': baseline_stats}
    # 2. 运行鲁棒A*
    print("运行鲁棒A*...")
    start_time = time.time()
    robust_path, robust_stats = a_star_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, use_robust_constraints=True)
    robust_time = time.time() - start_time
    robust_stats['computation_time'] = robust_time
    results['robust'] = {'path': robust_path, 'stats': robust_stats}
    print_comparison_results(baseline_stats, robust_stats)
    return results

def print_comparison_results(baseline_stats, robust_stats):
    print("\n" + "="*80)
    print("基线A* vs 鲁棒A* 性能对比")
    print("="*80)
    print(f"{'指标':<20} {'基线A*':<15} {'鲁棒A*':<15}")
    print("-" * 50)
    baseline_success = "✓" if baseline_stats['success'] else "✗"
    robust_success = "✓" if robust_stats['success'] else "✗"
    print(f"{'成功找到路径':<20} {baseline_success:<15} {robust_success:<15}")
    baseline_time = baseline_stats['computation_time']
    robust_time = robust_stats['computation_time']
    print(f"{'搜索时间(秒)':<20} {baseline_time:<15.4f} {robust_time:<15.4f}")
    baseline_expanded = baseline_stats['nodes_expanded']
    robust_expanded = robust_stats['nodes_expanded']
    print(f"{'扩展节点数':<20} {baseline_expanded:<15} {robust_expanded:<15}")
    print(f"{'边界外拒绝':<20} {baseline_stats['nodes_rejected_by_bounds']:<15} {robust_stats['nodes_rejected_by_bounds']:<15}")
    print(f"{'安全性拒绝':<20} {baseline_stats['nodes_rejected_by_safety']:<15} {robust_stats['nodes_rejected_by_safety']:<15}")
    print(f"{'碰撞拒绝':<20} {baseline_stats['nodes_rejected_by_collision']:<15} {robust_stats['nodes_rejected_by_collision']:<15}")
    if baseline_stats['success'] and robust_stats['success']:
        baseline_length = baseline_stats['path_length']
        robust_length = robust_stats['path_length']
        print(f"{'路径长度':<20} {baseline_length:<15} {robust_length:<15}")
        if baseline_length > 0:
            quality_ratio = robust_length / baseline_length
            print(f"{'路径质量比':<20} {'1.00':<15} {quality_ratio:<15.2f}")
    print("-" * 50)
    if baseline_time > 0 and robust_time > 0:
        if robust_time < baseline_time:
            speedup = baseline_time / robust_time
            print(f"性能提升: 鲁棒A*比基线A*快 {speedup:.1f}x")
        elif robust_time > baseline_time:
            slowdown = robust_time / baseline_time
            print(f"性能代价: 鲁棒A*比基线A*慢 {slowdown:.1f}x")
        else:
            print("性能相当")
    if robust_stats['success']:
        if robust_stats['nodes_rejected_by_safety'] > 0:
            print(f"安全保障: 鲁棒A*拒绝了 {robust_stats['nodes_rejected_by_safety']} 个不安全状态")
        else:
            print("安全保障: 所有搜索状态均在安全集内")
    print("="*80)

def get_path_cells_and_trajectory(path_indices, came_from=None):
    if not path_indices:
        return set(), []
    path_cells = set()
    continuous_trajectory = []
    current_state = indices_to_state(*path_indices[0])
    continuous_trajectory.append(current_state[:2])
    path_cells.add((path_indices[0][0], path_indices[0][1]))
    for i in range(1, len(path_indices)):
        prev_discrete_state = indices_to_state(*path_indices[i-1])
        curr_discrete_state = indices_to_state(*path_indices[i])
        best_omega = 0
        min_distance = float('inf')
        for omega in omega_space:
            predicted_next = unicycle_model(prev_discrete_state, omega)
            distance = np.sqrt((predicted_next[0] - curr_discrete_state[0])**2 + (predicted_next[1] - curr_discrete_state[1])**2)
            if distance < min_distance:
                min_distance = distance
                best_omega = omega
        actual_next_state = unicycle_model(prev_discrete_state, best_omega)
        continuous_trajectory.append(actual_next_state[:2])
        x1, y1 = prev_discrete_state[0], prev_discrete_state[1]
        x2, y2 = actual_next_state[0], actual_next_state[1]
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        num_samples = max(int(distance / (min(X_STEP, Y_STEP) * 0.2)), 10)
        for j in range(num_samples + 1):
            t = j / max(num_samples, 1)
            x_sample = x1 + t * (x2 - x1)
            y_sample = y1 + t * (y2 - y1)
            sample_indices = discretize_state(x_sample, y_sample, 0)
            path_cells.add((sample_indices[0], sample_indices[1]))
    return path_cells, continuous_trajectory

def create_original_path_visualization(S_infinity, obstacle_indices, baseline_path, robust_path, start_continuous, goal_continuous, safe_angle_count):
    """路径规划可视化 - 支持双路径对比显示"""
    # 创建大图，增大尺寸和DPI
    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
    
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
      # 绘制基线A*路径
    if baseline_path:
        # 获取完整的路径格子和连续轨迹
        baseline_cells, baseline_trajectory = get_path_cells_and_trajectory(baseline_path)
        
        # 绘制基线A*的连续轨迹
        trajectory_array = np.array(baseline_trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                color='#D32F2F', linewidth=3, label='标准A*轨迹', alpha=0.9)
        
        # 绘制关键点
        path_states = np.array([indices_to_state(*p) for p in baseline_path])
        ax.plot(path_states[:, 0], path_states[:, 1], 
                color='#F44336', marker='s', markersize=4, linewidth=0, 
                label='标准A*路径点', alpha=0.8)
    
    # 绘制鲁棒A*路径  
    if robust_path:
        # 获取完整的路径格子和连续轨迹
        robust_cells, robust_trajectory = get_path_cells_and_trajectory(robust_path)
        
        # 绘制所有路径经过的格子
        for ix, iy in robust_cells:
            rect = patches.Rectangle(
                (x_space[ix] - X_STEP/2, y_space[iy] - Y_STEP/2), 
                X_STEP, Y_STEP, 
                facecolor="#59A8F6FF", alpha=0.6, edgecolor="#5490F7FF", linewidth=1            )
            ax.add_patch(rect)
        
        # 绘制实际的连续轨迹
        trajectory_array = np.array(robust_trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                color="#0741FF", linewidth=3, label='安全A*轨迹', alpha=0.9)
        
        # 绘制关键点
        path_states = np.array([indices_to_state(*p) for p in robust_path])
        ax.plot(path_states[:, 0], path_states[:, 1], 
                color="#0521F1E9", marker='o', markersize=4, linewidth=0, 
                label='安全A*路径点', alpha=0.8)
        
        # 绘制方向箭头（每隔几个点绘制一个）
        for i in range(0, len(path_states), 2):  # 每隔2个点绘制一个箭头
            state = path_states[i]
            ax.arrow(state[0], state[1], 
                     0.4 * np.cos(state[2]), 0.4 * np.sin(state[2]), 
                     head_width=0.2, head_length=0.2, fc="#070FFF", ec="#071CFF", alpha=0.8)
    
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
        patches.Patch(color='#000000', label='障碍物'),
        patches.Patch(color='#59A8F6FF', label='路径覆盖')
    ]
    
    # 获取线条图例
    handles, labels = ax.get_legend_handles_labels()
    # 将图例放在图形外部右侧
    ax.legend(handles=handles + legend_patches, 
             bbox_to_anchor=(1.35, 1), loc='upper left')
    ax.set_xlim(X_MIN - 1, X_MAX + 1)
    ax.set_ylim(Y_MIN - 1, Y_MAX + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('基于安全状态转移图的独轮车路径规划', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存高分辨率图片
    plt.savefig('path_planning_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_safety_angle_visualization(S_infinity, obstacle_indices, start_continuous, goal_continuous):
    """创建安全角度箭头可视化图 - 展示θ维度信息"""
    
    # 创建超大图形以便看清箭头
    fig, ax = plt.subplots(figsize=(20, 16), dpi=150)
    
    # 计算每个位置的安全角度
    position_safety = {}
    for ix, iy, itheta in S_infinity:
        key = (ix, iy)
        if key not in position_safety:
            position_safety[key] = set()
        position_safety[key].add(itheta)
    
    # 设置箭头长度（稍微大一些以便看清）
    arrow_length = min(X_STEP, Y_STEP) * 0.35
    
    # 为每个格点绘制箭头
    for ix in range(len(x_space)):
        for iy in range(len(y_space)):
            x_pos = x_space[ix]
            y_pos = y_space[iy]
            
            # 检查是否是障碍物
            if (ix, iy) in obstacle_indices:
                # 绘制障碍物格子
                rect = patches.Rectangle(
                    (x_pos - X_STEP/2, y_pos - Y_STEP/2), 
                    X_STEP, Y_STEP, 
                    facecolor='black', alpha=0.8, edgecolor='gray'
                )
                ax.add_patch(rect)
                continue
            
            # 获取此位置的安全角度
            safe_angles = position_safety.get((ix, iy), set())
            
            # 为每个角度绘制箭头
            for itheta in range(len(theta_space)):
                theta = theta_space[itheta]
                
                # 计算箭头的起点和终点
                start_x = x_pos
                start_y = y_pos
                end_x = start_x + arrow_length * np.cos(theta)
                end_y = start_y + arrow_length * np.sin(theta)
                
                # 根据安全性选择颜色和粗细
                if itheta in safe_angles:
                    color = 'green'
                    alpha = 0.8
                    width = arrow_length*0.08
                else:
                    color = 'red'
                    alpha = 0.6
                    width = arrow_length*0.05
                
                # 绘制箭头（增大箭头尺寸）
                ax.arrow(start_x, start_y, 
                        end_x - start_x, end_y - start_y,
                        head_width=arrow_length*0.4, 
                        head_length=arrow_length*0.3,
                        fc=color, ec=color, alpha=alpha, width=width)
    
    # 绘制起点和终点
    ax.plot(start_continuous[0], start_continuous[1], 
            marker='o', color='blue', markersize=20, label='起点', markeredgecolor='white', markeredgewidth=3)
    ax.plot(goal_continuous[0], goal_continuous[1], 
            marker='*', color='gold', markersize=25, label='终点', markeredgecolor='black', markeredgewidth=2)
    
    # 添加网格线（更细的线）
    for x in x_space:
        ax.axvline(x - X_STEP/2, color='lightgray', linewidth=0.5, alpha=0.7)
        ax.axvline(x + X_STEP/2, color='lightgray', linewidth=0.5, alpha=0.7)
    for y in y_space:
        ax.axhline(y - Y_STEP/2, color='lightgray', linewidth=0.5, alpha=0.7)
        ax.axhline(y + Y_STEP/2, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # 创建图例
    legend_elements = [
        patches.Patch(color='green', alpha=0.8, label='安全角度（θ维度）'),
        patches.Patch(color='red', alpha=0.6, label='不安全角度'),
        patches.Patch(color='black', label='障碍物')
    ]
    
    # 获取起点终点图例
    handles, labels = ax.get_legend_handles_labels()
    
    # 设置图形属性
    ax.legend(handles=handles + legend_elements, fontsize=14,
             bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlim(X_MIN - 1, X_MAX + 1)
    ax.set_ylim(Y_MIN - 1, Y_MAX + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('安全角度可视化图 - θ维度展示\n'
                '绿色箭头：安全方向，红色箭头：不安全方向', 
                fontsize=18, fontweight='bold')
    ax.set_xlabel('X坐标', fontsize=14)
    ax.set_ylabel('Y坐标', fontsize=14)
    
    plt.tight_layout()
    
    # 保存超高分辨率图片
    plt.savefig('safe_angle_arrows.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# --- 4. 主程序 ---
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
    
    # 获取两种路径用于对比可视化
    baseline_path = comparison_results['baseline']['path']
    robust_path = comparison_results['robust']['path']

    # 3. 计算安全角度统计
    safe_angle_count = {}
    max_angles = len(theta_space)
    
    for ix, iy, itheta in S_infinity:
        key = (ix, iy)
        if key not in safe_angle_count:
            safe_angle_count[key] = 0
        safe_angle_count[key] += 1
    
    # 4. 生成路径规划对比可视化（显示两种算法的路径）
    print("生成双路径对比可视化...")
    create_original_path_visualization(S_infinity, obstacle_indices, baseline_path, robust_path,
                                       start_continuous, goal_continuous, safe_angle_count)
    
    # 5. 生成安全角度箭头可视化
    print("生成安全角度箭头可视化...")
    create_safety_angle_visualization(S_infinity, obstacle_indices, start_continuous, goal_continuous)    
    # 6. 打印关键统计指标
    print(f"\n系统性能总结:")
    print(f"安全集收敛: {len(S_infinity):,} 个状态")
    print(f"平均鲁棒性: {np.mean(list(safe_angle_count.values())):.2f}/{max_angles} 角度" if safe_angle_count else "N/A")
    print(f"状态空间缩减: {len(S_infinity)/(len(x_space)*len(y_space)*len(theta_space))*100:.1f}% 保留")
    
    # 对比两种算法的性能
    if comparison_results['robust']['stats']['success'] and comparison_results['baseline']['stats']['success']:
        robust_time = comparison_results['robust']['stats']['computation_time']
        baseline_time = comparison_results['baseline']['stats']['computation_time']
        if baseline_time > 0:
            if robust_time < baseline_time:
                speedup = baseline_time / robust_time
                print(f"鲁棒A*相比基线A*提速: {speedup:.1f}x")
            else:
                slowdown = robust_time / baseline_time
                print(f"鲁棒A*相比基线A*减速: {slowdown:.1f}x")
        
        # 路径质量对比
        robust_length = comparison_results['robust']['stats']['path_length']
        baseline_length = comparison_results['baseline']['stats']['path_length']
        if baseline_length > 0:
            quality_ratio = robust_length / baseline_length
            print(f"鲁棒A*路径长度相比基线A*: {quality_ratio:.2f}倍")
    
    print("="*60)
    print("可视化图像已保存:")
    print("• path_planning_comparison.png - 双算法路径对比图")
    print("• safe_angle_arrows.png - 安全角度箭头可视化图")
    print("分析完成")