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
V = 0.99 
DT = 1.0
# 状态空间 S = [x, y, theta] - 优化后的网格（平衡精度与计算效率）
X_MIN, X_MAX, X_STEP = -1, 11, 0.5  
Y_MIN, Y_MAX, Y_STEP = -1, 11, 0.5    
THETA_MIN, THETA_MAX, THETA_STEP = -np.pi, np.pi, np.pi / 8  # 角度分辨率
# 动作空间 A = [omega]
OMEGA_MIN, OMEGA_MAX, OMEGA_STEP = -np.pi *3/ 8, np.pi *3/ 8, np.pi / 8 # 动作空间

x_space = np.linspace(X_MIN, X_MAX, int((X_MAX - X_MIN) / X_STEP) + 1)
y_space = np.linspace(Y_MIN, Y_MAX, int((Y_MAX - Y_MIN) / Y_STEP) + 1) 
theta_space = np.linspace(THETA_MIN, THETA_MAX, int((THETA_MAX - THETA_MIN) / THETA_STEP) + 1)[:-1]
omega_space = np.linspace(OMEGA_MIN, OMEGA_MAX, int((OMEGA_MAX - OMEGA_MIN) / OMEGA_STEP) + 1)

# *** 定义扰动空间 W - 优化版本 ***
# 定义扰动集：只测试四个角点
epsilon = -1e-3   
wx_space = np.array([-X_STEP/2+1e-3, X_STEP/2-1e-3])
wy_space = np.array([-Y_STEP/2+1e-3, Y_STEP/2-1e-3])
wtheta_space = np.array([0.0])  # 暂不考虑角度扰动

# 创建扰动集 W (取四个角点)
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
    x_next = x + V * np.cos(theta) * DT
    y_next = y + V * np.sin(theta) * DT
    
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
    sample_distance = grid_diagonal * 0.2 # 采样间距为网格对角线的40%
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

def compute_robust_safe_set_optimized(obstacle_indices, W):
    """计算鲁棒安全集并预处理安全动作映射"""
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
            return set(), {}
            
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
            
            # 构建完安全集后，预计算所有状态的安全动作
            print("预计算每个状态的安全动作...")
            safe_actions_map = compute_safe_actions_for_all_states(Sk_plus_1, obstacle_indices, W_array, omega_list)
            
            return Sk_plus_1, safe_actions_map
        
        Sk = Sk_plus_1
        k += 1

def compute_safe_actions_for_all_states(S_infinity, obstacle_indices, W_array, omega_list):
    """
    预计算每个状态下的所有安全动作
    Args:
        S_infinity: 鲁棒安全集
        obstacle_indices: 障碍物索引集合
        W_array: 扰动数组
        omega_list: 动作列表
    Returns:
        dict: 状态索引 -> 安全动作列表的映射
    """
    safe_actions_map = {}
    
    print(f"预计算 {len(S_infinity)} 个状态的安全动作...")
    
    for s_indices in tqdm(S_infinity, desc="预计算安全动作", ncols=100):
        s_center = np.array([x_space[s_indices[0]], y_space[s_indices[1]], theta_space[s_indices[2]]])
        safe_actions = []
        
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
                if s_next_indices not in S_infinity:
                    is_action_robustly_safe = False
                    break
            
            if is_action_robustly_safe:
                safe_actions.append(omega)
        
        safe_actions_map[s_indices] = safe_actions
    return safe_actions_map



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
    x1, y1, theta1 = indices_to_state(*state_indices)
    x2, y2, _ = indices_to_state(*goal_indices)
    
    # 位置距离（欧几里得距离）
    position_distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # 计算从当前位置到目标位置的理想角度
    if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
        # 如果已经在目标位置，角度成本为0
        angle_cost = 0.0
    else:
        target_angle = np.arctan2(y2 - y1, x2 - x1)
        
        # 计算角度差值（考虑角度的周期性）
        angle_diff = target_angle - theta1
        # 将角度差值标准化到[-π, π]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
          # 角度成本：标准化到[0, 1]，1表示完全反向
        angle_cost = abs(angle_diff) / np.pi
    
    return position_distance + angle_cost

def a_star_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, use_robust_constraints=True, safe_actions_map=None):
    """
    A*搜索算法，支持鲁棒约束和基线模式对比  
    Args:
        start_indices: 起点离散索引
        goal_xy_indices: 目标点xy索引
        S_infinity: 鲁棒安全集
        obstacle_indices: 障碍物索引
        use_robust_constraints: 是否使用鲁棒安全集约束 (True=鲁棒A*, False=基线A*)
        safe_actions_map: 预计算的安全动作映射（可选）
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
        'actions_skipped_by_precompute': 0,
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
          # 检查是否到达目标区域（以目标点为中心的3×3区域）
        goal_center_x, goal_center_y = goal_xy_indices
        current_x, current_y = current_indices[0], current_indices[1]
          # 检查当前位置是否在目标区域内（3×3区域）
        if (abs(current_x - goal_center_x) <= 1 and abs(current_y - goal_center_y) <= 1):
            path = []
            while current_indices in came_from:
                path.append(current_indices)
                current_indices = came_from[current_indices]
            path.append(start_indices)
            search_stats['path_length'] = len(path)
            search_stats['success'] = True
            return path[::-1], search_stats

        # 使用预计算的安全动作（如果可用）
        if use_robust_constraints and safe_actions_map and current_indices in safe_actions_map:
            # 直接使用预计算的安全动作，跳过安全检查
            available_actions = safe_actions_map[current_indices]
            search_stats['actions_skipped_by_precompute'] += len(omega_space) - len(available_actions)
        else:
            # 使用所有动作
            available_actions = omega_space

        for omega in available_actions:
            neighbor_indices = get_next_state_indices_for_astar(current_indices, omega)
            
            # 边界检查（两种模式都需要）
            if neighbor_indices is None:
                search_stats['nodes_rejected_by_bounds'] += 1
                continue
            
            # 安全集约束检查（仅鲁棒模式且未使用预计算时）
            if use_robust_constraints and (not safe_actions_map or current_indices not in safe_actions_map):
                if neighbor_indices not in S_infinity:
                    search_stats['nodes_rejected_by_safety'] += 1
                    continue            
            
            # 路径碰撞检查（仅在未使用预计算时）
            if not safe_actions_map or current_indices not in safe_actions_map:
                current_state = indices_to_state(*current_indices)
                neighbor_state = indices_to_state(*neighbor_indices)
                if check_path_collision(current_state, neighbor_state, obstacle_indices):
                    search_stats['nodes_rejected_by_collision'] += 1
                    continue

            # 计算带微小角度权重的成本
            current_state = indices_to_state(*current_indices)
            neighbor_state = indices_to_state(*neighbor_indices)
            base_cost = 1.0
            # 角度变化成本（很小的权重，不阻碍转向）
            angle_change = abs(neighbor_state[2] - current_state[2])
            angle_change = min(angle_change, 2*np.pi - angle_change)  # 标准化到[0, π]
            # 角度变化成本：标准化到[0, 0.5]，仅用于打破平局
            angle_cost = angle_change / (2 * np.pi)
            
            step_cost = base_cost + angle_cost
            tentative_g_score = g_score[current_indices] + step_cost
            
            if neighbor_indices not in g_score or tentative_g_score < g_score[neighbor_indices]:
                came_from[neighbor_indices] = current_indices
                g_score[neighbor_indices] = tentative_g_score
                f_score[neighbor_indices] = tentative_g_score + heuristic(neighbor_indices, (*goal_xy_indices, 0))
                heapq.heappush(open_set, (f_score[neighbor_indices], neighbor_indices))
                search_stats['nodes_in_open_set'] += 1

    return None, search_stats

# --- 3. 辅助函数定义 ---

def compare_astar_methods(start_indices, goal_xy_indices, S_infinity, safe_actions_map, obstacle_indices):
    """
    对比两种A*方法的性能：基线A*和鲁棒A*
    """
    results = {}
    
    # 1. 运行基线A*
    start_time = time.time()
    baseline_path, baseline_stats = a_star_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, use_robust_constraints=False)
    baseline_time = time.time() - start_time
    baseline_stats['computation_time'] = baseline_time
    results['baseline'] = {'path': baseline_path, 'stats': baseline_stats}
    
    # 2. 运行鲁棒A*（使用预计算的安全动作）
    start_time = time.time()
    robust_path, robust_stats = a_star_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, use_robust_constraints=True, safe_actions_map=safe_actions_map)
    robust_time = time.time() - start_time
    robust_stats['computation_time'] = robust_time
    results['robust'] = {'path': robust_path, 'stats': robust_stats}
    
    print_comparison_results(baseline_stats, robust_stats)
    return results

def print_comparison_results(baseline_stats, robust_stats):
    """简化的两种算法性能对比打印"""
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
    
    if baseline_stats['success'] and robust_stats['success']:
        baseline_length = baseline_stats['path_length']
        robust_length = robust_stats['path_length']
        print(f"{'路径长度':<20} {baseline_length:<15} {robust_length:<15}")

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
    """Path planning visualization - supports dual path comparison"""
    # Create figure with smaller size for better coordinate display
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    
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
    
      # Draw baseline A* path
    if baseline_path:
        # 获取完整的路径格子和连续轨迹
        baseline_cells, baseline_trajectory = get_path_cells_and_trajectory(baseline_path)
        
        # Plot baseline A* continuous trajectory
        trajectory_array = np.array(baseline_trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                color='#D32F2F', linewidth=3, label='Standard A*', alpha=0.9)
        
        # 绘制关键点
        path_states = np.array([indices_to_state(*p) for p in baseline_path])
        ax.plot(path_states[:, 0], path_states[:, 1], 
                color='#F44336', marker='s', markersize=4, linewidth=0, 
                alpha=0.8)
    
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
        
        # Plot actual continuous trajectory
        trajectory_array = np.array(robust_trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                color="#0741FF", linewidth=3, label='Safe A*', alpha=0.9)
        
        # 绘制关键点
        path_states = np.array([indices_to_state(*p) for p in robust_path])
        ax.plot(path_states[:, 0], path_states[:, 1], 
                color="#0521F1E9", marker='o', markersize=4, linewidth=0, 
                alpha=0.8)
        
        # 绘制方向箭头（每隔几个点绘制一个）
        for i in range(0, len(path_states)):  # 每隔2个点绘制一个箭头
            state = path_states[i]
            ax.arrow(state[0], state[1],                     0.4 * np.cos(state[2]), 0.4 * np.sin(state[2]), 
                     head_width=0.2, head_length=0.2, fc="#070FFF", ec="#071CFF", alpha=0.8)
    
    ax.plot(start_continuous[0], start_continuous[1], marker='o', color='green', markersize=12, label='Start')
    
    # 绘制目标区域（3×3格子）
    goal_x, goal_y = goal_continuous[0], goal_continuous[1]
    goal_ix = int(round((goal_x - X_MIN) / X_STEP))
    goal_iy = int(round((goal_y - Y_MIN) / Y_STEP))
    
    # 绘制3×3目标区域
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            target_ix = goal_ix + dx
            target_iy = goal_iy + dy
            if 0 <= target_ix < len(x_space) and 0 <= target_iy < len(y_space):
                rect = patches.Rectangle(
                    (x_space[target_ix] - X_STEP/2, y_space[target_iy] - Y_STEP/2),                    X_STEP, Y_STEP,
                    facecolor='gold', alpha=0.3, edgecolor='goldenrod', linewidth=2
                )
                ax.add_patch(rect)
    
    # Mark goal region center
    ax.plot(goal_continuous[0], goal_continuous[1], marker='*', color='gold', markersize=25, label='Goal', markeredgecolor='black', markeredgewidth=2)
      # 添加网格线
    for x in x_space:
        ax.axvline(x - X_STEP/2, color='gray', linewidth=0.5, alpha=0.7)
        ax.axvline(x + X_STEP/2, color='gray', linewidth=0.5, alpha=0.7)
    for y in y_space:
        ax.axhline(y - Y_STEP/2, color='gray', linewidth=0.5, alpha=0.7)
        ax.axhline(y + Y_STEP/2, color='gray', linewidth=0.5, alpha=0.7)
    
    # Add grid lines with larger tick spacing for better readability
    ax.set_xticks(x_space[::4])  # Show every 4th tick
    ax.set_yticks(y_space[::4])  # Show every 4th tick
    ax.tick_params(axis='both', which='major', labelsize=12)  # Larger tick labels
    
    # Create legend elements
    legend_patches = [
        patches.Patch(color='#000000', label='Obstacles'),
        patches.Patch(color='#59A8F6FF', label='Path'),
        patches.Patch(color='gold', alpha=0.3, label='Goal Region')
    ]
    
    # Get line legend
    handles, labels = ax.get_legend_handles_labels()
    # Place legend inside the figure with transparency
    ax.legend(handles=handles + legend_patches, fontsize=10, markerscale=0.8,
             loc='upper left', framealpha=0.8, fancybox=True, shadow=True)
    ax.set_xlim(X_MIN - 1, X_MAX + 1)
    ax.set_ylim(Y_MIN - 1, Y_MAX + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    plt.tight_layout()
    
    # Save high resolution image
    plt.savefig('path_planning_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_safety_angle_visualization(S_infinity, obstacle_indices, start_continuous, goal_continuous):
    """Create safety angle arrow visualization - shows θ dimension information"""
    
    # Create smaller figure for better coordinate display
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
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
                        fc=color, ec=color, alpha=alpha, width=width)    # Draw start and end points
    ax.plot(start_continuous[0], start_continuous[1], 
            marker='o', color='blue', markersize=16, label='Start', markeredgecolor='white', markeredgewidth=2)
    ax.plot(goal_continuous[0], goal_continuous[1], 
            marker='*', color='gold', markersize=20, label='Goal', markeredgecolor='black', markeredgewidth=2)
    
    # 添加网格线（更细的线）
    for x in x_space:
        ax.axvline(x - X_STEP/2, color='lightgray', linewidth=0.5, alpha=0.7)
        ax.axvline(x + X_STEP/2, color='lightgray', linewidth=0.5, alpha=0.7)
    for y in y_space:
        ax.axhline(y - Y_STEP/2, color='lightgray', linewidth=0.5, alpha=0.7)
        ax.axhline(y + Y_STEP/2, color='lightgray', linewidth=0.5, alpha=0.7)    # Create legend
    legend_elements = [
        patches.Patch(color='green', alpha=0.8, label='Safe Angle'),
        patches.Patch(color='red', alpha=0.6, label='Unsafe Angle'),
        patches.Patch(color='black', label='Obstacle', alpha=0.8),
    ]
    
    # Get start/end point legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Set figure properties with larger tick spacing and labels
    ax.set_xticks(x_space[::4])  # Show every 4th tick
    ax.set_yticks(y_space[::4])  # Show every 4th tick
    ax.tick_params(axis='both', which='major', labelsize=12)  # Larger tick labels
    
    # Place legend inside the figure
    ax.legend(handles=handles + legend_elements, fontsize=10, markerscale=0.8,
             loc='lower right', framealpha=0.8, fancybox=True, shadow=True)
    ax.set_xlim(X_MIN - 1, X_MAX + 1)
    ax.set_ylim(Y_MIN - 1, Y_MAX + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    
    plt.tight_layout()
    
    # Save ultra high resolution image
    plt.savefig('safe_angle_arrows.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# --- 4. 主程序 ---
if __name__ == "__main__":    
    # 定义问题
    start_continuous = np.array([0., 0., np.pi / 4])  
    goal_continuous = np.array([10., 10.])
    
    start_indices = discretize_state(*start_continuous)
    goal_xy_indices = discretize_state(goal_continuous[0], goal_continuous[1], 0)[:2]

    # 生成简单的障碍物
    obstacle_indices = set()
    
    # # 添加一些简单的障碍物
    # for i in range(4, 16):
    #     obstacle_indices.add((i, 18))  # 水平障碍
    # for i in range(6, 16):
    #     obstacle_indices.add((i, 7)) 
    # for j in range(8, 19):
    #     obstacle_indices.add((18, j))  # 垂直障碍    obstacle_indices.add((4, 16))
    # obstacle_indices.add((3,17))
    # obstacle_indices.add((5,17))
     # 添加一些简单的障碍物
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
    # 1. 计算鲁棒安全集
    S_infinity, safe_actions_map = compute_robust_safe_set_optimized(obstacle_indices, W)

    if not S_infinity:
        exit()
      
    # 2. A*算法对比实验
    comparison_results = compare_astar_methods(start_indices, goal_xy_indices, S_infinity, safe_actions_map, obstacle_indices)
    
    # 获取路径用于可视化
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
    # 4. 生成可视化
    create_original_path_visualization(S_infinity, obstacle_indices, baseline_path, robust_path,
                                       start_continuous, goal_continuous, safe_angle_count)
    
    create_safety_angle_visualization(S_infinity, obstacle_indices, start_continuous, goal_continuous)