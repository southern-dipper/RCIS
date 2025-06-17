import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import time
import matplotlib.colors as mcolors
import random
from sklearn.neighbors import KDTree

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 1. 环境与模型定义 ---

# Unicycle模型参数  
V = 1.0 
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

# *** 定义扰动空间 W - 优化版本 ***
# 定义扰动集：只测试四个角点
epsilon = 1e-3  
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
                    break  # 找到一个安全动作就够了
        
        # 检查收敛
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

# --- 3. RRT 路径规划算法 ---

def angle_distance(theta1, theta2):
    """计算两个角度之间的最短距离（考虑循环性）"""
    diff = abs(theta1 - theta2)
    return min(diff, 2 * np.pi - diff)

def scaled_distance(state1, state2):
    """计算两个状态之间的欧氏距离（统一尺度）"""
    x1, y1, theta1 = state1
    x2, y2, theta2 = state2
    
    # 将角度缩放到与x,y相同的尺度
    theta_scale = (X_MAX - X_MIN) / (2 * np.pi)
    
    dx = x2 - x1
    dy = y2 - y1
    dtheta_scaled = angle_distance(theta1, theta2) * theta_scale
    
    return np.sqrt(dx*dx + dy*dy + dtheta_scaled*dtheta_scaled)

def build_safe_states_kdtree(S_infinity):
    """构建安全状态的KD-Tree"""
    if not S_infinity:
        return None, []
    
    # 将安全状态转换为连续状态并进行尺度统一
    safe_states_list = []
    theta_scale = (X_MAX - X_MIN) / (2 * np.pi)
    
    for ix, iy, itheta in S_infinity:
        state = indices_to_state(ix, iy, itheta)
        x, y, theta = state
        # 将theta缩放到与x,y相同的尺度
        theta_scaled = theta * theta_scale
        safe_states_list.append([x, y, theta, theta_scaled, ix, iy, itheta])  # 存储原始theta和缩放theta
    
    if not safe_states_list:
        return None, []
    
    # 构建KD-Tree（使用x, y, theta_scaled）
    safe_states_array = np.array(safe_states_list)
    kdtree_data = safe_states_array[:, [0, 1, 3]]  # x, y, theta_scaled
    kdtree = KDTree(kdtree_data)
    
    return kdtree, safe_states_list

class RRTNode:
    """RRT树节点"""
    def __init__(self, state, parent=None):
        self.state = state  # 连续状态 [x, y, theta]
        self.parent = parent
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)

def sample_random_state():
    """在3D状态空间中随机采样"""
    x = np.random.uniform(X_MIN, X_MAX)
    y = np.random.uniform(Y_MIN, Y_MAX)
    theta = np.random.uniform(THETA_MIN, THETA_MAX)
    return np.array([x, y, theta])

def find_nearest_node(tree_nodes, sample_state):
    """在RRT树中找到最近的节点"""
    min_dist = float('inf')
    nearest_node = None
    
    for node in tree_nodes:
        dist = scaled_distance(node.state, sample_state)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node

def steer(from_state, to_state, max_step=1.0):
    """从from_state向to_state方向扩展，最大步长为max_step"""
    dist = scaled_distance(from_state, to_state)
    
    if dist <= max_step:
        return to_state
    
    # 计算方向向量
    dx = to_state[0] - from_state[0]
    dy = to_state[1] - from_state[1]
    
    # 处理角度差（考虑循环性）
    dtheta = to_state[2] - from_state[2]
    if abs(dtheta) > np.pi:
        if dtheta > 0:
            dtheta -= 2 * np.pi
        else:
            dtheta += 2 * np.pi
    
    # 归一化并缩放
    theta_scale = (X_MAX - X_MIN) / (2 * np.pi)
    dtheta_scaled = dtheta * theta_scale
    
    # 计算实际的缩放因子
    total_scaled_dist = np.sqrt(dx*dx + dy*dy + dtheta_scaled*dtheta_scaled)
    scale_factor = max_step / total_scaled_dist
    
    # 计算新状态
    new_x = from_state[0] + dx * scale_factor
    new_y = from_state[1] + dy * scale_factor
    new_theta = from_state[2] + (dtheta * scale_factor)
    
    # 角度标准化
    new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
    
    return np.array([new_x, new_y, new_theta])

def is_goal_reached(state, goal_xy, tolerance=1.0):
    """检查是否到达目标区域（忽略theta）"""
    return (abs(state[0] - goal_xy[0]) <= tolerance and 
            abs(state[1] - goal_xy[1]) <= tolerance)

def rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, safe_actions_map=None, max_iterations=1000, goal_tolerance=1.0):
    """
    基线RRT搜索算法
    Returns:
        tuple: (path, tree_nodes) 其中path是连续状态列表，tree_nodes是所有树节点
    """
    print("运行基线RRT算法...")
    
    # 转换为连续状态
    start_state = indices_to_state(*start_indices)
    goal_xy = np.array([x_space[goal_xy_indices[0]], y_space[goal_xy_indices[1]]])
    
    # 初始化树
    root = RRTNode(start_state)
    tree_nodes = [root]
    
    # 构建安全状态KD-Tree（虽然基线RRT不用，但为了统一接口）
    kdtree, safe_states_list = build_safe_states_kdtree(S_infinity)
    
    for iteration in range(max_iterations):
        # 1. 随机采样
        sample_state = sample_random_state()
        
        # 2. 找到最近的树节点
        nearest_node = find_nearest_node(tree_nodes, sample_state)
        
        # 3. 向采样点扩展
        new_state = steer(nearest_node.state, sample_state, max_step=V*DT)
          # 4. 检查是否在边界内
        if not is_state_valid(new_state[0], new_state[1]):
            continue
        
        # 5. 检查路径碰撞（基线RRT需要做碰撞检查）
        if check_path_collision(nearest_node.state, new_state, obstacle_indices):
            continue
        
        # 6. 添加到树中
        new_node = RRTNode(new_state, nearest_node)
        nearest_node.add_child(new_node)
        tree_nodes.append(new_node)
        
        # 7. 检查是否到达目标
        if is_goal_reached(new_state, goal_xy, goal_tolerance):
            print(f"基线RRT找到路径！迭代次数: {iteration + 1}")
            # 回溯路径（连续状态）
            path = []
            current = new_node
            while current is not None:
                path.append(current.state.copy())  # 直接使用连续状态
                current = current.parent
            return path[::-1], tree_nodes  # 返回路径和整个树
        
        # 进度显示
        if (iteration + 1) % 100 == 0:
            print(f"基线RRT迭代进度: {iteration + 1}/{max_iterations}")
    
    print("基线RRT未找到路径")
    return None, tree_nodes

def find_nearest_safe_state(query_state, kdtree, safe_states_list, threshold=1.5):
    """
    在安全集中查找最近的安全状态
    Args:
        query_state: 查询状态 [x, y, theta]
        kdtree: 安全状态KD-Tree
        safe_states_list: 安全状态列表，格式：[x, y, theta, theta_scaled, ix, iy, itheta]
        threshold: 距离阈值
    Returns:
        tuple: (safe_state, distance) 或 (None, float('inf'))
    """
    if kdtree is None or not safe_states_list:
        return None, float('inf')
    
    # 将查询状态转换为与KD-Tree一致的格式
    theta_scale = (X_MAX - X_MIN) / (2 * np.pi)
    query_scaled = np.array([[query_state[0], query_state[1], query_state[2] * theta_scale]])  # 2D数组
    
    # 在KD-Tree中查找最近邻
    distances, indices = kdtree.query(query_scaled)
    distance = float(distances[0])  # 提取标量值并转换为Python float
    idx = int(indices[0].item())    # 安全地提取并转换为int索引
    
    # 检查距离是否在阈值内
    if distance <= threshold:
        # 返回原始的连续状态
        safe_state_info = safe_states_list[idx]
        safe_state = np.array([safe_state_info[0], safe_state_info[1], safe_state_info[2]])  # x, y, 原始theta
        return safe_state, distance
    else:
        return None, distance

def is_state_in_safe_set(state, S_infinity):
    """
    检查给定的连续状态是否在安全集内
    Args:
        state: 连续状态 [x, y, theta]
        S_infinity: 安全集（离散索引集合）
    Returns:
        bool: 是否安全
    """
    # 将连续状态离散化
    try:
        indices = discretize_state(*state)
        return indices in S_infinity
    except:
        return False

def safe_rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, safe_actions_map=None, max_iterations=1000, goal_tolerance=1.0):
    """
    安全RRT搜索算法 - 基于RPIS安全集
    核心思想：
    1. 在配置空间自由采样
    2. 如果扩展后的点不在安全集内，修正到最近的安全状态
    3. 不进行传统的碰撞检查，完全依赖安全集
    
    Args:
        start_indices: 起点离散索引
        goal_xy_indices: 目标点xy索引  
        S_infinity: 鲁棒安全集
        obstacle_indices: 障碍物索引（安全RRT中不直接使用）
        safe_actions_map: 预计算的安全动作映射（安全RRT中不直接使用）
        max_iterations: 最大迭代次数
        goal_tolerance: 目标容忍度
    Returns:
        tuple: (path, tree_nodes) 路径（连续状态列表）和树节点，或 (None, tree_nodes)
    """
    print("运行安全RRT算法...")
    
    # 转换为连续状态
    start_state = indices_to_state(*start_indices)
    goal_xy = np.array([x_space[goal_xy_indices[0]], y_space[goal_xy_indices[1]]])
    
    # 检查起点是否在安全集内
    if not is_state_in_safe_set(start_state, S_infinity):
        print("警告：起点不在安全集内")
        return None, []
    
    # 初始化树
    root = RRTNode(start_state)
    tree_nodes = [root]
    
    # 构建安全状态KD-Tree
    kdtree, safe_states_list = build_safe_states_kdtree(S_infinity)
    
    if kdtree is None:
        print("错误：无法构建安全状态KD-Tree")
        return None, []
    
    # 统计修正情况
    correction_stats = {'attempted': 0, 'successful': 0, 'failed': 0}
    
    for iteration in range(max_iterations):
        # 1. 随机采样
        sample_state = sample_random_state()
        
        # 2. 找到最近的树节点
        nearest_node = find_nearest_node(tree_nodes, sample_state)
        
        # 3. 向采样点扩展
        new_state = steer(nearest_node.state, sample_state, max_step=V*DT)
        
        # 4. 检查是否在边界内
        if not is_state_valid(new_state[0], new_state[1]):
            continue
        
        # 5. 核心安全检查：不在安全集内则尝试修正
        if not is_state_in_safe_set(new_state, S_infinity):
            correction_stats['attempted'] += 1
            
            # 尝试修正到最近的安全状态
            safe_state, distance = find_nearest_safe_state(
                new_state, kdtree, safe_states_list, 
                threshold=1.5 * V * DT  # 修正距离阈值
            )
            
            if safe_state is not None:
                new_state = safe_state
                correction_stats['successful'] += 1
            else:
                # 修正失败，放弃当前采样
                correction_stats['failed'] += 1
                continue
        
        # 6. 添加到树中
        new_node = RRTNode(new_state, nearest_node)
        nearest_node.add_child(new_node)
        tree_nodes.append(new_node)
        
        # 7. 检查是否到达目标
        if is_goal_reached(new_state, goal_xy, goal_tolerance):
            print(f"安全RRT找到路径！迭代次数: {iteration + 1}")
            print(f"修正统计 - 尝试: {correction_stats['attempted']}, 成功: {correction_stats['successful']}, 失败: {correction_stats['failed']}")
            
            # 回溯路径（连续状态）
            path = []
            current = new_node
            while current is not None:
                path.append(current.state.copy())
                current = current.parent
            return path[::-1], tree_nodes
        
        # 进度显示
        if (iteration + 1) % 100 == 0:
            print(f"安全RRT迭代进度: {iteration + 1}/{max_iterations}, 修正成功率: {correction_stats['successful']}/{correction_stats['attempted'] if correction_stats['attempted'] > 0 else 1:.2f}")
    
    print("安全RRT未找到路径")
    print(f"最终修正统计 - 尝试: {correction_stats['attempted']}, 成功: {correction_stats['successful']}, 失败: {correction_stats['failed']}")
    return None, tree_nodes

def count_tree_branches(tree_nodes):
    """
    统计RRT树的分叉数
    Args:
        tree_nodes: RRT树节点列表
    Returns:
        int: 分叉数（拥有2个或更多子节点的节点数量）
    """
    branch_count = 0
    for node in tree_nodes:
        if len(node.children) >= 2:
            branch_count += 1
    return branch_count

def count_tree_total_nodes(tree_nodes):
    """
    统计RRT树的总节点数
    Args:
        tree_nodes: RRT树节点列表
    Returns:
        int: 树的总节点数
    """
    return len(tree_nodes)

# --- 4. 辅助函数定义 ---

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

def create_rrt_path_visualization(S_infinity, obstacle_indices, rrt_result, safe_rrt_result, start_continuous, goal_continuous, safe_angle_count):
    """RRT路径规划可视化 - 支持基线RRT和安全RRT对比显示"""
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
      # 绘制基线RRT
    if rrt_result and rrt_result[0] is not None:
        rrt_path, rrt_tree = rrt_result
          # 统计路径节点数、树分叉数和树总节点数
        path_node_count = len(rrt_path)
        tree_branch_count = count_tree_branches(rrt_tree)
        tree_total_nodes = count_tree_total_nodes(rrt_tree)
        print(f"基线RRT - 路径节点数: {path_node_count}, 树分叉数: {tree_branch_count}, 树总节点数: {tree_total_nodes}")
        
          # 1. 先绘制整个树结构（所有分叉）
        for node in rrt_tree:
            if node.parent is not None:
                # 绘制从父节点到当前节点的连线
                ax.plot([node.parent.state[0], node.state[0]], 
                       [node.parent.state[1], node.state[1]], 
                       color='#333333', linewidth=1.2, alpha=0.7)  # 深灰色线条
        
        # 2. 绘制最终路径（连续状态）
        rrt_xy = np.array([[state[0], state[1]] for state in rrt_path])
        
        # 绘制RRT主路径
        ax.plot(rrt_xy[:, 0], rrt_xy[:, 1], 
                color='#D32F2F', linewidth=4, 
                label=f'基线RRT路径 (节点:{path_node_count}, 分叉:{tree_branch_count}, 总:{tree_total_nodes})', 
                alpha=0.9, zorder=5)
        
        # 绘制关键点
        ax.plot(rrt_xy[:, 0], rrt_xy[:, 1], 
                color='#F44336', marker='o', markersize=5, linewidth=0, 
                label='基线RRT路径点', alpha=0.8, zorder=6)
          # 绘制方向箭头（根据节点总数决定是否绘制箭头）
        if len(rrt_path) <= 10:  # 节点数小于等于10，为所有点画箭头
            for i in range(len(rrt_path)):
                state = rrt_path[i]
                ax.arrow(state[0], state[1],
                         0.3 * np.cos(state[2]), 0.3 * np.sin(state[2]), 
                         head_width=0.15, head_length=0.15, fc="#D32F2F", ec="#D32F2F", 
                         alpha=0.8, zorder=7)
        # 如果节点数大于10，则不画箭头    # 绘制安全RRT路径  
    if safe_rrt_result and safe_rrt_result[0] is not None:
        safe_rrt_path, safe_rrt_tree = safe_rrt_result
        
        # 统计路径节点数、树分叉数和树总节点数
        safe_path_node_count = len(safe_rrt_path)
        safe_tree_branch_count = count_tree_branches(safe_rrt_tree)
        safe_tree_total_nodes = count_tree_total_nodes(safe_rrt_tree)
        print(f"安全RRT - 路径节点数: {safe_path_node_count}, 树分叉数: {safe_tree_branch_count}, 树总节点数: {safe_tree_total_nodes}")
        
          # 1. 先绘制整个树结构（所有分叉）
        for node in safe_rrt_tree:
            if node.parent is not None:
                ax.plot([node.parent.state[0], node.state[0]], 
                       [node.parent.state[1], node.state[1]], 
                       color='#555555', linewidth=1.2, alpha=0.6)  # 深灰色线条
        
        # 2. 绘制最终路径
        safe_rrt_xy = np.array([[state[0], state[1]] for state in safe_rrt_path])
        
        # 绘制实际的连续轨迹
        ax.plot(safe_rrt_xy[:, 0], safe_rrt_xy[:, 1], 
                color="#0741FF", linewidth=4, 
                label=f'安全RRT路径 (节点:{safe_path_node_count}, 分叉:{safe_tree_branch_count}, 总:{safe_tree_total_nodes})', 
                alpha=0.9, zorder=5)
        
        # 绘制关键点
        ax.plot(safe_rrt_xy[:, 0], safe_rrt_xy[:, 1], 
                color="#0521F1E9", marker='s', markersize=5, linewidth=0, 
                label='安全RRT路径点', alpha=0.8, zorder=6)
        
        # 绘制方向箭头（根据节点总数决定是否绘制箭头）
        if len(safe_rrt_path) <= 10:  # 节点数小于等于10，为所有点画箭头
            for i in range(len(safe_rrt_path)):
                state = safe_rrt_path[i]
                ax.arrow(state[0], state[1],
                         0.4 * np.cos(state[2]), 0.4 * np.sin(state[2]), 
                         head_width=0.2, head_length=0.2, fc="#070FFF", ec="#071CFF", 
                         alpha=0.8, zorder=7)
        # 如果节点数大于10，则不画箭头
    
    ax.plot(start_continuous[0], start_continuous[1], marker='o', color='green', markersize=12, label='起点')
    
    # 绘制目标区域（圆形区域，因为是3D柱状区域在2D的投影）
    goal_x, goal_y = goal_continuous[0], goal_continuous[1]
    goal_circle = patches.Circle((goal_x, goal_y), radius=1.0, 
                                facecolor='gold', alpha=0.3, edgecolor='goldenrod', linewidth=2)
    ax.add_patch(goal_circle)
    
    # 标记目标区域中心点
    ax.plot(goal_continuous[0], goal_continuous[1], marker='*', color='gold', markersize=25, label='目标区域中心', markeredgecolor='black', markeredgewidth=2)
    
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
        plt.Line2D([0], [0], color='#333333', linewidth=1.2, alpha=0.7, label='基线RRT树分叉'),
        plt.Line2D([0], [0], color='#555555', linewidth=1.2, alpha=0.6, label='安全RRT树分叉'),
        patches.Patch(color='gold', alpha=0.3, label='目标区域 (3D柱状)')
    ]
    
    # 获取线条图例
    handles, labels = ax.get_legend_handles_labels()
    # 将图例放在图形外部右侧
    ax.legend(handles=handles + legend_patches, fontsize=14, markerscale=0.8,
             bbox_to_anchor=(1.35, 1), loc='upper left')
    ax.set_xlim(X_MIN - 1, X_MAX + 1)
    ax.set_ylim(Y_MIN - 1, Y_MAX + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('基于安全状态转移图的独轮车RRT路径规划', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存高分辨率图片
    plt.savefig('rrt_path_planning_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
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
    ax.legend(handles=handles + legend_elements, fontsize=14, markerscale=0.8,
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
    plt.savefig('rrt_safe_angle_arrows.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# --- 5. 主程序 ---
if __name__ == "__main__":    
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
    obstacle_indices.add((3, 17))
    obstacle_indices.add((5, 17))    # 1. 计算鲁棒安全集
    S_infinity, safe_actions_map = compute_robust_safe_set_optimized(obstacle_indices, W)

    if not S_infinity:
        print("鲁棒安全集为空，无法进行路径规划")
        exit()
    
    # 2. RRT算法实验
    print("开始基线RRT路径规划...")
    rrt_result = rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, safe_actions_map)
    
    print("开始安全RRT路径规划...")
    safe_rrt_result = safe_rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, safe_actions_map)

    # 3. 计算安全角度统计
    safe_angle_count = {}
    max_angles = len(theta_space)
    
    for ix, iy, itheta in S_infinity:
        key = (ix, iy)
        if key not in safe_angle_count:
            safe_angle_count[key] = 0
        safe_angle_count[key] += 1    
    
    # 4. 生成可视化
    create_rrt_path_visualization(S_infinity, obstacle_indices, rrt_result, safe_rrt_result,
                                  start_continuous, goal_continuous, safe_angle_count)
    
    #create_safety_angle_visualization(S_infinity, obstacle_indices, start_continuous, goal_continuous)
