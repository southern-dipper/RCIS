import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import time
import matplotlib.colors as mcolors
import random
from typing import List, Tuple, Set, Optional, Dict, Any

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

# RRT算法参数
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
    sample_distance = grid_diagonal * 0.2  # 采样间距
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
    """计算鲁棒安全集"""
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
                    break  # 找到一个安全动作就够了
          # 检查收敛
        if Sk_plus_1 == Sk:
            print(f"鲁棒安全集已收敛: {len(Sk_plus_1)} 个状态")
            return Sk_plus_1
        
        Sk = Sk_plus_1
        k += 1

# --- 3. RRT 路径规划算法 ---

def angle_distance(theta1, theta2):
    """计算两个角度之间的最短距离（考虑循环性）"""
    diff = abs(theta1 - theta2)
    return min(diff, 2 * np.pi - diff)

class RRTNode:
    """RRT树节点"""
    def __init__(self, state, parent=None):
        self.state = state  # 连续状态 [x, y, theta]
        self.parent = parent
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)

def is_goal_reached(state, goal_xy, tolerance=1.0):
    """检查是否到达目标区域（圆形区域，半径=tolerance）
    
    Args:
        state: 当前状态 [x, y, theta]
        goal_xy: 目标中心点 [x, y]
        tolerance: 目标区域半径（默认1.0，形成半径为1.0的圆形区域）
    
    Returns:
        bool: 是否到达目标区域
    
    注意：使用欧几里得距离，形成以goal_xy为中心的圆形区域
    """
    dx = state[0] - goal_xy[0]
    dy = state[1] - goal_xy[1]
    distance = np.sqrt(dx*dx + dy*dy)
    return distance <= tolerance

def rrt_search(start_indices: Tuple[int, int, int], goal_xy_indices: Tuple[int, int], 
               S_infinity: Set[Tuple[int, int, int]], obstacle_indices: Set[Tuple[int, int]], 
               max_iterations: int = 1000, goal_tolerance: float = 1.0, 
               goal_bias_prob: float = 0.1) -> Tuple[Optional[List], List, float]:
    """
    基线RRT搜索算法 - 2D采样 + 碰撞检查
    Returns:
        tuple: (path, tree_nodes, search_time) 其中path是连续状态列表，tree_nodes是所有树节点，search_time是搜索时间
    """
    start_time = time.time()
    
    # 转换为连续状态
    start_state = indices_to_state(*start_indices)
    goal_xy = np.array([x_space[goal_xy_indices[0]], y_space[goal_xy_indices[1]]])
      # 初始化树
    root = RRTNode(start_state)
    tree_nodes = [root]
    
    for iteration in range(max_iterations):
        # 1. 目标偏向采样策略
        if np.random.random() < goal_bias_prob:
            # 以一定概率直接朝向目标采样
            q_rand = np.array([goal_xy[0], goal_xy[1]])
        else:
            # 正常随机采样
            q_rand = np.array([
                np.random.uniform(X_MIN, X_MAX),
                np.random.uniform(Y_MIN, Y_MAX)
            ])
        
        # 2. 找到最近的树节点（2D距离）
        min_dist = float('inf')
        nearest_node = None
        for node in tree_nodes:
            dist = np.sqrt((node.state[0] - q_rand[0])**2 + (node.state[1] - q_rand[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        # 3. 计算朝向采样点的角度
        dx = q_rand[0] - nearest_node.state[0]
        dy = q_rand[1] - nearest_node.state[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:  # 避免除零
            continue
        theta_toward_sample = np.arctan2(dy, dx)
        
        # 4. 状态扩展
        max_step = V * DT
        if min_dist < max_step:
            q_new = q_rand
        else:
            direction = (q_rand - nearest_node.state[:2]) / min_dist
            q_new = nearest_node.state[:2] + max_step * direction
        
        # 组合新的3D状态
        new_state = np.array([q_new[0], q_new[1], theta_toward_sample])
        
        # 5. 检查是否在边界内
        if not is_state_valid(new_state[0], new_state[1]):
            continue
        
        # 6. 检查路径碰撞（基线RRT需要做碰撞检查）
        if check_path_collision(nearest_node.state, new_state, obstacle_indices):
            continue
        
        # 7. 添加到树中
        new_node = RRTNode(new_state, nearest_node)
        nearest_node.add_child(new_node)
        tree_nodes.append(new_node)
          # 8. 检查是否到达目标
        if is_goal_reached(new_state, goal_xy, goal_tolerance):
            end_time = time.time()
            search_time = end_time - start_time
            #print(f"迭代次数: {iteration + 1}, 搜索时间: {search_time:.3f}秒")
            # 回溯路径（连续状态）
            path = []
            current = new_node
            while current is not None:
                path.append(current.state.copy())  # 直接使用连续状态
                current = current.parent
            return path[::-1], tree_nodes, search_time  # 返回路径、树和搜索时间
        
        # 进度显示
        if (iteration + 1) % 100 == 0:
            print(f"基线RRT迭代进度: {iteration + 1}/{max_iterations}")    
    end_time = time.time()
    search_time = end_time - start_time
    print(f"基线RRT未找到路径，搜索时间: {search_time:.3f}秒")
    return None, tree_nodes, search_time

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

def safe_rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, max_iterations=3000, goal_tolerance=1.0, goal_bias_prob=0.1):
    """
    改进的安全RRT搜索算法 - 基于2D+角度调整策略
    Returns:
        tuple: (path, tree_nodes, search_time) 其中path是连续状态列表，tree_nodes是所有树节点，search_time是搜索时间
    """
    start_time = time.time()
    
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
      # 统计修正情况
    correction_stats = {'attempted': 0, 'successful': 0, 'failed': 0}
    
    for iteration in range(max_iterations):
        # 1. 目标偏向采样策略
        if np.random.random() < goal_bias_prob:
            # 以一定概率直接朝向目标采样
            q_rand = np.array([goal_xy[0], goal_xy[1]])
        else:
            # 正常随机采样
            q_rand = np.array([
                np.random.uniform(X_MIN, X_MAX),
                np.random.uniform(Y_MIN, Y_MAX)
            ])
        
        # 2. 找到最近的树节点（2D距离）
        min_dist = float('inf')
        nearest_node = None
        for node in tree_nodes:
            dist = np.sqrt((node.state[0] - q_rand[0])**2 + (node.state[1] - q_rand[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        # 3. 计算期望角度
        dx = q_rand[0] - nearest_node.state[0]
        dy = q_rand[1] - nearest_node.state[1]
        theta_desired = np.arctan2(dy, dx)
        
        # 4. 角度安全性检查与调整
        q_near_pos = nearest_node.state[:2]
        ix, iy, itheta = discretize_state(q_near_pos[0], q_near_pos[1], theta_desired)
        
        theta_target = None
        if (ix, iy, itheta) in S_infinity:
            theta_target = theta_desired
        else:
            # WARNING: 硬编码角度搜索，假设5个动作，优先右转
            for offset in [-1, 1, -2, 2]:
                candidate_theta_idx = (itheta + offset) % len(theta_space)
                if (ix, iy, candidate_theta_idx) in S_infinity:
                    theta_target = theta_space[candidate_theta_idx]
                    break
        
        if theta_target is None:
            continue  # 没找到安全角度，放弃当前采样
          # 5. 状态扩展
        max_step = V * DT
        if min_dist < max_step:
            q_new = q_rand
        else:
            direction = (q_rand - q_near_pos) / min_dist
            q_new = q_near_pos + max_step * direction
        
        # 验证扩展后状态的安全性
        new_state = np.array([q_new[0], q_new[1], theta_target])
        ix_new, iy_new, itheta_new = discretize_state(*new_state)
        
                
        if (ix_new, iy_new, itheta_new) in S_infinity:
            # 添加到树中
            # # 添加路径碰撞检查（
            path_collision = check_path_collision(nearest_node.state, new_state, obstacle_indices)
            if path_collision:
                continue
            new_node = RRTNode(new_state, nearest_node)
            nearest_node.add_child(new_node)
            tree_nodes.append(new_node)
            #     print(f"[调试] 安全RRT检测到路径碰撞！")
            #     print(f"  起点状态: {nearest_node.state}")
            #     print(f"  终点状态: {new_state}")
            #     start_indices = discretize_state(*nearest_node.state)
            #     end_indices = discretize_state(*new_state)
            #     print(f"  起点离散化: {start_indices}, 在安全集内: {start_indices in S_infinity}")
            #     print(f"  终点离散化: {end_indices}, 在安全集内: {end_indices in S_infinity}")
            #     print(f"  路径长度: {np.sqrt((new_state[0] - nearest_node.state[0])**2 + (new_state[1] - nearest_node.state[1])**2):.3f}")

        else:
            #直接放弃
            # continue

            # 局部搜索+放弃策略
            correction_stats['attempted'] += 1
            
            best_safe_state = None
            min_distance = float('inf')
            
            for offset_x in [-1, 0, 1]:
                for offset_y in [-1, 0, 1]:
                    for offset_theta in [-1, 0, 1]:
                        ix_candidate = ix_new + offset_x
                        iy_candidate = iy_new + offset_y
                        itheta_candidate = (itheta_new + offset_theta) % len(theta_space)
                        
                        if (ix_candidate, iy_candidate, itheta_candidate) in S_infinity:
                            candidate_state = indices_to_state(ix_candidate, iy_candidate, itheta_candidate)
                            
                            # 计算3D欧氏距离（步长权重对齐）
                            dx = (candidate_state[0] - q_new[0]) / X_STEP
                            dy = (candidate_state[1] - q_new[1]) / Y_STEP
                            dtheta = angle_distance(candidate_state[2], theta_target) / THETA_STEP
                            distance_3d = np.sqrt(dx*dx + dy*dy + dtheta*dtheta)
                            
                            if distance_3d < min_distance:
                                min_distance = distance_3d
                                best_safe_state = candidate_state
            
            if best_safe_state is not None:
                # 检查距离是否合理
                distance_2d = np.sqrt((best_safe_state[0] - nearest_node.state[0])**2 + 
                                    (best_safe_state[1] - nearest_node.state[1])**2)
                if distance_2d < 1 * max_step:
                    path_collision = check_path_collision(nearest_node.state, new_state, obstacle_indices)
                    if path_collision:
                        continue
                    new_node = RRTNode(best_safe_state, nearest_node)
                    nearest_node.add_child(new_node)
                    tree_nodes.append(new_node)
                    correction_stats['successful'] += 1
                else:
                    correction_stats['failed'] += 1
                    continue
            else:
                correction_stats['failed'] += 1
                continue
          # 检查是否到达目标
        if is_goal_reached(tree_nodes[-1].state, goal_xy, goal_tolerance):
            end_time = time.time()
            search_time = end_time - start_time
            #print(f"迭代次数: {iteration + 1}, 搜索时间: {search_time:.3f}秒")
            
            # 回溯路径
            path = []
            current = tree_nodes[-1]
            while current is not None:
                path.append(current.state.copy())
                current = current.parent
            return path[::-1], tree_nodes, search_time
        
        
    end_time = time.time()
    search_time = end_time - start_time
    print(f"改进安全RRT未找到路径，搜索时间: {search_time:.3f}秒")
    print(f"最终修正统计 - 尝试: {correction_stats['attempted']}, 成功: {correction_stats['successful']}, 失败: {correction_stats['failed']}")
    return None, tree_nodes, search_time

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
            if node.parent is not None:                # 绘制从父节点到当前节点的连线
                ax.plot([node.parent.state[0], node.state[0]], 
                       [node.parent.state[1], node.state[1]], 
                       color='#FFB3B3', linewidth=1.2, alpha=0.8)  # 淡红色线条
        
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
            if node.parent is not None:                ax.plot([node.parent.state[0], node.state[0]], 
                       [node.parent.state[1], node.state[1]], 
                       color='#B3D9FF', linewidth=1.2, alpha=0.8)  # 淡蓝色线条
        
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
                         alpha=0.8, zorder=7)        # 如果节点数大于10，则不画箭头
    
    ax.plot(start_continuous[0], start_continuous[1], marker='o', color='green', markersize=12, label='起点')
    
    # 绘制目标区域（圆形区域，半径1.0，与目标检测逻辑一致）
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
        patches.Patch(color='#000000', label='障碍物'),        plt.Line2D([0], [0], color='#FFB3B3', linewidth=1.2, alpha=0.8, label='基线RRT树分叉'),
        plt.Line2D([0], [0], color='#B3D9FF', linewidth=1.2, alpha=0.8, label='安全RRT树分叉'),
        patches.Patch(color='gold', alpha=0.3, label='目标区域 (圆形r=1.0)')
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
    #plt.savefig('rrt_path_planning_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# --- 5. 主程序 ---
if __name__ == "__main__":    
    # RRT算法参数
    GOAL_BIAS_PROBABILITY = 0.2  # 目标偏向采样概率
    
    # 定义问题
    start_continuous = np.array([0., 0., np.pi / 4])  
    goal_continuous = np.array([10., 10.])
    
    start_indices = discretize_state(*start_continuous)
    goal_xy_indices = discretize_state(goal_continuous[0], goal_continuous[1], 0)[:2]

    # 生成简单的障碍物
    obstacle_indices = set()
    
    # 添加一些简单的障碍物
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
    
    S_infinity = compute_robust_safe_set_optimized(obstacle_indices, W)

    if not S_infinity:
        print("鲁棒安全集为空，无法进行路径规划")
        exit()    # 2. RRT算法实验    print("开始基线RRT路径规划...")
    rrt_result = rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, goal_bias_prob=GOAL_BIAS_PROBABILITY)
    rrt_path, rrt_tree, rrt_time = rrt_result if rrt_result[0] is not None else (None, rrt_result[1], rrt_result[2])
    
    print("开始安全RRT路径规划...")
    safe_rrt_result = safe_rrt_search(start_indices, goal_xy_indices, S_infinity, obstacle_indices, goal_bias_prob=GOAL_BIAS_PROBABILITY)
    safe_rrt_path, safe_rrt_tree, safe_rrt_time = safe_rrt_result if safe_rrt_result[0] is not None else (None, safe_rrt_result[1], safe_rrt_result[2])
    
    # 输出时间对比
    print(f"\n=== 搜索时间对比 ===")
    print(f"基线RRT搜索时间: {rrt_time:.3f}秒")
    print(f"安全RRT搜索时间: {safe_rrt_time:.3f}秒")
    if rrt_time > 0:
        print(f"安全RRT相对基线RRT的时间比: {safe_rrt_time/rrt_time:.2f}x")
    
    # 重新包装结果以保持兼容性
    rrt_result = (rrt_path, rrt_tree)
    safe_rrt_result = (safe_rrt_path, safe_rrt_tree)

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
    

"""
基线RRT：保持原有的2D路径规划逻辑

改进的安全RRT：
1. 在2D空间采样 q_rand(x, y)
2. 寻找最近树节点 q_near，记录距离 d
3. 计算 q_near 指向 q_rand 的角度 θ_desired

4. 角度安全性检查与调整：
   离散化 (q_near.x, q_near.y, θ_desired) 为 (ix, iy, iθ)
     θ_target = None
   if (ix, iy, iθ) 在安全集内:
       θ_target = θ_desired  # 期望角度本身就安全
   else:
       # 在同一位置(ix, iy)寻找与θ_desired最接近的安全角度
       # WARNING: 这里硬编码了[-1,1,-2,2]，默认动作空间是5个动作，采用优先右转策略
       使用固定搜索序列找到与θ_desired最近的安全角度:
           for offset in [-1, 1, -2, 2]:  # 优先右转策略
               candidate_theta_idx = (iθ + offset) % len(theta_space)  # 处理角度循环
               if (ix, iy, candidate_theta_idx) 在安全集内:
                   θ_target = theta_space[candidate_theta_idx]
                   break
       
       if 没找到安全角度:
           放弃当前采样，continue

5. 状态扩展：
   if θ_target is not None:
       if d < 最大允许步长:
           q_new = q_rand  # 直接扩展到采样点
       else:
           # 在2D空间中按比例缩放到最大步长
           direction = (q_rand - q_near) / d  # 单位方向向量
           q_new = q_near + 最大允许步长 * direction
       
       # 验证扩展后状态的安全性
       离散化 (q_new.x, q_new.y, θ_target) 为 (ix_new, iy_new, iθ_target)
       
       if (ix_new, iy_new, iθ_target) 在安全集内:
           # 直接添加到树中（依赖安全集保证安全性，不做路径碰撞检查）
           将 (q_new.x, q_new.y, θ_target) 添加到树中
       else:
           # 方案2：局部搜索+放弃策略
           # 在q_new附近3x3x3邻域内搜索安全状态，使用3D欧氏距离
           best_safe_state = None
           min_distance = float('inf')
           
           for offset_x in [-1, 0, 1]:
               for offset_y in [-1, 0, 1]:
                   for offset_theta in [-1, 0, 1]:
                       ix_candidate = ix_new + offset_x
                       iy_candidate = iy_new + offset_y
                       iθ_candidate = (iθ_target + offset_theta) % len(theta_space)
                       
                       if (ix_candidate, iy_candidate, iθ_candidate) 在安全集内:
                           candidate_state = indices_to_state(ix_candidate, iy_candidate, iθ_candidate)
                           
                           # 计算3D欧氏距离（角度维度权重与位置维度对齐）
                           # 使用统一的步长权重：X_STEP, Y_STEP, THETA_STEP
                           dx = (candidate_state[0] - q_new[0]) / X_STEP
                           dy = (candidate_state[1] - q_new[1]) / Y_STEP  
                           dtheta = angle_distance(candidate_state[2], θ_target) / THETA_STEP
                           distance_3d = np.sqrt(dx*dx + dy*dy + dtheta*dtheta)
                           
                           if distance_3d < min_distance:
                               min_distance = distance_3d
                               best_safe_state = candidate_state
           
           if best_safe_state is not None:
               # 检查是否在合理距离内（使用2D距离检查与q_near的关系）
               distance_2d = np.sqrt((best_safe_state[0] - q_near[0])**2 + (best_safe_state[1] - q_near[1])**2)
               if distance_2d < 1.5 * 最大允许步长:
                   # 直接添加到树中（依赖安全集保证安全性）
                   将 best_safe_state 添加到树中
               else:
                   放弃当前采样  # 距离太远
           else:
               放弃当前采样  # 没找到合适的安全状态
"""
