"""
核心数学模型和基础工具函数
"""
import numpy as np
from config import *

def is_state_valid(x, y):
    """检查状态是否在有效边界内（不与边界碰撞）"""
    return X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX

def discretize_state(x, y, theta):
    """将连续状态离散化为网格索引"""
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
    """将网格索引转换为连续状态"""
    return np.array([x_space[ix], y_space[iy], theta_space[itheta]])

def unicycle_model(state, omega):
    """Unicycle运动学模型"""
    x, y, theta = state
    theta_next = theta + omega * DT
    theta_mid = theta + omega * DT / 2
    x_next = x + V * np.cos(theta_mid) * DT
    y_next = y + V * np.sin(theta_mid) * DT
    
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

def heuristic(state_indices, goal_indices):
    """A*搜索的启发式函数"""
    x1, y1, _ = indices_to_state(*state_indices)
    x2, y2, _ = indices_to_state(*goal_indices)
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
