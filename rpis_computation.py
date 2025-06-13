"""
鲁棒安全集计算模块
"""
import numpy as np
from tqdm import tqdm
from config import *
from core_models import *

def compute_robust_safe_set(obstacle_indices, W):
    """标准版本的鲁棒安全集计算"""
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
                    
                    # 检查路径是否与障碍物碰撞
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
                Sk_plus_1.add(s_indices)
        
        # 检查收敛
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
                    break  # 找到一个安全动作就够了
        
        # 检查收敛
        if Sk_plus_1 == Sk:
            print(f"鲁棒安全集已收敛: {len(Sk_plus_1)} 个状态")
            return Sk_plus_1
        
        Sk = Sk_plus_1
        k += 1
