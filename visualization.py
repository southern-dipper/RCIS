"""
可视化和路径处理模块
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from config import *
from core_models import *

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

def create_original_path_visualization(S_infinity, obstacle_indices, baseline_path, graph_path, start_continuous, goal_continuous, safe_angle_count):
    """恢复原来的单独大图可视化"""
    # 创建原来的大图，增大尺寸和DPI
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
      # 绘制标准A*路径
    if baseline_path:
        # 获取完整的路径格子和连续轨迹
        baseline_cells, baseline_trajectory = get_path_cells_and_trajectory(baseline_path)
          # 绘制标准A*的连续轨迹
        trajectory_array = np.array(baseline_trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                color='#D32F2F', linewidth=3, label='标准A*轨迹', alpha=0.9)
        
        # 绘制关键点
        path_states = np.array([indices_to_state(*p) for p in baseline_path])
        ax.plot(path_states[:, 0], path_states[:, 1], 
                color='#F44336', marker='s', markersize=4, linewidth=0, 
                label='标准A*路径点', alpha=0.8)
    
    # 绘制安全A*路径  
    if graph_path:
        # 获取完整的路径格子和连续轨迹
        graph_cells, graph_trajectory = get_path_cells_and_trajectory(graph_path)
          # 绘制所有路径经过的格子
        for ix, iy in graph_cells:
            rect = patches.Rectangle(
                (x_space[ix] - X_STEP/2, y_space[iy] - Y_STEP/2), 
                X_STEP, Y_STEP, 
                facecolor="#59A8F6FF", alpha=0.6, edgecolor="#5490F7FF", linewidth=1
            )
            ax.add_patch(rect)
        
        # 绘制实际的连续轨迹
        trajectory_array = np.array(graph_trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                color="#0741FF", linewidth=3, label='安全A*轨迹', alpha=0.9)
        
        # 绘制关键点
        path_states = np.array([indices_to_state(*p) for p in graph_path])
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


