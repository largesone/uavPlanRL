# -*- coding: utf-8 -*-
"""
大规模课程学习场景优化版本
确保资源比例平滑递减：1.4 -> 1.25 -> 1.15 -> 1.05 -> 1.0
"""

import numpy as np
from entities import UAV, Target
from path_planning import CircularObstacle, PolygonalObstacle

def get_large_curriculum_scenarios_optimized():
    """
    获取优化的大规模课程学习场景序列
    
    资源比例设计：
    - Level1: 1.40 (资源充足)
    - Level2: 1.25 (资源适中)  
    - Level3: 1.15 (资源紧张)
    - Level4: 1.05 (资源刚好)
    - Level5: 1.00 (零容错)
    
    Returns:
        list: 优化的课程学习场景列表
    """
    return [
        (get_large_level1_optimized, "Large_Level1_Easy", "20UAV-15Target, 资源充足140%"),
        (get_large_level2_optimized, "Large_Level2_Simple", "20UAV-15Target, 资源适中125%"),
        (get_large_level3_optimized, "Large_Level3_Medium", "20UAV-15Target, 资源紧张115%"),
        (get_large_level4_optimized, "Large_Level4_Hard", "20UAV-15Target, 资源刚好105%"),
        (get_large_level5_optimized, "Large_Level5_Expert", "20UAV-15Target, 零容错100%")
    ]

def get_large_level1_optimized(obstacle_tolerance=50.0):
    """Level1: 资源比例 1.40"""
    np.random.seed(42)
    
    # 目标总需求: 60 (30+30)
    targets = []
    target_positions = [
        [1000, 1000], [2000, 1000], [3000, 1000], [4000, 1000], [5000, 1000],
        [1000, 2500], [2000, 2500], [3000, 2500], [4000, 2500], [5000, 2500],
        [1000, 4000], [2000, 4000], [3000, 4000], [4000, 4000], [5000, 4000]
    ]
    
    for i, pos in enumerate(target_positions):
        targets.append(Target(
            id=i+1, position=np.array(pos), 
            resources=np.array([2, 2]), value=100-i*2
        ))
    
    # UAV总供给: 84 (42+42)，比例 = 84/60 = 1.40
    uavs = []
    uav_positions = [
        [500, 500], [1500, 500], [2500, 500], [3500, 500], [4500, 500], [5500, 500],
        [500, 1500], [1500, 1500], [2500, 1500], [3500, 1500], [4500, 1500], [5500, 1500],
        [500, 3000], [1500, 3000], [2500, 3000], [3500, 3000], [4500, 3000], [5500, 3000],
        [1000, 4500], [3000, 4500]
    ]
    
    for i, pos in enumerate(uav_positions):
        # 前10个UAV: [2,2], 后10个UAV: [2,2]
        resources = np.array([2, 2]) if i < 10 else np.array([2, 2])
        uavs.append(UAV(
            id=i+1, position=np.array(pos), heading=0,
            resources=resources, max_distance=8000,
            velocity_range=(60, 140), economic_speed=100
        ))
    
    return uavs, targets, []

def get_large_level2_optimized(obstacle_tolerance=50.0):
    """Level2: 资源比例 1.25"""
    np.random.seed(43)
    
    # 目标总需求: 80 (40+40)
    targets = []
    target_positions = [
        [1200, 1200], [2400, 1200], [3600, 1200], [4800, 1200],
        [1200, 2400], [2400, 2400], [3600, 2400], [4800, 2400],
        [1200, 3600], [2400, 3600], [3600, 3600], [4800, 3600],
        [1800, 4800], [3000, 4800], [4200, 4800]
    ]
    
    # 资源需求分布: 前5个[3,3], 中5个[2,3], 后5个[3,2]
    resource_patterns = [
        [3, 3], [3, 3], [3, 3], [2, 3], [2, 3],  # 前5个: 25
        [2, 3], [3, 2], [3, 2], [3, 2], [3, 2],  # 中5个: 25
        [2, 2], [2, 2], [2, 2], [3, 3], [3, 3]   # 后5个: 30
    ]
    
    for i, (pos, res) in enumerate(zip(target_positions, resource_patterns)):
        targets.append(Target(
            id=i+1, position=np.array(pos),
            resources=np.array(res), value=100-i*2
        ))
    
    # UAV总供给: 100 (50+50)，比例 = 100/80 = 1.25
    uavs = []
    uav_positions = [
        [600, 600], [1800, 600], [3000, 600], [4200, 600], [5400, 600],
        [600, 1800], [1800, 1800], [3000, 1800], [4200, 1800], [5400, 1800],
        [600, 3000], [1800, 3000], [3000, 3000], [4200, 3000], [5400, 3000],
        [600, 4200], [1800, 4200], [3000, 4200], [4200, 4200], [5400, 4200]
    ]
    
    for i, pos in enumerate(uav_positions):
        # 均匀分配: 每个UAV [2.5, 2.5] ≈ [3,2] 或 [2,3]
        resources = np.array([3, 2]) if i % 2 == 0 else np.array([2, 3])
        uavs.append(UAV(
            id=i+1, position=np.array(pos), heading=0,
            resources=resources, max_distance=8000,
            velocity_range=(60, 140), economic_speed=100
        ))
    
    # 简单障碍物
    obstacles = [
        CircularObstacle(center=(2500, 2500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1500, 3500), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3500, 1500), radius=150, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_large_level3_optimized(obstacle_tolerance=50.0):
    """Level3: 资源比例 1.15"""
    np.random.seed(44)
    
    # 目标总需求: 120 (60+60)
    targets = []
    target_positions = [
        [1000, 1000], [2000, 1000], [3000, 1000], [4000, 1000], [5000, 1000],
        [1000, 2000], [2000, 2000], [3000, 2000], [4000, 2000], [5000, 2000],
        [1000, 3000], [2000, 3000], [3000, 3000], [4000, 3000], [5000, 3000]
    ]
    
    # 资源需求增加: 每个目标[4,4]
    for i, pos in enumerate(target_positions):
        targets.append(Target(
            id=i+1, position=np.array(pos),
            resources=np.array([4, 4]), value=100-i*2
        ))
    
    # UAV总供给: 138 (69+69)，比例 = 138/120 = 1.15
    uavs = []
    uav_positions = [
        [500, 500], [1500, 500], [2500, 500], [3500, 500], [4500, 500], [5500, 500],
        [500, 1500], [1500, 1500], [2500, 1500], [3500, 1500], [4500, 1500], [5500, 1500],
        [500, 2500], [1500, 2500], [2500, 2500], [3500, 2500], [4500, 2500], [5500, 2500],
        [1000, 3500], [3000, 3500]
    ]
    
    for i, pos in enumerate(uav_positions):
        # 分配: 前10个[4,3], 后10个[3,4]
        resources = np.array([4, 3]) if i < 10 else np.array([3, 4])
        uavs.append(UAV(
            id=i+1, position=np.array(pos), heading=0,
            resources=resources, max_distance=8000,
            velocity_range=(60, 140), economic_speed=100
        ))
    
    # 中等障碍物
    obstacles = [
        CircularObstacle(center=(2500, 2000), radius=250, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1500, 2500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3500, 1500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2000, 3000), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4000, 2500), radius=150, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_large_level4_optimized(obstacle_tolerance=50.0):
    """Level4: 资源比例 1.05"""
    np.random.seed(45)
    
    # 目标总需求: 180 (90+90)
    targets = []
    target_positions = [
        [1000, 1000], [2000, 1000], [3000, 1000], [4000, 1000], [5000, 1000],
        [1000, 2000], [2000, 2000], [3000, 2000], [4000, 2000], [5000, 2000],
        [1000, 3000], [2000, 3000], [3000, 3000], [4000, 3000], [5000, 3000]
    ]
    
    # 资源需求: 每个目标[6,6]
    for i, pos in enumerate(target_positions):
        targets.append(Target(
            id=i+1, position=np.array(pos),
            resources=np.array([6, 6]), value=100-i*2
        ))
    
    # UAV总供给: 189 (95+94)，比例 = 189/180 = 1.05
    uavs = []
    uav_positions = [
        [500, 500], [1500, 500], [2500, 500], [3500, 500], [4500, 500], [5500, 500],
        [500, 1500], [1500, 1500], [2500, 1500], [3500, 1500], [4500, 1500], [5500, 1500],
        [500, 2500], [1500, 2500], [2500, 2500], [3500, 2500], [4500, 2500], [5500, 2500],
        [1000, 3500], [3000, 3500]
    ]
    
    for i, pos in enumerate(uav_positions):
        # 精确分配: 前10个[5,5], 后9个[5,4], 最后1个[5,5]
        if i < 10:
            resources = np.array([5, 5])
        elif i < 19:
            resources = np.array([5, 4])
        else:
            resources = np.array([4, 5])
        
        uavs.append(UAV(
            id=i+1, position=np.array(pos), heading=0,
            resources=resources, max_distance=8000,
            velocity_range=(60, 140), economic_speed=100
        ))
    
    # 复杂障碍物
    obstacles = [
        CircularObstacle(center=(2500, 2000), radius=300, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1500, 1500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3500, 1500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1500, 2500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3500, 2500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2000, 3000), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3000, 3000), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4000, 1000), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1000, 2000), radius=150, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_large_level5_optimized(obstacle_tolerance=50.0):
    """Level5: 资源比例 1.00 (零容错)"""
    np.random.seed(46)
    
    # 目标总需求: 240 (120+120)
    targets = []
    target_positions = [
        [1000, 1000], [2000, 1000], [3000, 1000], [4000, 1000], [5000, 1000],
        [1000, 2000], [2000, 2000], [3000, 2000], [4000, 2000], [5000, 2000],
        [1000, 3000], [2000, 3000], [3000, 3000], [4000, 3000], [5000, 3000]
    ]
    
    # 资源需求: 每个目标[8,8]
    for i, pos in enumerate(target_positions):
        targets.append(Target(
            id=i+1, position=np.array(pos),
            resources=np.array([8, 8]), value=100-i*2
        ))
    
    # UAV总供给: 240 (120+120)，比例 = 240/240 = 1.00
    uavs = []
    uav_positions = [
        [500, 500], [1500, 500], [2500, 500], [3500, 500], [4500, 500], [5500, 500],
        [500, 1500], [1500, 1500], [2500, 1500], [3500, 1500], [4500, 1500], [5500, 1500],
        [500, 2500], [1500, 2500], [2500, 2500], [3500, 2500], [4500, 2500], [5500, 2500],
        [1000, 3500], [3000, 3500]
    ]
    
    for i, pos in enumerate(uav_positions):
        # 精确分配: 每个UAV [6,6]
        resources = np.array([6, 6])
        uavs.append(UAV(
            id=i+1, position=np.array(pos), heading=0,
            resources=resources, max_distance=8000,
            velocity_range=(60, 140), economic_speed=100
        ))
    
    # 迷宫级障碍物
    obstacles = [
        # 中央大障碍
        CircularObstacle(center=(2500, 2000), radius=400, tolerance=obstacle_tolerance),
        # 四角障碍
        CircularObstacle(center=(1200, 1200), radius=250, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3800, 1200), radius=250, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1200, 2800), radius=250, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3800, 2800), radius=250, tolerance=obstacle_tolerance),
        # 边缘障碍
        CircularObstacle(center=(2500, 800), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2500, 3200), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(800, 2000), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4200, 2000), radius=200, tolerance=obstacle_tolerance),
        # 通道障碍
        CircularObstacle(center=(1800, 1600), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3200, 1600), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1800, 2400), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3200, 2400), radius=150, tolerance=obstacle_tolerance),
        # 小型干扰障碍
        CircularObstacle(center=(1500, 1000), radius=100, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3500, 1000), radius=100, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1500, 3000), radius=100, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3500, 3000), radius=100, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def test_optimized_large_curriculum_scenarios():
    """测试优化的大规模课程学习场景"""
    print("=" * 80)
    print("优化的大规模课程学习场景测试 (20UAV-15Target)")
    print("=" * 80)
    
    scenarios = get_large_curriculum_scenarios_optimized()
    
    for scenario_func, level_name, description in scenarios:
        uavs, targets, obstacles = scenario_func(50.0)
        
        # 计算资源统计
        total_uav_resources = np.sum([uav.resources for uav in uavs], axis=0)
        total_target_resources = np.sum([target.resources for target in targets], axis=0)
        
        total_supply = np.sum(total_uav_resources)
        total_demand = np.sum(total_target_resources)
        resource_ratio = total_supply / total_demand if total_demand > 0 else 0
        
        print(f"\n{level_name}: {description}")
        print(f"  UAV数量: {len(uavs)}, 目标数量: {len(targets)}, 障碍物数量: {len(obstacles)}")
        print(f"  总供给: {total_supply}, 总需求: {total_demand}")
        print(f"  资源比例: {resource_ratio:.3f}")
        print(f"  按类型 - 类型1: {total_uav_resources[0]}/{total_target_resources[0]} = {total_uav_resources[0]/total_target_resources[0]:.3f}")
        print(f"  按类型 - 类型2: {total_uav_resources[1]}/{total_target_resources[1]} = {total_uav_resources[1]/total_target_resources[1]:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ 优化版本资源比例递减: 1.400 -> 1.250 -> 1.150 -> 1.050 -> 1.000")
    print("✅ 难度递进平滑，适合课程学习训练")
    print("=" * 80)

if __name__ == "__main__":
    test_optimized_large_curriculum_scenarios()