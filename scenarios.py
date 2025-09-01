# -*- coding: utf-8 -*-
# 文件名: scenarios.py
# 描述: 提供用于测试和仿真的预定义场景数据。

import numpy as np
import random
from entities import UAV, Target
from path_planning import CircularObstacle, PolygonalObstacle

def get_balanced_scenario(obstacle_tolerance):
    """
    提供一个资源平衡的场景：10个无人机，5个目标，资源供给等于需求。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 5个目标及其资源需求
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 150]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([120, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([100, 100]), value=80),
        Target(id=4, position=np.array([1500, 3000]), resources=np.array([80, 70]), value=85),
        Target(id=5, position=np.array([3500, 1500]), resources=np.array([50, 50]), value=75)
    ]
    
    # 计算总需求
    total_demand = np.array([500, 500])  # [150+120+100+80+50, 150+130+100+70+50]
    
    # 10个无人机，资源分配等于总需求
    uavs = [
        UAV(id=1, position=np.array([500, 500]), heading=np.pi/4, resources=np.array([80, 70]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([4500, 500]), heading=3*np.pi/4, resources=np.array([70, 80]), max_distance=6000, velocity_range=(60, 160), economic_speed=110),
        UAV(id=3, position=np.array([500, 3500]), heading=-np.pi/4, resources=np.array([60, 60]), max_distance=6000, velocity_range=(55, 155), economic_speed=105),
        UAV(id=4, position=np.array([4500, 3500]), heading=-3*np.pi/4, resources=np.array([50, 50]), max_distance=6000, velocity_range=(65, 165), economic_speed=115),
        UAV(id=5, position=np.array([2500, 500]), heading=np.pi/2, resources=np.array([40, 40]), max_distance=6000, velocity_range=(70, 170), economic_speed=120),
        UAV(id=6, position=np.array([500, 2000]), heading=0, resources=np.array([50, 50]), max_distance=6000, velocity_range=(45, 145), economic_speed=95),
        UAV(id=7, position=np.array([4500, 2000]), heading=np.pi, resources=np.array([40, 40]), max_distance=6000, velocity_range=(75, 175), economic_speed=125),
        UAV(id=8, position=np.array([1500, 500]), heading=np.pi/3, resources=np.array([30, 40]), max_distance=6000, velocity_range=(40, 140), economic_speed=90),
        UAV(id=9, position=np.array([3500, 500]), heading=2*np.pi/3, resources=np.array([40, 30]), max_distance=6000, velocity_range=(80, 180), economic_speed=130),
        UAV(id=10, position=np.array([2500, 3500]), heading=-np.pi/2, resources=np.array([40, 40]), max_distance=6000, velocity_range=(55, 155), economic_speed=105)
    ]
    
    # 设计合理的障碍物
    obstacles = [
        # 中央障碍区域
        CircularObstacle(center=(2500, 2000), radius=300, tolerance=obstacle_tolerance),
        
        # 四个角落的障碍物
        PolygonalObstacle(vertices=[(500, 500), (1000, 700), (700, 1000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(4000, 500), (4500, 700), (4500, 300)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(500, 3500), (700, 3000), (1000, 3500)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(4000, 3500), (4500, 3300), (4300, 3000)], tolerance=obstacle_tolerance),
        
        # 通道障碍物
        CircularObstacle(center=(1800, 1800), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3200, 2200), radius=200, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_simple_convergence_test_scenario(obstacle_tolerance=50.0):
    """
    提供一个简化的测试场景，用于算法收敛性测试。
    特点：
    - 2个UAV，1个Target
    - 无障碍物
    - 简单的资源分配
    - 快速收敛验证
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 创建2个UAV
    uavs = [
        UAV(
            id="UAV_1",
            position=np.array([0.0, 0.0, 10.0]),
            heading=0.0,
            resources=np.array([30.0, 30.0, 30.0]),
            max_distance=100.0,
            velocity_range=(0.0, 20.0),
            economic_speed=15.0
        ),
        UAV(
            id="UAV_2", 
            position=np.array([10.0, 10.0, 10.0]),
            heading=0.0,
            resources=np.array([30.0, 30.0, 30.0]),
            max_distance=100.0,
            velocity_range=(0.0, 20.0),
            economic_speed=15.0
        )
    ]
    
    # 创建1个Target
    targets = [
        Target(
            id="Target_1",
            position=np.array([50.0, 50.0, 0.0]),
            resources=np.array([50.0, 50.0, 50.0]),
            value=100.0
        )
    ]
    
    # 无障碍物，简化测试
    obstacles = []
    
    return uavs, targets, obstacles

def get_minimal_test_scenario(obstacle_tolerance=50.0):
    """
    提供最小化测试场景，用于快速验证算法基本功能。
    特点：
    - 1个UAV，1个Target
    - 无障碍物
    - 最简单的资源分配
    - 极快速收敛
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 创建1个UAV
    uavs = [
        UAV(
            id="UAV_1",
            position=np.array([0.0, 0.0, 10.0]),
            heading=0.0,
            resources=np.array([50.0, 50.0, 50.0]),
            max_distance=100.0,
            velocity_range=(0.0, 20.0),
            economic_speed=15.0
        )
    ]
    
    # 创建1个Target
    targets = [
        Target(
            id="Target_1",
            position=np.array([50.0, 50.0, 0.0]),
            resources=np.array([50.0, 50.0, 50.0]),
            value=100.0
        )
    ]
    
    # 无障碍物
    obstacles = []
    
    return uavs, targets, obstacles

def get_new_experimental_scenario(obstacle_tolerance):
    """
    提供一个根据用户指定信息新增的实验场景。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离 (此场景中未使用)。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 为未在输入中指定的无人机参数设置合理的默认值
    default_max_distance = 6000
    default_velocity_range = (50, 150)
    default_economic_speed = 100

    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([2, 1]), 
            max_distance=default_max_distance, velocity_range=default_velocity_range, economic_speed=default_economic_speed),
        UAV(id=2, position=np.array([1500, 0]), heading=np.pi / 6, resources=np.array([0, 1]), 
            max_distance=default_max_distance, velocity_range=default_velocity_range, economic_speed=default_economic_speed),
        UAV(id=3, position=np.array([3000, 0]), heading=3 * np.pi / 4, resources=np.array([3, 2]), 
            max_distance=default_max_distance, velocity_range=default_velocity_range, economic_speed=default_economic_speed),
        UAV(id=4, position=np.array([2000, 2000]), heading=np.pi / 6, resources=np.array([2, 1]), 
            max_distance=default_max_distance, velocity_range=default_velocity_range, economic_speed=default_economic_speed)
    ]
    
    targets = [
        Target(id=1, position=np.array([1500, 1500]), resources=np.array([4, 2]), value=100),
        Target(id=2, position=np.array([2000, 1000]), resources=np.array([3, 3]), value=90)
    ]

    # 此场景没有定义障碍物
    obstacles = []
    
    return uavs, targets, obstacles
def get_small_scenario(obstacle_tolerance):
    """
    提供一个预置的小规模测试场景。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # [修改] 增加无人机初始资源，使总供给大于总需求
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([140, 65]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([75, 156]), max_distance=6000, velocity_range=(60, 180), economic_speed=120),
        # UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([130, 185]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([85, 95]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([120, 119]), max_distance=6000, velocity_range=(50, 160), economic_speed=110)
    ]
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 205]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([160, 100]), value=80)
    ]
    obstacles = [
        CircularObstacle(center=(1200, 1200), radius=400, tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=obstacle_tolerance)
    ]
    return uavs, targets, obstacles


def get_complex_scenario(obstacle_tolerance):
    """
    提供一个随机生成的大规模复杂场景。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    COMPLEX_UAV_COUNT = 15
    COMPLEX_TARGET_COUNT = 15
    COMPLEX_OBSTACLE_COUNT = 8
    MAP_SIZE_COMPLEX = (8000, 6000)
    
    uavs = [
        UAV(id=i + 1,
            position=np.array([random.uniform(0, MAP_SIZE_COMPLEX[0]), random.uniform(0, MAP_SIZE_COMPLEX[1])]),
            heading=random.uniform(0, 2 * np.pi),
            resources=np.array([random.randint(80, 120), random.randint(80, 120)]),
            max_distance=15000,
            velocity_range=(50, 180),
            economic_speed=120) for i in range(COMPLEX_UAV_COUNT)
    ]
    
    targets = [
        Target(id=i + 1,
               position=np.array([random.uniform(MAP_SIZE_COMPLEX[0] * 0.1, MAP_SIZE_COMPLEX[0] * 0.9),
                                  random.uniform(MAP_SIZE_COMPLEX[1] * 0.1, MAP_SIZE_COMPLEX[1] * 0.9)]),
               resources=np.array([random.randint(100, 200), random.randint(100, 200)]),
               value=random.randint(80, 150)) for i in range(COMPLEX_TARGET_COUNT)
    ]
    
    obstacles = []
    for _ in range(COMPLEX_OBSTACLE_COUNT):
        center = (random.uniform(0, MAP_SIZE_COMPLEX[0]), random.uniform(0, MAP_SIZE_COMPLEX[1]))
        if random.random() > 0.5:
            obstacles.append(CircularObstacle(
                center=center,
                radius=random.uniform(MAP_SIZE_COMPLEX[0] * 0.05, MAP_SIZE_COMPLEX[0] * 0.1),
                tolerance=obstacle_tolerance))
        else:
            num_verts = random.randint(3, 6)
            radius = random.uniform(MAP_SIZE_COMPLEX[0] * 0.06, MAP_SIZE_COMPLEX[0] * 0.12)
            angles = np.sort(np.random.rand(num_verts) * 2 * np.pi)
            obstacles.append(PolygonalObstacle(
                vertices=[(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles],
                tolerance=obstacle_tolerance))
                
    return uavs, targets, obstacles

def get_complex_scenario_v2(obstacle_tolerance):
    """
    提供一个新的复杂场景：10个无人机，5个固定目标，资源需求从少于到等于到多余。
    保持障碍物与原场景一致以确保可比性。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 固定5个目标及其资源需求
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 120]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([90, 100]), value=80),
        Target(id=4, position=np.array([1500, 3000]), resources=np.array([80, 90]), value=85),
        Target(id=5, position=np.array([3500, 1500]), resources=np.array([70, 80]), value=75)
    ]
    
    # 计算总需求
    total_demand = np.array([500, 520])  # [150+110+90+80+70, 120+130+100+90+80]
    
    # 10个无人机，资源分配从少于到等于到多余需求
    uavs = [
        # 场景1：资源不足（总供给 < 总需求）
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([90, 65]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([70, 105]), max_distance=6000, velocity_range=(60, 180), economic_speed=120),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([110, 85]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([80, 95]), max_distance=6000, velocity_range=(50, 160), economic_speed=110),
        UAV(id=5, position=np.array([2500, 0]), heading=np.pi / 4, resources=np.array([60, 70]), max_distance=6000, velocity_range=(55, 170), economic_speed=125),
        UAV(id=6, position=np.array([0, 2000]), heading=np.pi / 3, resources=np.array([50, 60]), max_distance=6000, velocity_range=(65, 190), economic_speed=140),
        UAV(id=7, position=np.array([4000, 0]), heading=-np.pi / 3, resources=np.array([40, 50]), max_distance=6000, velocity_range=(45, 155), economic_speed=105),
        UAV(id=8, position=np.array([5000, 2000]), heading=np.pi / 2, resources=np.array([30, 40]), max_distance=6000, velocity_range=(75, 185), economic_speed=135),
        UAV(id=9, position=np.array([1000, 4000]), heading=-np.pi / 4, resources=np.array([20, 30]), max_distance=6000, velocity_range=(40, 145), economic_speed=95),
        UAV(id=10, position=np.array([4000, 4000]), heading=np.pi, resources=np.array([10, 20]), max_distance=6000, velocity_range=(80, 195), economic_speed=145)
    ]
    
    # 保持与原场景相同的障碍物
    obstacles = [
        CircularObstacle(center=(1200, 1200), radius=400, tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_complex_scenario_v3(obstacle_tolerance):
    """
    提供另一个复杂场景：10个无人机，5个固定目标，资源供给等于需求。
    保持障碍物与原场景一致以确保可比性。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 固定5个目标及其资源需求
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 120]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([90, 100]), value=80),
        Target(id=4, position=np.array([1500, 3000]), resources=np.array([80, 90]), value=85),
        Target(id=5, position=np.array([3500, 1500]), resources=np.array([70, 80]), value=75)
    ]
    
    # 计算总需求
    total_demand = np.array([500, 520])  # [150+110+90+80+70, 120+130+100+90+80]
    
    # 10个无人机，资源分配等于总需求
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([100, 80]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([80, 100]), max_distance=6000, velocity_range=(60, 180), economic_speed=120),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([90, 90]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([70, 80]), max_distance=6000, velocity_range=(50, 160), economic_speed=110),
        UAV(id=5, position=np.array([2500, 0]), heading=np.pi / 4, resources=np.array([60, 70]), max_distance=6000, velocity_range=(55, 170), economic_speed=125),
        UAV(id=6, position=np.array([0, 2000]), heading=np.pi / 3, resources=np.array([50, 60]), max_distance=6000, velocity_range=(65, 190), economic_speed=140),
        UAV(id=7, position=np.array([4000, 0]), heading=-np.pi / 3, resources=np.array([40, 50]), max_distance=6000, velocity_range=(45, 155), economic_speed=105),
        UAV(id=8, position=np.array([5000, 2000]), heading=np.pi / 2, resources=np.array([30, 40]), max_distance=6000, velocity_range=(75, 185), economic_speed=135),
        UAV(id=9, position=np.array([1000, 4000]), heading=-np.pi / 4, resources=np.array([20, 30]), max_distance=6000, velocity_range=(40, 145), economic_speed=95),
        UAV(id=10, position=np.array([4000, 4000]), heading=np.pi, resources=np.array([10, 20]), max_distance=6000, velocity_range=(80, 195), economic_speed=145)
    ]
    
    # 保持与原场景相同的障碍物
    obstacles = [
        CircularObstacle(center=(1200, 1200), radius=400, tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_complex_scenario_v4(obstacle_tolerance):
    """
    提供最复杂的场景：10个无人机，5个固定目标，资源供给超过需求20%。
    保持障碍物与原场景一致以确保可比性。
    
    Args:
        obstacle_tolerance (float): 障碍物的安全容忍距离。

    Returns:
        tuple: 包含无人机列表、目标列表和障碍物列表。
    """
    # 固定5个目标及其资源需求
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([150, 120]), value=100),
        Target(id=2, position=np.array([3000, 2500]), resources=np.array([110, 130]), value=90),
        Target(id=3, position=np.array([2500, 1000]), resources=np.array([90, 100]), value=80),
        Target(id=4, position=np.array([1500, 3000]), resources=np.array([80, 90]), value=85),
        Target(id=5, position=np.array([3500, 1500]), resources=np.array([70, 80]), value=75)
    ]
    
    # 计算总需求
    total_demand = np.array([500, 520])  # [150+110+90+80+70, 120+130+100+90+80]
    # 总供给 = 总需求 * 1.2 (超过20%)
    total_supply = total_demand * 1.2  # [600, 624]
    
    # 10个无人机，资源分配超过总需求20%
    uavs = [
        UAV(id=1, position=np.array([0, 0]), heading=np.pi / 6, resources=np.array([120, 100]), max_distance=6000, velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([5000, 0]), heading=np.pi, resources=np.array([100, 120]), max_distance=6000, velocity_range=(60, 180), economic_speed=120),
        UAV(id=3, position=np.array([0, 4000]), heading=-np.pi / 2, resources=np.array([110, 110]), max_distance=6000, velocity_range=(70, 200), economic_speed=150),
        UAV(id=4, position=np.array([5000, 4000]), heading=-np.pi / 2, resources=np.array([90, 100]), max_distance=6000, velocity_range=(50, 160), economic_speed=110),
        UAV(id=5, position=np.array([2500, 0]), heading=np.pi / 4, resources=np.array([80, 90]), max_distance=6000, velocity_range=(55, 170), economic_speed=125),
        UAV(id=6, position=np.array([0, 2000]), heading=np.pi / 3, resources=np.array([70, 80]), max_distance=6000, velocity_range=(65, 190), economic_speed=140),
        UAV(id=7, position=np.array([4000, 0]), heading=-np.pi / 3, resources=np.array([60, 70]), max_distance=6000, velocity_range=(45, 155), economic_speed=105),
        UAV(id=8, position=np.array([5000, 2000]), heading=np.pi / 2, resources=np.array([50, 60]), max_distance=6000, velocity_range=(75, 185), economic_speed=135),
        UAV(id=9, position=np.array([1000, 4000]), heading=-np.pi / 4, resources=np.array([40, 50]), max_distance=6000, velocity_range=(40, 145), economic_speed=95),
        UAV(id=10, position=np.array([4000, 4000]), heading=np.pi, resources=np.array([30, 40]), max_distance=6000, velocity_range=(80, 195), economic_speed=145)
    ]
    
    # 保持与原场景相同的障碍物
    obstacles = [
        CircularObstacle(center=(1200, 1200), radius=400, tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3500, 3500), (4500, 4000), (4500, 3000)], tolerance=obstacle_tolerance),
        PolygonalObstacle(vertices=[(3800, 500), (4800, 800), (4500, 1500), (3500, 1200)], tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles

def get_rl_advantage_scenario(obstacle_tolerance=50.0):
    """
    创建体现RL算法优势的复杂场景
    
    该场景设计特点：
    1. 动态资源需求：目标资源需求随时间变化
    2. 多约束优化：同时考虑距离、时间、资源匹配
    3. 不确定性：障碍物随机分布，路径规划复杂
    4. 协作要求：需要多无人机协同完成任务
    5. 实时适应：环境状态动态变化
    
    Args:
        obstacle_tolerance (float): 障碍物容差
        
    Returns:
        tuple: (uavs, targets, obstacles)
    """
    print("创建RL优势体现场景...")
    
    # 创建更多无人机和目标，增加问题复杂度
    num_uavs = 15
    num_targets = 25
    num_obstacles = 30
    
    # 初始化无人机
    uavs = []
    for i in range(num_uavs):
        # 分散的初始位置
        x = random.uniform(50, 450)
        y = random.uniform(50, 450)
        position = np.array([x, y])
        
        # 多样化的资源配置
        if i < 5:
            # 高容量无人机
            resources = np.array([random.uniform(800, 1200), random.uniform(600, 1000)])
        elif i < 10:
            # 中等容量无人机
            resources = np.array([random.uniform(500, 800), random.uniform(400, 600)])
        else:
            # 低容量无人机
            resources = np.array([random.uniform(200, 500), random.uniform(150, 400)])
        
        uav = UAV(
            id=i,
            position=position,
            resources=resources.copy(),
            velocity_range=(80, 120),
            max_distance=2000,
            heading=0.0,
            economic_speed=100.0
        )
        uavs.append(uav)
    
    # 创建目标，具有动态资源需求
    targets = []
    for i in range(num_targets):
        # 分散的目标位置
        x = random.uniform(100, 400)
        y = random.uniform(100, 400)
        position = np.array([x, y])
        
        # 动态资源需求：不同目标有不同优先级
        if i < 8:
            # 高优先级目标
            resources = np.array([random.uniform(600, 1000), random.uniform(400, 800)])
        elif i < 15:
            # 中等优先级目标
            resources = np.array([random.uniform(300, 600), random.uniform(200, 500)])
        else:
            # 低优先级目标
            resources = np.array([random.uniform(100, 300), random.uniform(80, 250)])
        
        target = Target(
            id=i,
            position=position,
            resources=resources.copy(),
            value=1.0
        )
        targets.append(target)
    
    # 创建复杂的障碍物分布
    obstacles = []
    for i in range(num_obstacles):
        # 创建不同类型的障碍物
        if i < 10:
            # 大型障碍物
            x = random.uniform(150, 350)
            y = random.uniform(150, 350)
            radius = random.uniform(30, 60)
        elif i < 20:
            # 中型障碍物
            x = random.uniform(100, 400)
            y = random.uniform(100, 400)
            radius = random.uniform(15, 30)
        else:
            # 小型障碍物
            x = random.uniform(50, 450)
            y = random.uniform(50, 450)
            radius = random.uniform(5, 15)
        
        # 确保障碍物不会完全阻塞路径
        position = np.array([x, y])
        
        # 创建障碍物类（如果不存在则使用简单的圆形障碍物）
        try:
            from entities import Obstacle
            obstacle = Obstacle(position, radius)
        except ImportError:
            # 如果Obstacle类不存在，创建一个简单的障碍物类
            class SimpleObstacle:
                def __init__(self, position, radius):
                    self.position = position
                    self.radius = radius
                
                def check_line_segment_collision(self, p1, p2):
                    # 简单的线段碰撞检测
                    return False  # 简化处理
            
            obstacle = SimpleObstacle(position, radius)
        
        # 检查是否与无人机或目标位置冲突
        min_distance_to_uavs = min(float(np.linalg.norm(position - uav.position)) for uav in uavs)
        min_distance_to_targets = min(float(np.linalg.norm(position - target.position)) for target in targets)
        
        if min_distance_to_uavs > radius + obstacle_tolerance and min_distance_to_targets > radius + obstacle_tolerance:
            obstacles.append(obstacle)
    
    print(f"RL优势场景创建完成:")
    print(f"  - 无人机数量: {len(uavs)}")
    print(f"  - 目标数量: {len(targets)}")
    print(f"  - 障碍物数量: {len(obstacles)}")
    print(f"  - 场景复杂度: 高 (动态约束 + 多目标优化 + 不确定性)")
    
    return uavs, targets, obstacles

def get_strategic_trap_scenario(obstacle_tolerance=50.0):
    """
    创建"战略价值陷阱"场景
    
    场景特点：
    1. 高价值陷阱目标：在地图偏远角落放置价值极高的目标，被密集障碍物包围
    2. 中价值集群目标：地图中心区域放置3-4个中等价值目标，距离较近
    3. 资源异构性：部分无人机携带更多A类资源，部分携带更多B类资源
    4. 中心目标集群对A、B类资源都有需求
    
    Args:
        obstacle_tolerance: 障碍物安全距离
        
    Returns:
        uavs: 无人机列表
        targets: 目标列表  
        obstacles: 障碍物列表
    """
    # 创建无人机 - 资源异构性设计
    uavs = [
        # A类资源丰富的无人机 (位置在中心区域)
        UAV(1, np.array([200, 200]), 0.0, np.array([150, 50]), 1000, (20, 40), 30),
        UAV(2, np.array([250, 200]), 0.0, np.array([140, 60]), 1000, (20, 40), 30),
        UAV(3, np.array([200, 250]), 0.0, np.array([160, 40]), 1000, (20, 40), 30),
        
        # B类资源丰富的无人机 (位置在边缘区域)
        UAV(4, np.array([100, 100]), 0.0, np.array([50, 150]), 1000, (20, 40), 30),
        UAV(5, np.array([150, 100]), 0.0, np.array([60, 140]), 1000, (20, 40), 30),
        UAV(6, np.array([100, 150]), 0.0, np.array([40, 160]), 1000, (20, 40), 30),
        
        # 平衡型无人机 (位置在中间区域)
        UAV(7, np.array([300, 300]), 0.0, np.array([100, 100]), 1000, (20, 40), 30),
        UAV(8, np.array([350, 300]), 0.0, np.array([90, 110]), 1000, (20, 40), 30),
    ]
    
    # 创建目标 - 战略价值陷阱设计
    targets = [
        # 高价值陷阱目标 (在偏远角落，被障碍物包围)
        Target(1, np.array([50, 50]), np.array([200, 200]), 200),  # 价值极高，但难以到达
        
        # 中价值集群目标 (在中心区域，易于协同)
        Target(2, np.array([220, 220]), np.array([80, 80]), 80),   # 中心集群1
        Target(3, np.array([240, 220]), np.array([70, 90]), 80),   # 中心集群2  
        Target(4, np.array([220, 240]), np.array([90, 70]), 80),   # 中心集群3
        Target(5, np.array([240, 240]), np.array([75, 85]), 80),   # 中心集群4
        
        # 边缘目标 (中等价值，需要特定资源)
        Target(6, np.array([400, 100]), np.array([60, 120]), 60),  # 需要更多B类资源
        Target(7, np.array([100, 400]), np.array([120, 60]), 60),  # 需要更多A类资源
    ]
    
    # 创建障碍物 - 形成战略陷阱（改进版）
    obstacles = [
        # 包围高价值陷阱目标的障碍物（减少密度，确保可达性）
        CircularObstacle(center=(40, 40), radius=8, tolerance=obstacle_tolerance),   # 陷阱目标周围
        CircularObstacle(center=(60, 40), radius=8, tolerance=obstacle_tolerance),
        CircularObstacle(center=(40, 60), radius=8, tolerance=obstacle_tolerance),
        CircularObstacle(center=(60, 60), radius=8, tolerance=obstacle_tolerance),
        
        # 中心区域的轻微障碍物 (不影响协同)
        CircularObstacle(center=(200, 200), radius=5, tolerance=obstacle_tolerance),
        CircularObstacle(center=(260, 200), radius=5, tolerance=obstacle_tolerance),
        CircularObstacle(center=(200, 260), radius=5, tolerance=obstacle_tolerance),
        CircularObstacle(center=(260, 260), radius=5, tolerance=obstacle_tolerance),
        
        # 边缘区域的障碍物（减少数量）
        CircularObstacle(center=(350, 100), radius=8, tolerance=obstacle_tolerance),
        CircularObstacle(center=(100, 350), radius=8, tolerance=obstacle_tolerance),
        
        # 路径上的障碍物（减少数量和大小）
        CircularObstacle(center=(150, 150), radius=12, tolerance=obstacle_tolerance),
        CircularObstacle(center=(300, 150), radius=12, tolerance=obstacle_tolerance),
        CircularObstacle(center=(150, 300), radius=12, tolerance=obstacle_tolerance),
        CircularObstacle(center=(300, 300), radius=12, tolerance=obstacle_tolerance),
    ]
    
    print("战略价值陷阱场景已创建:")
    print(f"  - 无人机数量: {len(uavs)} (A类资源丰富: 3架, B类资源丰富: 3架, 平衡型: 2架)")
    print(f"  - 目标数量: {len(targets)} (高价值陷阱: 1个, 中价值集群: 4个, 边缘目标: 2个)")
    print(f"  - 障碍物数量: {len(obstacles)} (密集包围陷阱目标)")
    print("  - 场景特点: 资源异构性 + 价值陷阱 + 协同挑战")
    
    return uavs, targets, obstacles

# =============================================================================
# 课程学习场景 - 从易到难的训练序列
# =============================================================================
def generate_curriculum_scenarios(config):
    """根据config中定义的统一模板，生成课程学习所需的固定场景集"""

    scenarios_by_difficulty = {}

    stages = {
        'easy': 10,
        'medium': 30,
        'hard': 50
    }

    for stage_name, num_scenarios in stages.items():
        scenarios_by_difficulty[stage_name] = []
        template = config.SCENARIO_TEMPLATES[stage_name]

        for _ in range(num_scenarios):
            uav_num = np.random.randint(template['uav_num_range'][0], template['uav_num_range'][1] + 1)
            target_num = np.random.randint(template['target_num_range'][0], template['target_num_range'][1] + 1)
            obstacle_num = np.random.randint(template['obstacle_num_range'][0], template['obstacle_num_range'][1] + 1)
            resource_abundance = np.random.uniform(*template['resource_abundance_range'])

            scenario_instance = _generate_scenario(config, uav_num, target_num, obstacle_num, resource_abundance)
            scenarios_by_difficulty[stage_name].append(scenario_instance)

    return scenarios_by_difficulty

    
def _generate_scenario(config, uav_num, target_num, obstacle_num, resource_abundance):
    """
    根据指定的参数随机生成一个场景实例。
    这是生成课程学习中每个固定场景的核心辅助函数。
    优化：确保障碍物与目标、无人机不重叠，障碍物大小合理。
    """
    uavs = []
    targets = []
    obstacles = []

    map_size = config.MAP_SIZE
    num_resource_types = config.RESOURCE_DIM
    
    # 定义安全距离
    min_distance_between_entities = 200.0  # 实体间最小距离
    min_distance_to_obstacles = 100.0      # 到障碍物的最小距离
    max_obstacle_radius = 120.0            # 障碍物最大半径
    min_obstacle_radius = 30.0             # 障碍物最小半径

    # 1. 随机生成目标，确保不重叠
    total_demand = np.zeros(num_resource_types)
    target_positions = []
    
    for i in range(target_num):
        target_id = i + 1
        max_attempts = 100
        
        for attempt in range(max_attempts):
            position = np.random.rand(2) * map_size
            
            # 检查与已有目标的距离
            valid_position = True
            for existing_pos in target_positions:
                distance = np.linalg.norm(position - existing_pos)
                if distance < min_distance_between_entities:
                    valid_position = False
                    break
            
            if valid_position:
                target_positions.append(position)
                # 为每种资源类型生成一个随机需求
                resources = np.random.randint(50, 151, size=num_resource_types)
                value = np.random.randint(80, 121)
                targets.append(Target(id=target_id, position=position, resources=resources, value=value))
                total_demand += resources
                break
        else:
            # 如果无法找到合适位置，使用随机位置
            position = np.random.rand(2) * map_size
            resources = np.random.randint(50, 151, size=num_resource_types)
            value = np.random.randint(80, 121)
            targets.append(Target(id=target_id, position=position, resources=resources, value=value))
            total_demand += resources

    # 2. 【修复】根据总需求和资源富裕度，生成合理的UAV资源
    # 先计算期望的总供给（浮点数）
    expected_total_supply = total_demand * resource_abundance
    # 转换为整数供给
    total_supply_int = np.round(expected_total_supply).astype(int)
    
    # 【修复】确保最小资源量，防止UAV资源为0
    min_resource_per_uav = 10  # 每个UAV每种资源至少10单位
    min_total_supply = uav_num * min_resource_per_uav
    
    # 如果期望供给太小，调整到合理范围
    for r_type in range(num_resource_types):
        if total_supply_int[r_type] < min_total_supply:
            print(f"[WARNING] 资源类型{r_type}供给过少({total_supply_int[r_type]})，调整到最小值({min_total_supply})")
            total_supply_int[r_type] = min_total_supply
    
    # 【修复】智能分配整数资源给每个UAV，确保不为0
    uav_resources = np.zeros((uav_num, num_resource_types), dtype=int)
    if uav_num > 0:
        for r_type in range(num_resource_types):
            remaining_supply = total_supply_int[r_type]
            
            # 【修复】首先为每个UAV分配最小资源
            for uav_idx in range(uav_num):
                uav_resources[uav_idx, r_type] = min_resource_per_uav
                remaining_supply -= min_resource_per_uav
            
            # 随机分配剩余资源
            while remaining_supply > 0:
                uav_idx = np.random.randint(0, uav_num)
                allocation = min(remaining_supply, np.random.randint(1, min(20, remaining_supply + 1)))
                uav_resources[uav_idx, r_type] += allocation
                remaining_supply -= allocation
        
        # 【验证】确保没有UAV的资源为0
        for uav_idx in range(uav_num):
            for r_type in range(num_resource_types):
                if uav_resources[uav_idx, r_type] == 0:
                    uav_resources[uav_idx, r_type] = min_resource_per_uav
                    print(f"[FIX] UAV {uav_idx+1} 资源类型 {r_type} 从0调整为 {min_resource_per_uav}")
    
    # 【调试】输出资源分配信息
    actual_total_supply = np.sum(uav_resources, axis=0)
    actual_abundance = actual_total_supply / (total_demand + 1e-6)
    print(f"[DEBUG] 场景生成资源分配:")
    print(f"  目标总需求: {total_demand}")
    print(f"  期望总供给: {expected_total_supply}")
    print(f"  实际总供给: {actual_total_supply}")
    print(f"  期望充裕度: {resource_abundance}")
    print(f"  实际充裕度: {actual_abundance}")

    uav_positions = []
    for i in range(uav_num):
        uav_id = i + 1
        max_attempts = 100
        
        for attempt in range(max_attempts):
            position = np.random.rand(2) * map_size
            
            # 检查与已有无人机和目标的距离
            valid_position = True
            
            # 检查与目标的距离
            for target_pos in target_positions:
                distance = np.linalg.norm(position - target_pos)
                if distance < min_distance_between_entities:
                    valid_position = False
                    break
            
            # 检查与已有无人机的距离
            if valid_position:
                for existing_pos in uav_positions:
                    distance = np.linalg.norm(position - existing_pos)
                    if distance < min_distance_between_entities:
                        valid_position = False
                        break
            
            if valid_position:
                uav_positions.append(position)
                heading = np.random.uniform(0, 2 * np.pi)
                max_distance = config.UAV_MAX_DISTANCE
                velocity_range = config.UAV_VELOCITY_RANGE
                economic_speed = config.UAV_ECONOMIC_SPEED
                
                uavs.append(UAV(
                    id=uav_id,
                    position=position,
                    heading=heading,
                    resources=uav_resources[i],
                    max_distance=max_distance,
                    velocity_range=velocity_range,
                    economic_speed=economic_speed
                ))
                break
        else:
            # 如果无法找到合适位置，使用随机位置
            position = np.random.rand(2) * map_size
            heading = np.random.uniform(0, 2 * np.pi)
            max_distance = config.UAV_MAX_DISTANCE
            velocity_range = config.UAV_VELOCITY_RANGE
            economic_speed = config.UAV_ECONOMIC_SPEED
            
            uavs.append(UAV(
                id=uav_id,
                position=position,
                heading=heading,
                resources=uav_resources[i],
                max_distance=max_distance,
                velocity_range=velocity_range,
                economic_speed=economic_speed
            ))

    # 3. 随机生成障碍物，确保不与目标、无人机重叠，大小合理
    obstacle_positions = []
    for i in range(obstacle_num):
        max_attempts = 100
        
        for attempt in range(max_attempts):
            center = np.random.rand(2) * map_size
            radius = np.random.uniform(min_obstacle_radius, max_obstacle_radius)
            
            # 检查与目标的距离
            valid_position = True
            for target_pos in target_positions:
                distance = np.linalg.norm(center - target_pos)
                if distance < (radius + min_distance_to_obstacles):
                    valid_position = False
                    break
            
            # 检查与无人机的距离
            if valid_position:
                for uav_pos in uav_positions:
                    distance = np.linalg.norm(center - uav_pos)
                    if distance < (radius + min_distance_to_obstacles):
                        valid_position = False
                        break
            
            # 检查与已有障碍物的距离
            if valid_position:
                for existing_center, existing_radius in obstacle_positions:
                    distance = np.linalg.norm(center - existing_center)
                    if distance < (radius + existing_radius + min_distance_to_obstacles):
                        valid_position = False
                        break
            
            if valid_position:
                obstacle_positions.append((center, radius))
                tolerance = getattr(config, 'OBSTACLE_TOLERANCE', 50.0)
                obstacles.append(CircularObstacle(center=center, radius=radius, tolerance=tolerance))
                break
        else:
            # 如果无法找到合适位置，使用随机位置但减小半径
            center = np.random.rand(2) * map_size
            radius = min_obstacle_radius  # 使用最小半径
            tolerance = getattr(config, 'OBSTACLE_TOLERANCE', 50.0)
            obstacles.append(CircularObstacle(center=center, radius=radius, tolerance=tolerance))

    # 将生成的实体打包成一个标准的字典格式
    return _create_scenario_dict(uavs, targets, obstacles, resource_abundance)

def _create_scenario_dict(uavs, targets, obstacles, resource_abundance_rate):
    """将实体列表打包成字典，以便于在环境中加载。"""
    return {
        'uavs': uavs,
        'targets': targets,
        'obstacles': obstacles,
        'resource_abundance_rate': resource_abundance_rate
    }



def _generate_curriculum_scenario(num_uavs, num_targets, resource_ratio, difficulty, obstacle_tolerance):
    """
    生成单个课程学习场景
    
    Args:
        num_uavs (int): UAV数量
        num_targets (int): 目标数量
        resource_ratio (float): 资源充足率 (>1.0表示资源充足)
        difficulty (str): 难度等级 ('easy', 'medium', 'hard')
        obstacle_tolerance (float): 障碍物容差
        
    Returns:
        tuple: (uavs, targets, obstacles)
    """
    # 地图大小根据难度调整
    if difficulty == 'easy':
        map_size = (3000, 3000)
        obstacle_count = np.random.randint(0, 3)
    elif difficulty == 'medium':
        map_size = (5000, 5000)
        obstacle_count = np.random.randint(3, 8)
    else:  # hard
        map_size = (8000, 8000)
        obstacle_count = np.random.randint(8, 15)
    
    # 生成目标
    targets = []
    total_demand = np.array([0.0, 0.0])
    
    for i in range(num_targets):
        # 随机位置（避免边界）
        position = np.array([
            np.random.uniform(map_size[0] * 0.2, map_size[0] * 0.8),
            np.random.uniform(map_size[1] * 0.2, map_size[1] * 0.8)
        ])
        
        # 资源需求根据难度调整，使用整数
        if difficulty == 'easy':
            resources = np.random.randint(50, 101, 2)  # 50-100的整数
        elif difficulty == 'medium':
            resources = np.random.randint(80, 151, 2)  # 80-150的整数
        else:  # hard
            resources = np.random.randint(120, 201, 2)  # 120-200的整数
        
        value = np.random.uniform(80, 120)
        
        target = Target(
            id=i,
            position=position,
            resources=resources,
            value=value
        )
        targets.append(target)
        total_demand += resources
    
    # 生成UAV，确保总供给满足资源充足率，使用整数资源
    uavs = []
    # 计算期望的总供给（浮点数）
    expected_total_supply = total_demand * resource_ratio
    # 转换为整数供给
    total_supply_int = np.round(expected_total_supply).astype(int)
    
    # 将总供给分配给各个UAV，使用整数分配
    supply_per_uav_float = total_supply_int / num_uavs
    
    for i in range(num_uavs):
        # 随机位置（通常在地图边缘）
        if np.random.random() < 0.5:
            # 水平边缘
            x = np.random.uniform(0, map_size[0])
            y = np.random.choice([
                np.random.uniform(0, map_size[1] * 0.1),
                np.random.uniform(map_size[1] * 0.9, map_size[1])
            ])
        else:
            # 垂直边缘
            x = np.random.choice([
                np.random.uniform(0, map_size[0] * 0.1),
                np.random.uniform(map_size[0] * 0.9, map_size[0])
            ])
            y = np.random.uniform(0, map_size[1])
        
        position = np.array([x, y])
        
        # 资源分配（在平均值基础上加入随机性），使用整数
        resources_float = supply_per_uav_float * np.random.uniform(0.7, 1.3, 2)
        resources = np.round(np.maximum(resources_float, 10.0)).astype(int)  # 确保最小资源且为整数
        
        # 确保资源不为全零
        if np.all(resources == 0):
            resources = np.ones(2, dtype=int)
        
        # 其他参数
        heading = np.random.uniform(0, 2 * np.pi)
        max_distance = np.random.uniform(4000, 8000)
        velocity_range = (
            np.random.uniform(50, 80),
            np.random.uniform(120, 180)
        )
        economic_speed = np.random.uniform(90, 130)
        
        uav = UAV(
            id=i,
            position=position,
            heading=heading,
            resources=resources,
            max_distance=max_distance,
            velocity_range=velocity_range,
            economic_speed=economic_speed
        )
        uavs.append(uav)
    
    # 生成障碍物
    obstacles = []
    for _ in range(obstacle_count):
        # 随机位置（避免与UAV和目标重叠）
        attempts = 0
        while attempts < 10:
            center = (
                np.random.uniform(map_size[0] * 0.1, map_size[0] * 0.9),
                np.random.uniform(map_size[1] * 0.1, map_size[1] * 0.9)
            )
            
            # 检查与UAV和目标的距离
            min_distance_to_entities = float('inf')
            for uav in uavs:
                dist = np.linalg.norm(np.array(center) - uav.position)
                min_distance_to_entities = min(min_distance_to_entities, dist)
            
            for target in targets:
                dist = np.linalg.norm(np.array(center) - target.position)
                min_distance_to_entities = min(min_distance_to_entities, dist)
            
            # 如果距离足够远，创建障碍物
            if min_distance_to_entities > 200:
                if np.random.random() < 0.7:
                    # 圆形障碍物
                    radius = np.random.uniform(50, 150)
                    obstacle = CircularObstacle(center, radius, obstacle_tolerance)
                else:
                    # 多边形障碍物
                    num_vertices = np.random.randint(3, 6)
                    radius = np.random.uniform(60, 120)
                    angles = np.sort(np.random.uniform(0, 2*np.pi, num_vertices))
                    vertices = [
                        (center[0] + radius * np.cos(angle),
                         center[1] + radius * np.sin(angle))
                        for angle in angles
                    ]
                    obstacle = PolygonalObstacle(vertices, obstacle_tolerance)
                
                obstacles.append(obstacle)
                break
            
            attempts += 1
    
    return uavs, targets, obstacles

def get_curriculum_level1_scenario(obstacle_tolerance=50.0):
    """
    课程学习第1级：最简单场景
    - 2个UAV，1个目标
    - 资源充足（供给 > 需求 50%）
    - 无障碍物，距离适中
    """
    targets = [
        Target(id=1, position=np.array([2000, 2000]), resources=np.array([30, 30]), value=100)
    ]
    
    uavs = [
        UAV(id=1, position=np.array([1500, 1500]), heading=np.pi/4, 
            resources=np.array([25, 25]), max_distance=5000, 
            velocity_range=(80, 120), economic_speed=100),
        UAV(id=2, position=np.array([2500, 1500]), heading=3*np.pi/4, 
            resources=np.array([25, 25]), max_distance=5000, 
            velocity_range=(80, 120), economic_speed=100)
    ]
    
    obstacles = []  # 无障碍物
    
    return uavs, targets, obstacles


def get_curriculum_level2_scenario(obstacle_tolerance=50.0):
    """
    课程学习第2级：简单场景
    - 3个UAV，2个目标
    - 资源适中（供给 > 需求 20%）
    - 少量障碍物
    """
    targets = [
        Target(id=1, position=np.array([1500, 2500]), resources=np.array([40, 30]), value=100),
        Target(id=2, position=np.array([2500, 2500]), resources=np.array([30, 40]), value=90)
    ]
    
    uavs = [
        UAV(id=1, position=np.array([1000, 1000]), heading=np.pi/4, 
            resources=np.array([30, 25]), max_distance=5000, 
            velocity_range=(80, 120), economic_speed=100),
        UAV(id=2, position=np.array([2000, 1000]), heading=np.pi/2, 
            resources=np.array([25, 30]), max_distance=5000, 
            velocity_range=(80, 120), economic_speed=100),
        UAV(id=3, position=np.array([3000, 1000]), heading=3*np.pi/4, 
            resources=np.array([25, 25]), max_distance=5000, 
            velocity_range=(80, 120), economic_speed=100)
    ]
    
    obstacles = [
        CircularObstacle(center=(2000, 1750), radius=150, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_curriculum_level3_scenario(obstacle_tolerance=50.0):
    """
    课程学习第3级：中等场景
    - 4个UAV，3个目标
    - 资源刚好（供给 = 需求 110%）
    - 中等障碍物，需要协同
    """
    targets = [
        Target(id=1, position=np.array([1500, 2500]), resources=np.array([50, 40]), value=100),
        Target(id=2, position=np.array([2500, 2500]), resources=np.array([40, 50]), value=90),
        Target(id=3, position=np.array([2000, 1500]), resources=np.array([30, 30]), value=80)
    ]
    
    uavs = [
        UAV(id=1, position=np.array([500, 500]), heading=np.pi/4, 
            resources=np.array([35, 30]), max_distance=6000, 
            velocity_range=(70, 130), economic_speed=100),
        UAV(id=2, position=np.array([3500, 500]), heading=3*np.pi/4, 
            resources=np.array([30, 35]), max_distance=6000, 
            velocity_range=(70, 130), economic_speed=100),
        UAV(id=3, position=np.array([500, 3500]), heading=-np.pi/4, 
            resources=np.array([30, 30]), max_distance=6000, 
            velocity_range=(70, 130), economic_speed=100),
        UAV(id=4, position=np.array([3500, 3500]), heading=-3*np.pi/4, 
            resources=np.array([25, 25]), max_distance=6000, 
            velocity_range=(70, 130), economic_speed=100)
    ]
    
    obstacles = [
        CircularObstacle(center=(2000, 2000), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1200, 1200), radius=120, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_curriculum_level4_scenario(obstacle_tolerance=50.0):
    """
    课程学习第4级：困难场景
    - 4个UAV，3个目标
    - 资源紧张（供给 = 需求 100%）
    - 较多障碍物，需要协同
    """
    targets = [
        Target(id=1, position=np.array([1500, 2500]), resources=np.array([3, 2]), value=100),
        Target(id=2, position=np.array([2500, 2500]), resources=np.array([2, 3]), value=90),
        Target(id=3, position=np.array([2000, 1500]), resources=np.array([2, 2]), value=80)
    ]
    
    uavs = [
        UAV(id=1, position=np.array([500, 500]), heading=np.pi/4, 
            resources=np.array([2, 1]), max_distance=6000, 
            velocity_range=(60, 140), economic_speed=100),
        UAV(id=2, position=np.array([3500, 500]), heading=3*np.pi/4, 
            resources=np.array([2, 2]), max_distance=6000, 
            velocity_range=(60, 140), economic_speed=100),
        UAV(id=3, position=np.array([500, 3500]), heading=-np.pi/4, 
            resources=np.array([2, 2]), max_distance=6000, 
            velocity_range=(60, 140), economic_speed=100),
        UAV(id=4, position=np.array([3500, 3500]), heading=-3*np.pi/4, 
            resources=np.array([1, 2]), max_distance=6000, 
            velocity_range=(60, 140), economic_speed=100)
    ]
    
    obstacles = [
        CircularObstacle(center=(2000, 2000), radius=300, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1200, 1800), radius=180, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2800, 1800), radius=180, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_curriculum_level5_scenario(obstacle_tolerance=50.0):
    """
    课程学习第5级：最困难场景（接近experimental场景）
    - 4个UAV，3个目标
    - 资源极度紧张（供给 = 需求，零容错）
    - 复杂障碍物布局，必须协同
    """
    targets = [
        Target(id=1, position=np.array([1500, 2500]), resources=np.array([4, 3]), value=100),
        Target(id=2, position=np.array([2500, 2500]), resources=np.array([3, 4]), value=90),
        Target(id=3, position=np.array([2000, 1500]), resources=np.array([3, 3]), value=80)
    ]
    
    uavs = [
        UAV(id=1, position=np.array([500, 500]), heading=np.pi/4, 
            resources=np.array([3, 2]), max_distance=6000, 
            velocity_range=(50, 150), economic_speed=100),
        UAV(id=2, position=np.array([3500, 500]), heading=3*np.pi/4, 
            resources=np.array([2, 3]), max_distance=6000, 
            velocity_range=(50, 150), economic_speed=100),
        UAV(id=3, position=np.array([500, 3500]), heading=-np.pi/4, 
            resources=np.array([3, 3]), max_distance=6000, 
            velocity_range=(50, 150), economic_speed=100),
        UAV(id=4, position=np.array([3500, 3500]), heading=-3*np.pi/4, 
            resources=np.array([2, 2]), max_distance=6000, 
            velocity_range=(50, 150), economic_speed=100)
    ]
    
    obstacles = [
        CircularObstacle(center=(2000, 2000), radius=350, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1000, 2000), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3000, 2000), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2000, 1000), radius=150, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_curriculum_scenarios():
    """
    获取完整的课程学习场景序列 - 优化版本
    
    难度递进设计原则：
    1. 逐步增加UAV和目标数量
    2. 逐步降低资源充足度
    3. 逐步增加障碍物复杂度
    4. 逐步增加协同需求
    
    Returns:
        list: 包含所有课程学习场景的列表，每个元素为(场景函数, 场景名称, 难度描述)
    """
    return [
        (get_curriculum_level1_scenario, "Level1_Easy", "2UAV-1Target, 资源充足150%, 无障碍物"),
        (get_curriculum_level2_scenario, "Level2_Simple", "3UAV-2Target, 资源适中125%, 简单障碍物"),
        (get_curriculum_level3_scenario, "Level3_Medium", "4UAV-3Target, 资源紧张110%, 中等障碍物"),
        (get_curriculum_level4_scenario, "Level4_Hard", "5UAV-4Target, 资源刚好105%, 复杂障碍物"),
        (get_curriculum_level5_scenario, "Level5_Expert", "6UAV-5Target, 零容错100%, 迷宫障碍物")
    ]


# =============================================================================
# 大规模课程学习场景 - 20架无人机，15个目标
# =============================================================================

def get_large_curriculum_level1_scenario(obstacle_tolerance=50.0):
    """
    大规模课程学习第1级：简单场景 - 优化版本
    - 20个UAV，15个目标
    - 资源充足（供给 > 需求 140%）
    - 无障碍物，分布均匀
    """
    # 设置随机种子确保可重复性
    np.random.seed(42)
    
    # 15个目标，资源需求较小且固定
    targets = []
    target_positions = [
        [1000, 1000], [2000, 1000], [3000, 1000], [4000, 1000], [5000, 1000],
        [1000, 2500], [2000, 2500], [3000, 2500], [4000, 2500], [5000, 2500],
        [1000, 4000], [2000, 4000], [3000, 4000], [4000, 4000], [5000, 4000]
    ]
    
    # 固定资源需求，总需求 = 60 (30+30)
    target_resources = [
        [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],  # 前5个目标
        [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],  # 中5个目标
        [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]   # 后5个目标
    ]
    
    for i, (pos, res) in enumerate(zip(target_positions, target_resources)):
        targets.append(Target(
            id=i+1, 
            position=np.array(pos), 
            resources=np.array(res), 
            value=100 - i*2
        ))
    
    # 20个UAV，资源充足，总供给 = 84 (42+42)，供需比 = 1.4
    uavs = []
    uav_positions = [
        [500, 500], [1500, 500], [2500, 500], [3500, 500], [4500, 500], [5500, 500],
        [500, 1500], [1500, 1500], [2500, 1500], [3500, 1500], [4500, 1500], [5500, 1500],
        [500, 3000], [1500, 3000], [2500, 3000], [3500, 3000], [4500, 3000], [5500, 3000],
        [1000, 4500], [3000, 4500]
    ]
    
    # 固定资源分配，确保总供给 = 84
    uav_resources = [
        [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],  # 前5个UAV: 20
        [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],  # 中5个UAV: 20  
        [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],  # 后5个UAV: 20
        [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]   # 最后5个UAV: 20
    ]
    
    for i, (pos, res) in enumerate(zip(uav_positions, uav_resources)):
        uavs.append(UAV(
            id=i+1,
            position=np.array(pos),
            heading=np.random.uniform(0, 2*np.pi),
            resources=np.array(res),
            max_distance=8000,
            velocity_range=(60, 140),
            economic_speed=100
        ))
    
    obstacles = []  # 无障碍物
    
    return uavs, targets, obstacles


def get_large_curriculum_level2_scenario(obstacle_tolerance=50.0):
    """
    大规模课程学习第2级：中等场景
    - 20个UAV，15个目标
    - 资源适中（供给 > 需求 25%）
    - 少量障碍物
    """
    # 15个目标，资源需求增加
    targets = []
    target_positions = [
        [1200, 1200], [2400, 1200], [3600, 1200], [4800, 1200],
        [1200, 2400], [2400, 2400], [3600, 2400], [4800, 2400],
        [1200, 3600], [2400, 3600], [3600, 3600], [4800, 3600],
        [1800, 4800], [3000, 4800], [4200, 4800]
    ]
    
    for i, pos in enumerate(target_positions):
        # 资源需求在2-4之间，总需求约67
        resources = np.array([
            np.random.randint(2, 5),  # 资源类型1: 2-4
            np.random.randint(2, 5)   # 资源类型2: 2-4
        ])
        targets.append(Target(
            id=i+1, 
            position=np.array(pos), 
            resources=resources, 
            value=120 - i*3
        ))
    
    # 20个UAV，资源适中，总供给约84（25%余量）
    uavs = []
    uav_positions = [
        [600, 600], [1800, 600], [3000, 600], [4200, 600], [5400, 600],
        [600, 1800], [1800, 1800], [3000, 1800], [4200, 1800], [5400, 1800],
        [600, 3000], [1800, 3000], [3000, 3000], [4200, 3000], [5400, 3000],
        [600, 4200], [1800, 4200], [3000, 4200], [4200, 4200], [5400, 4200]
    ]
    
    for i, pos in enumerate(uav_positions):
        # 每个UAV携带2-5单位资源
        resources = np.array([
            np.random.randint(2, 4),  # 资源类型1: 2-3
            np.random.randint(2, 4)   # 资源类型2: 2-3
        ])
        uavs.append(UAV(
            id=i+1,
            position=np.array(pos),
            heading=np.random.uniform(0, 2*np.pi),
            resources=resources,
            max_distance=8000,
            velocity_range=(60, 140),
            economic_speed=100
        ))
    
    # 少量障碍物
    obstacles = [
        CircularObstacle(center=(2500, 2500), radius=300, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1500, 3500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3500, 1500), radius=200, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_large_curriculum_level3_scenario(obstacle_tolerance=50.0):
    """
    大规模课程学习第3级：困难场景
    - 20个UAV，15个目标
    - 资源紧张（供给 > 需求 15%）
    - 中等障碍物，开始需要协同
    """
    # 15个目标，资源需求进一步增加
    targets = []
    target_clusters = [
        # 集群1：左上角
        [[1000, 1000], [1500, 1000], [1000, 1500], [1500, 1500]],
        # 集群2：右上角
        [[4000, 1000], [4500, 1000], [4000, 1500], [4500, 1500]],
        # 集群3：中央
        [[2500, 2500], [3000, 2500], [2500, 3000]],
        # 集群4：下方
        [[1500, 4000], [2500, 4000], [3500, 4000], [4500, 4000]]
    ]
    
    target_id = 1
    for cluster in target_clusters:
        for pos in cluster:
            # 资源需求在3-6之间，总需求约135
            resources = np.array([
                np.random.randint(3, 7),  # 资源类型1: 3-6
                np.random.randint(3, 7)   # 资源类型2: 3-6
            ])
            targets.append(Target(
                id=target_id, 
                position=np.array(pos), 
                resources=resources, 
                value=150 - target_id*4
            ))
            target_id += 1
            if target_id > 15:
                break
        if target_id > 15:
            break
    
    # 20个UAV，资源紧张，总供给约155（15%余量）
    uavs = []
    uav_formations = [
        # 编队1：左侧
        [[500, 500], [500, 1500], [500, 2500], [500, 3500], [500, 4500]],
        # 编队2：上方
        [[1500, 500], [2500, 500], [3500, 500], [4500, 500], [5500, 500]],
        # 编队3：右侧
        [[5500, 1500], [5500, 2500], [5500, 3500], [5500, 4500]],
        # 编队4：下方
        [[1500, 5000], [2500, 5000], [3500, 5000], [4500, 5000]],
        # 编队5：中央预备队
        [[2000, 2000], [3000, 2000]]
    ]
    
    uav_id = 1
    for formation in uav_formations:
        for pos in formation:
            # 每个UAV携带2-4单位资源（降低供给以保持递进）
            resources = np.array([
                np.random.randint(2, 4),  # 资源类型1: 2-3
                np.random.randint(2, 4)   # 资源类型2: 2-3
            ])
            uavs.append(UAV(
                id=uav_id,
                position=np.array(pos),
                heading=np.random.uniform(0, 2*np.pi),
                resources=resources,
                max_distance=8000,
                velocity_range=(50, 150),
                economic_speed=100
            ))
            uav_id += 1
            if uav_id > 20:
                break
        if uav_id > 20:
            break
    
    # 中等障碍物
    obstacles = [
        CircularObstacle(center=(2750, 2750), radius=400, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1250, 2750), radius=250, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4250, 2750), radius=250, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2750, 1250), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2750, 4250), radius=200, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_large_curriculum_level4_scenario(obstacle_tolerance=50.0):
    """
    大规模课程学习第4级：很困难场景
    - 20个UAV，15个目标
    - 资源刚好（供给 = 需求 105%）
    - 较多障碍物，必须协同
    """
    # 15个目标，高资源需求，需要协同
    targets = []
    target_positions = [
        [1200, 1200], [2000, 1200], [2800, 1200], [3600, 1200], [4400, 1200],
        [1200, 2400], [2000, 2400], [2800, 2400], [3600, 2400], [4400, 2400],
        [1200, 3600], [2000, 3600], [2800, 3600], [3600, 3600], [4400, 3600]
    ]
    
    for i, pos in enumerate(target_positions):
        # 高资源需求，单个UAV无法满足，总需求约180
        resources = np.array([
            np.random.randint(5, 9),   # 资源类型1: 5-8
            np.random.randint(5, 9)    # 资源类型2: 5-8
        ])
        targets.append(Target(
            id=i+1, 
            position=np.array(pos), 
            resources=resources, 
            value=200 - i*5
        ))
    
    # 20个UAV，资源刚好，总供给约189（5%余量）
    uavs = []
    uav_positions = [
        [600, 600], [1400, 600], [2200, 600], [3000, 600], [3800, 600], [4600, 600],
        [600, 1400], [1400, 1400], [2200, 1400], [3000, 1400], [3800, 1400], [4600, 1400],
        [600, 2800], [1400, 2800], [2200, 2800], [3000, 2800], [3800, 2800], [4600, 2800],
        [1800, 4200], [3400, 4200]
    ]
    
    for i, pos in enumerate(uav_positions):
        # 每个UAV携带4-6单位资源
        resources = np.array([
            np.random.randint(4, 7),  # 资源类型1: 4-6
            np.random.randint(4, 7)   # 资源类型2: 4-6
        ])
        uavs.append(UAV(
            id=i+1,
            position=np.array(pos),
            heading=np.random.uniform(0, 2*np.pi),
            resources=resources,
            max_distance=8000,
            velocity_range=(40, 160),
            economic_speed=100
        ))
    
    # 较多障碍物，形成复杂地形
    obstacles = [
        # 中央大障碍物
        CircularObstacle(center=(2800, 2400), radius=500, tolerance=obstacle_tolerance),
        # 四角障碍物
        CircularObstacle(center=(1000, 1000), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4600, 1000), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1000, 3800), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4600, 3800), radius=200, tolerance=obstacle_tolerance),
        # 通道障碍物
        CircularObstacle(center=(2800, 1000), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2800, 3800), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1000, 2400), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4600, 2400), radius=150, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_large_curriculum_level5_scenario(obstacle_tolerance=50.0):
    """
    大规模课程学习第5级：极困难场景
    - 20个UAV，15个目标
    - 资源极度紧张（供给 = 需求，零容错）
    - 复杂障碍物布局，高度协同需求
    """
    # 15个目标，极高资源需求
    targets = []
    # 目标分布在战略要点
    strategic_positions = [
        [1500, 1500], [3000, 1500], [4500, 1500],  # 北线
        [1500, 2500], [3000, 2500], [4500, 2500],  # 中线
        [1500, 3500], [3000, 3500], [4500, 3500],  # 南线
        [750, 2000], [2250, 2000], [3750, 2000],   # 西-中-东要点
        [2250, 1000], [2250, 3000], [3750, 3000]   # 关键节点
    ]
    
    for i, pos in enumerate(strategic_positions):
        # 极高资源需求，必须多UAV协同，总需求约270
        resources = np.array([
            np.random.randint(8, 12),  # 资源类型1: 8-11
            np.random.randint(8, 12)   # 资源类型2: 8-11
        ])
        targets.append(Target(
            id=i+1, 
            position=np.array(pos), 
            resources=resources, 
            value=300 - i*8
        ))
    
    # 20个UAV，资源极度紧张，总供给约270（零容错）
    uavs = []
    # UAV部署在外围，需要突破障碍物
    perimeter_positions = [
        [500, 500], [1000, 500], [2000, 500], [3000, 500], [4000, 500], [5000, 500],
        [500, 1000], [5000, 1000], [500, 2000], [5000, 2000], [500, 3000], [5000, 3000],
        [500, 4000], [1000, 4000], [2000, 4000], [3000, 4000], [4000, 4000], [5000, 4000],
        [2500, 500], [2500, 4000]
    ]
    
    for i, pos in enumerate(perimeter_positions):
        # 每个UAV携带6-8单位资源
        resources = np.array([
            np.random.randint(6, 9),  # 资源类型1: 6-8
            np.random.randint(6, 9)   # 资源类型2: 6-8
        ])
        uavs.append(UAV(
            id=i+1,
            position=np.array(pos),
            heading=np.random.uniform(0, 2*np.pi),
            resources=resources,
            max_distance=8000,
            velocity_range=(30, 170),
            economic_speed=100
        ))
    
    # 复杂障碍物布局，形成迷宫式地形
    obstacles = [
        # 中央堡垒
        CircularObstacle(center=(2750, 2250), radius=600, tolerance=obstacle_tolerance),
        
        # 四个象限的主要障碍物
        CircularObstacle(center=(1500, 1500), radius=300, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4000, 1500), radius=300, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1500, 3000), radius=300, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4000, 3000), radius=300, tolerance=obstacle_tolerance),
        
        # 通道控制点
        CircularObstacle(center=(2750, 1000), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2750, 3500), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(1000, 2250), radius=200, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4500, 2250), radius=200, tolerance=obstacle_tolerance),
        
        # 次要阻塞点
        CircularObstacle(center=(1750, 2250), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(3750, 2250), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2750, 1750), radius=150, tolerance=obstacle_tolerance),
        CircularObstacle(center=(2750, 2750), radius=150, tolerance=obstacle_tolerance),
        
        # 边缘干扰障碍物
        CircularObstacle(center=(750, 750), radius=100, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4750, 750), radius=100, tolerance=obstacle_tolerance),
        CircularObstacle(center=(750, 3750), radius=100, tolerance=obstacle_tolerance),
        CircularObstacle(center=(4750, 3750), radius=100, tolerance=obstacle_tolerance)
    ]
    
    return uavs, targets, obstacles


def get_large_curriculum_scenarios():
    """
    获取大规模课程学习场景序列（20UAV-15Target）
    
    Returns:
        list: 包含所有大规模课程学习场景的列表
    """
    return [
        (get_large_curriculum_level1_scenario, "Large_Level1_Easy", "20UAV-15Target, 资源充足140%"),
        (get_large_curriculum_level2_scenario, "Large_Level2_Simple", "20UAV-15Target, 资源适中125%"),
        (get_large_curriculum_level3_scenario, "Large_Level3_Medium", "20UAV-15Target, 资源紧张115%"),
        (get_large_curriculum_level4_scenario, "Large_Level4_Hard", "20UAV-15Target, 资源刚好105%"),
        (get_large_curriculum_level5_scenario, "Large_Level5_Expert", "20UAV-15Target, 零容错100%")
    ]


def analyze_large_scenario_resources(scenario_func, scenario_name):
    """
    分析大规模场景的资源配置
    
    Args:
        scenario_func: 场景生成函数
        scenario_name: 场景名称
    """
    uavs, targets, obstacles = scenario_func(50.0)
    
    # 统计资源
    total_uav_resources = sum(np.sum(uav.resources) for uav in uavs)
    total_target_demands = sum(np.sum(target.resources) for target in targets)
    
    # 按资源类型统计
    uav_resources_by_type = np.sum([uav.resources for uav in uavs], axis=0)
    target_demands_by_type = np.sum([target.resources for target in targets], axis=0)
    
    # 协同需求分析
    max_single_uav = max(np.sum(uav.resources) for uav in uavs)
    max_target_demand = max(np.sum(target.resources) for target in targets)
    min_target_demand = min(np.sum(target.resources) for target in targets)
    
    print(f"\n{scenario_name} 资源分析:")
    print(f"  UAV数量: {len(uavs)}, 目标数量: {len(targets)}, 障碍物数量: {len(obstacles)}")
    print(f"  总供给: {total_uav_resources}, 总需求: {total_target_demands}")
    print(f"  资源比例: {total_uav_resources/total_target_demands:.3f}")
    print(f"  按类型 - 类型1: {uav_resources_by_type[0]}/{target_demands_by_type[0]} = {uav_resources_by_type[0]/target_demands_by_type[0]:.3f}")
    print(f"  按类型 - 类型2: {uav_resources_by_type[1]}/{target_demands_by_type[1]} = {uav_resources_by_type[1]/target_demands_by_type[1]:.3f}")
    print(f"  目标需求范围: {min_target_demand} - {max_target_demand}")
    print(f"  最大单UAV资源: {max_single_uav}")
    print(f"  需要协同: {'是' if max_target_demand > max_single_uav else '否'}")
    
    return {
        'uav_count': len(uavs),
        'target_count': len(targets),
        'obstacle_count': len(obstacles),
        'resource_ratio': total_uav_resources/total_target_demands,
        'cooperation_required': max_target_demand > max_single_uav,
        'max_target_demand': max_target_demand,
        'max_single_uav': max_single_uav
    }


# 测试大规模场景的函数
def test_large_curriculum_scenarios():
    """测试大规模课程学习场景"""
    print("=" * 80)
    print("大规模课程学习场景测试 (20UAV-15Target)")
    print("=" * 80)
    
    large_scenarios = get_large_curriculum_scenarios()
    scenario_stats = []
    
    for scenario_func, level_name, description in large_scenarios:
        stats = analyze_large_scenario_resources(scenario_func, f"{level_name}: {description}")
        stats['name'] = level_name
        stats['description'] = description
        scenario_stats.append(stats)
    
    # 总结分析
    print(f"\n{'='*80}")
    print("大规模课程难度递进分析")
    print("=" * 80)
    
    print(f"{'级别':<20} {'UAV':<5} {'目标':<5} {'障碍物':<8} {'资源比例':<10} {'需要协同':<8}")
    print("-" * 80)
    
    for stats in scenario_stats:
        cooperation = "是" if stats['cooperation_required'] else "否"
        print(f"{stats['name']:<20} {stats['uav_count']:<5} {stats['target_count']:<5} "
              f"{stats['obstacle_count']:<8} {stats['resource_ratio']:<10.3f} {cooperation:<8}")
    
    # 验证难度递进
    resource_ratios = [stats['resource_ratio'] for stats in scenario_stats]
    print(f"\n资源比例变化: {' -> '.join([f'{r:.3f}' for r in resource_ratios])}")
    
    # 检查递进趋势
    is_progressive = all(resource_ratios[i] >= resource_ratios[i+1] 
                        for i in range(len(resource_ratios)-1))
    
    if is_progressive:
        print("✅ 大规模课程设计合理，难度呈递进趋势")
    else:
        print("⚠️ 大规模课程设计需要调整")
    
    # 协同需求分析
    cooperation_levels = [stats['cooperation_required'] for stats in scenario_stats]
    if any(cooperation_levels):
        first_cooperation = next(i for i, coop in enumerate(cooperation_levels) if coop)
        print(f"协同需求从级别 {first_cooperation + 1} 开始")
    
    print(f"\n大规模场景特点:")
    print(f"  - 规模: 20架无人机，15个目标")
    print(f"  - 复杂度: 从简单分布到复杂迷宫地形")
    print(f"  - 协同需求: 从单机作战到必须多机协同")
    print(f"  - 资源管理: 从充足到零容错的精确分配")


# 更新原有的课程学习场景获取函数，支持选择规模
def get_curriculum_scenarios(large_scale=False):
    """
    获取课程学习场景序列
    
    Args:
        large_scale (bool): 是否使用大规模场景（20UAV-15Target）
    
    Returns:
        list: 包含课程学习场景的列表
    """
    if large_scale:
        return get_large_curriculum_scenarios()
    else:
        # 原有的小规模场景
        return [
            (get_curriculum_level1_scenario, "Level1_Easy", "2UAV-1Target, 资源充足150%"),
            (get_curriculum_level2_scenario, "Level2_Simple", "3UAV-2Target, 资源适中130%"),
            (get_curriculum_level3_scenario, "Level3_Medium", "4UAV-3Target, 资源刚好110%"),
            (get_curriculum_level4_scenario, "Level4_Hard", "4UAV-3Target, 资源紧张100%"),
            (get_curriculum_level5_scenario, "Level5_Expert", "4UAV-3Target, 零容错100%")
        ]