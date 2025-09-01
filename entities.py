# -*- coding: utf-8 -*-
# 文件名: entities.py
# 描述: 定义项目中的核心实体，如无人机(UAV)和目标(Target)。

import numpy as np

class UAV:
    """数据类：定义无人机的所有属性和状态"""
    def __init__(self, id, position, heading, resources, max_distance, velocity_range, economic_speed):
        self.id = id
        self.position = np.array(position)
        self.heading = heading
        self.resources = np.array(resources)
        self.initial_resources = np.array(resources)  # 确保正确保存初始资源
        self.max_distance = max_distance
        self.velocity_range = velocity_range
        self.economic_speed = economic_speed
        self.task_sequence = []
        self.current_distance = 0
        self.current_position = np.array(position)
        self.previous_position = None  # 用于塑形奖励计算
        
        # [协同增效新增] 计算单次最大运载能力，用于判断协同必要性
        self.max_payload = np.sum(self.initial_resources)  # 单机最大运载能力

    def reset(self):
        """重置无人机的状态到初始值"""
        self.resources = self.initial_resources.copy()
        self.current_distance = 0
        self.current_position = self.position.copy()
        self.task_sequence = []
        self.previous_position = None  # 重置位置历史

class Target:
    """数据类：定义目标的所有属性和状态"""
    def __init__(self, id, position, resources, value):
        self.id = id
        self.position = np.array(position)
        self.resources = np.array(resources)
        self.value = value
        self.allocated_uavs = []
        self.remaining_resources = np.array(resources)

    def reset(self):
        """重置目标的状态到初始值"""
        self.allocated_uavs = []
        self.remaining_resources = self.resources.copy()