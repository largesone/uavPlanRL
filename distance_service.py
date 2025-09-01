# -*- coding: utf-8 -*-
# 文件名: distance_service.py
# 描述: 统一的距离计算服务，支持快速欧氏距离和高精度PH-RRT*路径计算的无缝切换

import numpy as np
from typing import Tuple, Optional, List, Union
from path_planning import PHCurveRRTPlanner
import time
import logging

class DistanceCalculationService:
    """
    统一的距离计算服务
    
    功能特性：
    - 支持快速欧氏距离计算
    - 支持高精度PH-RRT*路径计算
    - 根据配置文件自动切换计算模式
    - 提供缓存机制提高性能
    - 支持批量计算
    """
    
    def __init__(self, config, obstacles=None):
        """
        初始化距离计算服务
        
        Args:
            config: 配置对象，包含USE_PHRRT_DURING_TRAINING和USE_PHRRT_DURING_PLANNING参数
            obstacles: 障碍物列表，用于PH-RRT*计算
        """
        self.config = config
        self.obstacles = obstacles or []
        
        # 缓存机制
        self.distance_cache = {}
        self.path_cache = {}
        self.cache_enabled = getattr(config, 'ENABLE_DISTANCE_CACHE', True)
        
        # 性能统计
        self.stats = {
            'euclidean_calls': 0,
            'phrrt_calls': 0,
            'cache_hits': 0,
            'total_euclidean_time': 0.0,
            'total_phrrt_time': 0.0
        }
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
    def calculate_distance(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                          mode: str = 'auto', return_path: bool = False) -> Union[float, Tuple[float, np.ndarray]]:
        """
        计算两点间的距离
        
        Args:
            start_pos: 起始位置 [x, y]
            end_pos: 结束位置 [x, y]
            mode: 计算模式 ('auto', 'euclidean', 'phrrt', 'training', 'planning')
            return_path: 是否返回路径点
            
        Returns:
            float: 距离值
            或 Tuple[float, np.ndarray]: (距离值, 路径点数组) 当return_path=True时
        """
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        
        # 生成缓存键
        cache_key = self._generate_cache_key(start_pos, end_pos, mode, return_path)
        
        # 检查缓存
        if self.cache_enabled and cache_key in self.distance_cache:
            self.stats['cache_hits'] += 1
            cached_result = self.distance_cache[cache_key]
            return cached_result
        
        # 确定计算模式
        use_phrrt = self._should_use_phrrt(mode)
        
        if use_phrrt and self.obstacles:
            # 使用PH-RRT*计算
            distance, path = self._calculate_phrrt_distance(start_pos, end_pos)
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"使用高精度PH-RRT算法计算距离: {distance:.2f}")
        else:
            # 使用欧氏距离计算
            distance, path = self._calculate_euclidean_distance(start_pos, end_pos, return_path)
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"使用快速近似算法计算距离: {distance:.2f}")
        
        # 准备返回结果
        if return_path:
            result = (distance, path)
        else:
            result = distance
        
        # 缓存结果
        if self.cache_enabled:
            self.distance_cache[cache_key] = result
        
        return result
    
    def calculate_batch_distances(self, start_positions: List[np.ndarray], 
                                 end_positions: List[np.ndarray], 
                                 mode: str = 'auto') -> np.ndarray:
        """
        批量计算距离
        
        Args:
            start_positions: 起始位置列表
            end_positions: 结束位置列表
            mode: 计算模式
            
        Returns:
            np.ndarray: 距离矩阵
        """
        if len(start_positions) != len(end_positions):
            raise ValueError("起始位置和结束位置列表长度必须相同")
        
        distances = []
        for start_pos, end_pos in zip(start_positions, end_positions):
            distance = self.calculate_distance(start_pos, end_pos, mode)
            distances.append(distance)
        
        return np.array(distances)
    
    def calculate_distance_matrix(self, positions: List[np.ndarray], 
                                 mode: str = 'auto') -> np.ndarray:
        """
        计算位置列表的距离矩阵
        
        Args:
            positions: 位置列表
            mode: 计算模式
            
        Returns:
            np.ndarray: 距离矩阵 [n_positions, n_positions]
        """
        n = len(positions)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.calculate_distance(positions[i], positions[j], mode)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # 对称矩阵
        
        return distance_matrix
    
    def _should_use_phrrt(self, mode: str) -> bool:
        """
        根据模式和配置确定是否使用PH-RRT*
        
        Args:
            mode: 计算模式
            
        Returns:
            bool: 是否使用PH-RRT*
        """
        if mode == 'euclidean':
            return False
        elif mode == 'phrrt':
            return True
        elif mode == 'training':
            return getattr(self.config, 'USE_PHRRT_DURING_TRAINING', False)
        elif mode == 'planning':
            return getattr(self.config, 'USE_PHRRT_DURING_PLANNING', False)
        elif mode == 'auto':
            # 自动模式：根据上下文判断
            # 如果在训练过程中，使用训练配置
            # 如果在规划过程中，使用规划配置
            # 默认使用训练配置
            return getattr(self.config, 'USE_PHRRT_DURING_TRAINING', False)
        else:
            self.logger.warning(f"未知的计算模式: {mode}，使用欧氏距离")
            return False
    
    def _calculate_euclidean_distance(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                    return_path: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """
        计算欧氏距离
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            return_path: 是否返回路径
            
        Returns:
            Tuple[float, Optional[np.ndarray]]: (距离, 路径点)
        """
        start_time = time.time()
        
        distance = np.linalg.norm(end_pos - start_pos)
        
        if return_path:
            # 生成简单的直线路径
            path = np.array([start_pos, end_pos])
        else:
            path = None
        
        # 更新统计
        self.stats['euclidean_calls'] += 1
        self.stats['total_euclidean_time'] += time.time() - start_time
        
        return distance, path
    
    def _calculate_phrrt_distance(self, start_pos: np.ndarray, end_pos: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        使用PH-RRT*计算距离和路径
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            
        Returns:
            Tuple[float, np.ndarray]: (距离, 路径点)
        """
        start_time = time.time()
        
        try:
            # 创建PH-RRT*规划器
            planner = PHCurveRRTPlanner(
                start=start_pos,
                goal=end_pos,
                start_heading=0.0,  # 默认朝向
                goal_heading=0.0,   # 默认朝向
                obstacles=self.obstacles,
                config=self.config
            )
            
            # 执行路径规划
            result = planner.plan()
            
            if result is not None:
                path, distance = result
                path = np.array(path)
            else:
                # 规划失败，回退到欧氏距离
                self.logger.warning(f"PH-RRT*规划失败，从 {start_pos} 到 {end_pos}，回退到欧氏距离")
                distance, path = self._calculate_euclidean_distance(start_pos, end_pos, return_path=True)
            
        except Exception as e:
            # 异常处理，回退到欧氏距离
            self.logger.error(f"PH-RRT*计算异常: {e}，回退到欧氏距离")
            distance, path = self._calculate_euclidean_distance(start_pos, end_pos, return_path=True)
        
        # 更新统计
        self.stats['phrrt_calls'] += 1
        self.stats['total_phrrt_time'] += time.time() - start_time
        
        return distance, path
    
    def _generate_cache_key(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                           mode: str, return_path: bool) -> str:
        """
        生成缓存键
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            mode: 计算模式
            return_path: 是否返回路径
            
        Returns:
            str: 缓存键
        """
        # 将位置坐标四舍五入到小数点后2位，减少缓存键的数量
        start_rounded = np.round(start_pos, 2)
        end_rounded = np.round(end_pos, 2)
        
        return f"{start_rounded[0]},{start_rounded[1]}_{end_rounded[0]},{end_rounded[1]}_{mode}_{return_path}"
    
    def clear_cache(self):
        """清空缓存"""
        self.distance_cache.clear()
        self.path_cache.clear()
        self.logger.info("距离计算缓存已清空")
    
    def get_statistics(self) -> dict:
        """
        获取性能统计信息
        
        Returns:
            dict: 统计信息
        """
        total_calls = self.stats['euclidean_calls'] + self.stats['phrrt_calls']
        
        stats = self.stats.copy()
        stats.update({
            'total_calls': total_calls,
            'cache_hit_rate': self.stats['cache_hits'] / max(total_calls, 1),
            'avg_euclidean_time': self.stats['total_euclidean_time'] / max(self.stats['euclidean_calls'], 1),
            'avg_phrrt_time': self.stats['total_phrrt_time'] / max(self.stats['phrrt_calls'], 1),
            'cache_size': len(self.distance_cache)
        })
        
        return stats
    
    def print_statistics(self):
        """打印性能统计信息"""
        stats = self.get_statistics()
        
        print("=" * 50)
        print("距离计算服务统计信息")
        print("=" * 50)
        print(f"总调用次数: {stats['total_calls']}")
        print(f"欧氏距离调用: {stats['euclidean_calls']}")
        print(f"PH-RRT*调用: {stats['phrrt_calls']}")
        print(f"缓存命中次数: {stats['cache_hits']}")
        print(f"缓存命中率: {stats['cache_hit_rate']:.2%}")
        print(f"缓存大小: {stats['cache_size']}")
        print(f"平均欧氏距离计算时间: {stats['avg_euclidean_time']:.4f}秒")
        print(f"平均PH-RRT*计算时间: {stats['avg_phrrt_time']:.4f}秒")
        print(f"总欧氏距离计算时间: {stats['total_euclidean_time']:.2f}秒")
        print(f"总PH-RRT*计算时间: {stats['total_phrrt_time']:.2f}秒")
        print("=" * 50)

# 全局距离计算服务实例
_global_distance_service = None

def get_distance_service(config=None, obstacles=None):
    """
    获取全局距离计算服务实例
    
    Args:
        config: 配置对象（仅在首次调用时需要）
        obstacles: 障碍物列表（仅在首次调用时需要）
        
    Returns:
        DistanceCalculationService: 距离计算服务实例
    """
    global _global_distance_service
    
    if _global_distance_service is None:
        if config is None:
            raise ValueError("首次调用get_distance_service时必须提供config参数")
        _global_distance_service = DistanceCalculationService(config, obstacles)
    
    return _global_distance_service

def reset_distance_service():
    """重置全局距离计算服务"""
    global _global_distance_service
    _global_distance_service = None