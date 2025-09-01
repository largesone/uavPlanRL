#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强训练配置
优化训练过程，提高模型在利用环境下的稳定性
"""

import numpy as np
from baseline_config import BaselineConfig

class EnhancedTrainingConfig(BaselineConfig):
    """增强训练配置 - 优化利用阶段表现"""
    
    # ============= 优化的探索策略参数 =============
    EPSILON_START = 0.9      # 初始探索率
    EPSILON_END = 0.01       # 更低的最终探索率（从0.1降到0.01）
    EPSILON_DECAY = 0.9995   # 更慢的衰减率，让模型有更多时间学习
    
    # ============= 分阶段训练参数 =============
    EXPLORATION_PHASE_RATIO = 0.3   # 探索阶段占比（前30%）
    EXPLOITATION_PHASE_RATIO = 0.7  # 利用阶段占比（后70%）
    
    # ============= 模型保存参数 =============
    SAVE_TOP_N_MODELS = 5           # 保存前N个最优模型
    MODEL_SAVE_INTERVAL = 50        # 模型保存间隔
    MIN_EPISODES_FOR_SAVE = 100     # 最少训练轮次才开始保存
    
    # ============= 推理集成参数 =============
    ENSEMBLE_SIZE = 5               # 集成模型数量
    SOFTMAX_TEMPERATURE = 1.0       # Softmax温度参数
    ENSEMBLE_VOTING_METHOD = 'weighted_softmax'  # 集成投票方法
    
    @classmethod
    def get_epsilon_schedule(cls, episode, total_episodes):
        """
        获取动态epsilon调度
        
        Args:
            episode: 当前轮次
            total_episodes: 总轮次
            
        Returns:
            float: 当前轮次的epsilon值
        """
        # 计算当前阶段
        exploration_episodes = int(total_episodes * cls.EXPLORATION_PHASE_RATIO)
        
        if episode < exploration_episodes:
            # 探索阶段：正常衰减
            progress = episode / exploration_episodes
            epsilon = cls.EPSILON_START * (cls.EPSILON_DECAY ** episode)
        else:
            # 利用阶段：快速衰减到更低值
            exploitation_episode = episode - exploration_episodes
            exploitation_total = total_episodes - exploration_episodes
            
            # 从当前epsilon快速衰减到EPSILON_END
            current_epsilon = cls.EPSILON_START * (cls.EPSILON_DECAY ** exploration_episodes)
            progress = exploitation_episode / exploitation_total
            
            # 指数衰减到最终值
            epsilon = current_epsilon * ((cls.EPSILON_END / current_epsilon) ** progress)
        
        return max(epsilon, cls.EPSILON_END)
    
    @classmethod
    def should_save_model(cls, episode, current_score, score_history):
        """
        判断是否应该保存当前模型
        
        Args:
            episode: 当前轮次
            current_score: 当前分数
            score_history: 历史分数列表
            
        Returns:
            bool: 是否保存模型
        """
        # 最少训练轮次检查
        if episode < cls.MIN_EPISODES_FOR_SAVE:
            return False
        
        # 间隔检查
        if episode % cls.MODEL_SAVE_INTERVAL != 0:
            return False
        
        # 分数检查：当前分数在历史前N名
        if len(score_history) < cls.SAVE_TOP_N_MODELS:
            return True
        
        # 检查是否比历史最差的保存模型更好
        sorted_scores = sorted(score_history, reverse=True)
        return current_score > sorted_scores[cls.SAVE_TOP_N_MODELS - 1]
    
    @classmethod
    def get_model_filename(cls, episode, score, rank=None):
        """
        生成模型文件名
        
        Args:
            episode: 轮次
            score: 分数
            rank: 排名（可选）
            
        Returns:
            str: 模型文件名
        """
        if rank is not None:
            return f"model_ep{episode:06d}_score{score:.1f}_rank{rank}.pth"
        else:
            return f"model_ep{episode:06d}_score{score:.1f}.pth"