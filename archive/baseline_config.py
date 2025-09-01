#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线配置 - 稳定训练版本
记录当前可稳定训练的参数配置
"""

class BaselineConfig:
    """基线配置类 - 已验证的稳定训练参数"""
    
    # ============= 基础训练参数 =============
    LEARNING_RATE = 1e-5  # 学习率
    BATCH_SIZE = 16       # 批次大小
    MEMORY_SIZE = 15000   # 经验回放缓冲区大小
    GAMMA = 0.99          # 折扣因子
    
    # ============= 探索策略参数 =============
    EPSILON_START = 0.9    # 初始探索率
    EPSILON_END = 0.1      # 最终探索率
    EPSILON_DECAY = 0.995  # 探索率衰减
    
    # ============= 网络更新参数 =============
    TARGET_UPDATE_FREQ = 20  # 目标网络更新频率
    GRADIENT_CLIP = 0.5      # 梯度裁剪
    WEIGHT_DECAY = 2e-5      # 权重衰减
    
    # ============= 奖励系统参数 =============
    ENABLE_PBRS = True       # 启用PBRS
    PBRS_TYPE = 'synergy'    # PBRS类型
    REWARD_NORMALIZATION = False  # 奖励归一化
    REWARD_SCALE = 1.0       # 奖励缩放
    REWARD_CLIP_MIN = -500.0 # 奖励裁剪下限
    REWARD_CLIP_MAX = 2000.0 # 奖励裁剪上限
    
    # ============= 训练控制参数 =============
    PATIENCE = 100           # 早停耐心值
    MAX_EPISODES = 1000      # 最大训练轮次
    
    # ============= 网络架构参数 =============
    NETWORK_TYPE = 'ZeroShotGNN'  # 网络类型
    HIDDEN_DIM = 128         # 隐藏层维度
    
    # ============= 分阶段训练参数 =============
    EXPLORATION_PHASE_RATIO = 1.0   # 基线配置不分阶段，全程探索
    EXPLOITATION_PHASE_RATIO = 0.0  # 基线配置不分阶段
    
    # ============= 模型保存参数 =============
    SAVE_TOP_N_MODELS = 1           # 基线只保存最终模型
    MODEL_SAVE_INTERVAL = 1000      # 基线不频繁保存
    MIN_EPISODES_FOR_SAVE = 1000    # 基线只在最后保存
    
    # ============= 调试参数 =============
    ENABLE_REWARD_DEBUG = False  # 奖励调试（训练时关闭以提高性能）
    
    @classmethod
    def get_baseline_config(cls):
        """获取基线配置字典"""
        return {
            'LEARNING_RATE': cls.LEARNING_RATE,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'MEMORY_SIZE': cls.MEMORY_SIZE,
            'GAMMA': cls.GAMMA,
            'EPSILON_START': cls.EPSILON_START,
            'EPSILON_END': cls.EPSILON_END,
            'EPSILON_DECAY': cls.EPSILON_DECAY,
            'TARGET_UPDATE_FREQ': cls.TARGET_UPDATE_FREQ,
            'GRADIENT_CLIP': cls.GRADIENT_CLIP,
            'WEIGHT_DECAY': cls.WEIGHT_DECAY,
            'ENABLE_PBRS': cls.ENABLE_PBRS,
            'PBRS_TYPE': cls.PBRS_TYPE,
            'REWARD_NORMALIZATION': cls.REWARD_NORMALIZATION,
            'REWARD_SCALE': cls.REWARD_SCALE,
            'REWARD_CLIP_MIN': cls.REWARD_CLIP_MIN,
            'REWARD_CLIP_MAX': cls.REWARD_CLIP_MAX,
            'PATIENCE': cls.PATIENCE,
            'MAX_EPISODES': cls.MAX_EPISODES,
            'NETWORK_TYPE': cls.NETWORK_TYPE,
            'HIDDEN_DIM': cls.HIDDEN_DIM,
            'ENABLE_REWARD_DEBUG': cls.ENABLE_REWARD_DEBUG
        }
    
    @classmethod
    def apply_to_config(cls, config):
        """将基线配置应用到现有配置对象"""
        baseline = cls.get_baseline_config()
        for key, value in baseline.items():
            setattr(config, key, value)
        return config