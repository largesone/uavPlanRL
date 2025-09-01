#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复奖励学习问题的解决方案
基于调试分析结果，提供具体的修复策略
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区 - 重点学习高奖励经验"""
    
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done):
        """添加经验，高奖励经验获得更高优先级"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        # 对于高奖励经验，给予额外的优先级加成
        if reward > 500:  # 高奖励阈值
            max_prio = max(max_prio, abs(reward) / 100.0)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """采样时偏向高优先级经验"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        """更新优先级"""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def beta_by_frame(self, frame_idx):
        """动态调整beta值"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

class RewardNormalizer:
    """奖励标准化器 - 解决奖励方差过大问题"""
    
    def __init__(self, clip_range=(-10, 10), momentum=0.99):
        self.clip_range = clip_range
        self.momentum = momentum
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
    
    def normalize(self, reward):
        """标准化奖励"""
        # 更新运行统计
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * delta ** 2
        
        # 标准化
        std = np.sqrt(self.running_var + 1e-8)
        normalized_reward = (reward - self.running_mean) / std
        
        # 裁剪到合理范围
        normalized_reward = np.clip(normalized_reward, self.clip_range[0], self.clip_range[1])
        
        return normalized_reward
    
    def denormalize(self, normalized_reward):
        """反标准化（用于显示原始奖励）"""
        std = np.sqrt(self.running_var + 1e-8)
        return normalized_reward * std + self.running_mean

class AdaptiveRewardShaping:
    """自适应奖励塑形 - 增强高价值状态的学习"""
    
    def __init__(self, success_bonus=1000, synergy_bonus=300, progress_weight=0.1):
        self.success_bonus = success_bonus
        self.synergy_bonus = synergy_bonus
        self.progress_weight = progress_weight
        self.episode_rewards = deque(maxlen=100)
        
    def shape_reward(self, base_reward, is_success=False, synergy_count=0, progress_score=0):
        """塑形奖励以增强学习效果"""
        shaped_reward = base_reward
        
        # 成功奖励加成
        if is_success:
            shaped_reward += self.success_bonus
            
        # 协同奖励加成
        if synergy_count > 0:
            shaped_reward += synergy_count * self.synergy_bonus
            
        # 进度奖励
        shaped_reward += progress_score * self.progress_weight
        
        # 记录奖励用于自适应调整
        self.episode_rewards.append(shaped_reward)
        
        return shaped_reward
    
    def adapt_bonuses(self):
        """根据学习进度自适应调整奖励参数"""
        if len(self.episode_rewards) < 50:
            return
            
        recent_rewards = list(self.episode_rewards)[-50:]
        success_rate = len([r for r in recent_rewards if r > 800]) / len(recent_rewards)
        
        # 如果成功率太低，增加奖励
        if success_rate < 0.1:
            self.success_bonus = min(self.success_bonus * 1.1, 2000)
            self.synergy_bonus = min(self.synergy_bonus * 1.05, 500)
        # 如果成功率太高，适当减少奖励
        elif success_rate > 0.8:
            self.success_bonus = max(self.success_bonus * 0.95, 500)
            self.synergy_bonus = max(self.synergy_bonus * 0.98, 200)

class EnhancedDQNLoss:
    """增强的DQN损失函数 - 重点学习高价值经验"""
    
    def __init__(self, high_reward_threshold=500, high_reward_weight=2.0):
        self.high_reward_threshold = high_reward_threshold
        self.high_reward_weight = high_reward_weight
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def compute_loss(self, q_values, target_q_values, rewards, is_weights=None):
        """计算加权损失"""
        td_errors = target_q_values - q_values
        losses = self.mse_loss(q_values, target_q_values)
        
        # 对高奖励经验给予更高权重
        reward_weights = torch.ones_like(rewards)
        high_reward_mask = rewards > self.high_reward_threshold
        reward_weights[high_reward_mask] = self.high_reward_weight
        
        # 应用奖励权重
        losses = losses * reward_weights
        
        # 应用重要性采样权重
        if is_weights is not None:
            losses = losses * torch.FloatTensor(is_weights)
        
        return losses.mean(), td_errors.abs()

class LearningRateScheduler:
    """学习率调度器 - 优化学习过程"""
    
    def __init__(self, initial_lr=1e-4, min_lr=1e-6, decay_steps=1000, warmup_steps=100):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
    def get_lr(self):
        """获取当前学习率"""
        self.step_count += 1
        
        # 预热阶段
        if self.step_count <= self.warmup_steps:
            return self.initial_lr * (self.step_count / self.warmup_steps)
        
        # 余弦衰减
        progress = min((self.step_count - self.warmup_steps) / self.decay_steps, 1.0)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

class HighRewardExperienceTracker:
    """高奖励经验追踪器"""
    
    def __init__(self, threshold=800):
        self.threshold = threshold
        self.high_reward_experiences = []
        self.total_experiences = 0
        
    def track_experience(self, reward, state, action, next_state):
        """追踪经验"""
        self.total_experiences += 1
        
        if reward > self.threshold:
            self.high_reward_experiences.append({
                'reward': reward,
                'state': state,
                'action': action,
                'next_state': next_state,
                'timestamp': self.total_experiences
            })
    
    def get_high_reward_ratio(self):
        """获取高奖励经验比例"""
        if self.total_experiences == 0:
            return 0.0
        return len(self.high_reward_experiences) / self.total_experiences
    
    def sample_high_reward_experiences(self, n=10):
        """采样高奖励经验用于额外训练"""
        if len(self.high_reward_experiences) < n:
            return self.high_reward_experiences
        return random.sample(self.high_reward_experiences, n)

def create_enhanced_training_config():
    """创建增强的训练配置"""
    config = {
        # 网络参数
        'learning_rate': 1e-4,
        'batch_size': 32,  # 增加批次大小
        'target_update_freq': 1000,
        'gradient_clip': 1.0,  # 梯度裁剪
        
        # 探索参数
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        
        # 经验回放参数
        'replay_buffer_size': 100000,
        'prioritized_replay': True,
        'alpha': 0.6,
        'beta_start': 0.4,
        
        # 奖励参数
        'reward_normalization': True,
        'reward_clipping': (-10, 10),
        'success_bonus': 1000,
        'synergy_bonus': 300,
        
        # 训练参数
        'high_reward_weight': 3.0,  # 高奖励经验权重
        'high_reward_threshold': 500,
        'additional_high_reward_training': True,  # 额外的高奖励训练
        
        # 学习率调度
        'lr_scheduling': True,
        'warmup_steps': 1000,
        'decay_steps': 10000,
    }
    
    return config

def integrate_fixes_example():
    """展示如何集成修复方案的示例代码"""
    
    example_code = '''
# 1. 初始化增强组件
config = create_enhanced_training_config()
replay_buffer = PrioritizedReplayBuffer(
    capacity=config['replay_buffer_size'],
    alpha=config['alpha'],
    beta_start=config['beta_start']
)
reward_normalizer = RewardNormalizer(clip_range=config['reward_clipping'])
reward_shaper = AdaptiveRewardShaping(
    success_bonus=config['success_bonus'],
    synergy_bonus=config['synergy_bonus']
)
enhanced_loss = EnhancedDQNLoss(
    high_reward_threshold=config['high_reward_threshold'],
    high_reward_weight=config['high_reward_weight']
)
lr_scheduler = LearningRateScheduler(initial_lr=config['learning_rate'])
experience_tracker = HighRewardExperienceTracker()

# 2. 在训练循环中使用
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    step_rewards = []
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # 奖励塑形
        is_success = info.get('mission_complete', False)
        synergy_count = info.get('synergy_attacks', 0)
        progress_score = info.get('progress_score', 0)
        
        shaped_reward = reward_shaper.shape_reward(
            reward, is_success, synergy_count, progress_score
        )
        
        # 奖励标准化
        if config['reward_normalization']:
            normalized_reward = reward_normalizer.normalize(shaped_reward)
        else:
            normalized_reward = shaped_reward
        
        # 存储经验
        replay_buffer.push(state, action, normalized_reward, next_state, done)
        experience_tracker.track_experience(shaped_reward, state, action, next_state)
        
        episode_reward += shaped_reward
        step_rewards.append(shaped_reward)
        state = next_state
    
    # 训练网络
    if len(replay_buffer.buffer) > config['batch_size']:
        # 常规训练
        experiences, indices, weights = replay_buffer.sample(config['batch_size'])
        loss, td_errors = train_step(experiences, weights, enhanced_loss)
        replay_buffer.update_priorities(indices, td_errors.numpy())
        
        # 额外的高奖励经验训练
        if config['additional_high_reward_training']:
            high_reward_exp = experience_tracker.sample_high_reward_experiences(10)
            if high_reward_exp:
                train_on_high_reward_experiences(high_reward_exp, enhanced_loss)
    
    # 更新学习率
    if config['lr_scheduling']:
        new_lr = lr_scheduler.get_lr()
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    # 自适应调整奖励参数
    if episode % 100 == 0:
        reward_shaper.adapt_bonuses()
        
        # 打印统计信息
        high_reward_ratio = experience_tracker.get_high_reward_ratio()
        print(f"Episode {episode}: High reward ratio = {high_reward_ratio:.3f}")
'''
    
    return example_code

if __name__ == "__main__":
    print("=" * 60)
    print("奖励学习问题修复方案")
    print("=" * 60)
    
    print("\n1. 问题分析:")
    print("   - 奖励方差过大 (std=357.84)")
    print("   - 1000分奖励获得率低 (10%)")
    print("   - Q值学习不稳定")
    
    print("\n2. 解决方案:")
    print("   ✓ 优先经验回放 - 重点学习高奖励经验")
    print("   ✓ 奖励标准化 - 解决方差过大问题")
    print("   ✓ 自适应奖励塑形 - 动态调整奖励参数")
    print("   ✓ 增强损失函数 - 高奖励经验加权")
    print("   ✓ 学习率调度 - 优化学习过程")
    print("   ✓ 高奖励经验追踪 - 额外训练机制")
    
    print("\n3. 配置建议:")
    config = create_enhanced_training_config()
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n4. 集成示例:")
    print("   请参考 integrate_fixes_example() 函数中的代码")
    
    print("\n5. 预期效果:")
    print("   - 1000分奖励获得率提升至 30%+")
    print("   - 奖励方差降低至合理范围")
    print("   - Q值学习更加稳定")
    print("   - 整体收敛速度提升")
    
    print("\n" + "=" * 60)
    
    # 保存集成示例代码
    with open("integration_example.py", "w", encoding="utf-8") as f:
        f.write("# 奖励学习修复方案集成示例\n")
        f.write("# 将以下代码集成到你的训练循环中\n\n")
        f.write(integrate_fixes_example())
    
    print("集成示例代码已保存至: integration_example.py")
