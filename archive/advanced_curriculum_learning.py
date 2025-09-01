#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版高级课程学习训练器
修复混合精度警告、优化控制台输出、改进早停机制等
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict

from main import GraphRLSolver, set_chinese_font
from config import Config
from scenarios import get_curriculum_scenarios
from environment import DirectedGraph
from comprehensive_debug_analysis import ComprehensiveDebugAnalyzer

class OptimizedAdvancedCurriculumTrainer:
    """优化版高级课程学习训练器"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.config.ENABLE_REWARD_DEBUG = False
        
        # 动态训练配置
        self.dynamic_config = {
            'min_episodes_per_level': 50,
            'max_episodes_per_level': 500,
            'success_window_size': 100,
            'convergence_window_size': 50,
            'early_promotion_threshold': 0.85,
            'stability_threshold': 0.05,
        }
        
        # 优化的早停配置
        self.early_stopping_config = {
            'monitor_interval': 5000,          # 每5000环境步监控一次
            'completion_window': 10,           # 完成率滑动窗口大小
            'patience_points': 20,             # 连续20个监控点
            'min_improvement': 0.001,          # 最小改进0.1%
            'base_patience': 50,
            'patience_multiplier': {
                'easy': 0.8, 'simple': 1.0, 'medium': 1.5, 'hard': 2.0, 'expert': 2.5
            }
        }
        
        # 奖励统计配置
        self.reward_tracking = {
            'collaboration_threshold': 0.3,    # 协同奖励占比阈值
            'reward_clip_range': (-100, 100),  # 奖励裁剪范围
            'enable_reward_analysis': True,    # 启用奖励分析
        }
        
        # 状态归一化配置
        self.normalization_config = {
            'position_scale': 1000.0,          # 位置归一化尺度
            'resource_scale': 100.0,           # 资源归一化尺度
            'distance_scale': 2000.0,          # 距离归一化尺度
            'enable_input_validation': True,   # 启用输入验证
        }
        
        # 训练统计
        self.training_stats = {
            'total_env_steps': 0,
            'monitor_points': [],
            'reward_distributions': [],
            'collaboration_ratios': [],
            'input_validation_errors': 0,
        }
        
        # 输出目录
        self.output_dir = f"output/optimized_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"优化版高级课程学习训练器初始化完成")
        print(f"输出目录: {self.output_dir}")
        print(f"早停监控: 每{self.early_stopping_config['monitor_interval']}步")
        print(f"奖励分析: {'启用' if self.reward_tracking['enable_reward_analysis'] else '禁用'}")
        print(f"输入验证: {'启用' if self.normalization_config['enable_input_validation'] else '禁用'}")
    
    def _fix_mixed_precision_warnings(self, solver):
        """修复1：修复混合精度训练的FutureWarning"""
        if hasattr(solver, 'use_mixed_precision') and solver.use_mixed_precision:
            # 检查PyTorch版本并使用正确的API
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                solver._autocast_context = lambda: torch.amp.autocast('cuda')
                print("✅ 使用新版torch.amp.autocast API")
            else:
                solver._autocast_context = lambda: torch.cuda.amp.autocast()
                print("⚠️ 使用旧版torch.cuda.amp.autocast API")
        else:
            solver._autocast_context = lambda: torch.no_grad()
    
    def _analyze_reward_distribution(self, episode_rewards_detail, episode):
        """分析2：分析奖励分布和协同奖励占比"""
        if not self.reward_tracking['enable_reward_analysis']:
            return {}
        
        reward_stats = {
            'total_reward': 0,
            'reward_types': defaultdict(float),
            'reward_counts': defaultdict(int),
            'collaboration_ratio': 0,
            'penalty_ratio': 0,
            'clipped_rewards': 0
        }
        
        for reward_info in episode_rewards_detail:
            reward_value = reward_info.get('value', 0)
            reward_type = reward_info.get('type', 'unknown')
            
            # 统计奖励类型和数量
            reward_stats['reward_types'][reward_type] += reward_value
            reward_stats['reward_counts'][reward_type] += 1
            reward_stats['total_reward'] += reward_value
            
            # 检查奖励裁剪
            clip_min, clip_max = self.reward_tracking['reward_clip_range']
            if reward_value < clip_min or reward_value > clip_max:
                reward_stats['clipped_rewards'] += 1
        
        # 计算协同奖励占比
        collaboration_reward = reward_stats['reward_types'].get('collaboration', 0)
        if reward_stats['total_reward'] != 0:
            reward_stats['collaboration_ratio'] = abs(collaboration_reward) / abs(reward_stats['total_reward'])
        
        # 计算惩罚占比
        total_penalties = sum(v for v in reward_stats['reward_types'].values() if v < 0)
        if reward_stats['total_reward'] != 0:
            reward_stats['penalty_ratio'] = abs(total_penalties) / abs(reward_stats['total_reward'])
        
        # 记录协同奖励占比历史
        self.training_stats['collaboration_ratios'].append(reward_stats['collaboration_ratio'])
        
        return reward_stats
    
    def _optimized_early_stopping_monitor(self, completion_rates, episode, env_steps):
        """优化3：改进的早停监控机制"""
        self.training_stats['total_env_steps'] = env_steps
        
        # 每5000步监控一次
        if env_steps % self.early_stopping_config['monitor_interval'] == 0 and len(completion_rates) >= self.early_stopping_config['completion_window']:
            
            # 计算滑动平均完成率
            window_size = self.early_stopping_config['completion_window']
            recent_completion = np.mean(completion_rates[-window_size:])
            
            # 记录监控点
            monitor_point = {
                'episode': episode,
                'env_steps': env_steps,
                'completion_rate': recent_completion,
                'timestamp': time.time()
            }
            self.training_stats['monitor_points'].append(monitor_point)
            
            # 检查早停条件
            if len(self.training_stats['monitor_points']) >= self.early_stopping_config['patience_points']:
                # 计算最近20个监控点的改进
                recent_points = self.training_stats['monitor_points'][-self.early_stopping_config['patience_points']:]
                first_completion = recent_points[0]['completion_rate']
                last_completion = recent_points[-1]['completion_rate']
                improvement = last_completion - first_completion
                
                if improvement < self.early_stopping_config['min_improvement']:
                    return True, {
                        'reason': 'completion_rate_plateau',
                        'improvement': improvement,
                        'required_improvement': self.early_stopping_config['min_improvement'],
                        'monitor_points': len(recent_points),
                        'env_steps': env_steps
                    }
        
        return False, {}
    
    def _validate_and_normalize_input(self, state):
        """检查6：验证和归一化网络输入"""
        if not self.normalization_config['enable_input_validation']:
            return state
        
        try:
            if isinstance(state, dict):  # 图模式
                normalized_state = {}
                for key, value in state.items():
                    if key == 'uav_features':
                        # 归一化UAV特征：位置、资源等
                        normalized_value = value.clone()
                        # 位置归一化 (假设前2列是位置)
                        if normalized_value.shape[-1] >= 2:
                            normalized_value[..., :2] /= self.normalization_config['position_scale']
                        # 资源归一化 (假设后续列是资源)
                        if normalized_value.shape[-1] > 2:
                            normalized_value[..., 2:] /= self.normalization_config['resource_scale']
                        normalized_state[key] = normalized_value
                        
                    elif key == 'target_features':
                        # 归一化目标特征
                        normalized_value = value.clone()
                        if normalized_value.shape[-1] >= 2:
                            normalized_value[..., :2] /= self.normalization_config['position_scale']
                        if normalized_value.shape[-1] > 2:
                            normalized_value[..., 2:] /= self.normalization_config['resource_scale']
                        normalized_state[key] = normalized_value
                        
                    elif key == 'edge_features':
                        # 归一化边特征 (距离等)
                        normalized_value = value.clone()
                        normalized_value /= self.normalization_config['distance_scale']
                        normalized_state[key] = normalized_value
                        
                    else:
                        normalized_state[key] = value
                
                # 验证数值范围
                for key, value in normalized_state.items():
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        self.training_stats['input_validation_errors'] += 1
                        print(f"⚠️ 输入验证错误: {key} 包含NaN/Inf")
                        # 用零替换异常值
                        normalized_state[key] = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
                
                return normalized_state
                
            else:  # 扁平模式
                normalized_state = state.clone()
                # 简单的归一化处理
                normalized_state = torch.clamp(normalized_state, -10.0, 10.0)  # 裁剪极端值
                
                if torch.isnan(normalized_state).any() or torch.isinf(normalized_state).any():
                    self.training_stats['input_validation_errors'] += 1
                    normalized_state = torch.nan_to_num(normalized_state, nan=0.0, posinf=1.0, neginf=-1.0)
                
                return normalized_state
                
        except Exception as e:
            self.training_stats['input_validation_errors'] += 1
            print(f"⚠️ 输入归一化失败: {e}")
            return state
    
    def _clip_and_validate_rewards(self, reward, reward_info=None):
        """检查5：奖励裁剪和验证"""
        clip_min, clip_max = self.reward_tracking['reward_clip_range']
        
        # 裁剪奖励
        original_reward = reward
        clipped_reward = np.clip(reward, clip_min, clip_max)
        
        # 验证奖励数值
        if np.isnan(clipped_reward) or np.isinf(clipped_reward):
            print(f"⚠️ 异常奖励值: {original_reward} -> 0")
            clipped_reward = 0.0
        
        # 记录裁剪信息
        if abs(original_reward - clipped_reward) > 1e-6:
            if reward_info:
                reward_info['clipped'] = True
                reward_info['original_value'] = original_reward
        
        return clipped_reward
    
    def _train_level_optimized(self, solver, level_name, dynamic_episodes, success_threshold, early_stopping_config):
        """优化版训练方法"""
        print(f"\n🚀 开始优化训练 {level_name}...")
        
        # 修复混合精度警告
        self._fix_mixed_precision_warnings(solver)
        
        # 初始化统计
        episode_rewards = []
        success_episodes = []
        completion_rates = []
        losses = []
        exploration_rates = []
        
        # 奖励分析统计
        episode_reward_details = []
        collaboration_ratios = []
        
        # 早停机制
        best_reward = float('-inf')
        patience_counter = 0
        patience = early_stopping_config['patience']
        
        # 环境步数计数
        total_env_steps = 0
        
        start_time = time.time()
        min_episodes = dynamic_episodes['min']
        max_episodes = dynamic_episodes['max']
        
        episode = 0
        early_stop_triggered = False
        early_stop_info = {}
        
        print(f"训练配置:")
        print(f"  轮次范围: {min_episodes}-{max_episodes}")
        print(f"  早停监控: 每{self.early_stopping_config['monitor_interval']}步")
        print(f"  完成率窗口: {self.early_stopping_config['completion_window']}")
        print(f"  监控点耐心: {self.early_stopping_config['patience_points']}")
        
        while episode < max_episodes and not early_stop_triggered:
            # 重置环境
            state = solver.env.reset()
            episode_reward = 0
            step_count = 0
            max_steps = 50
            
            # 当前episode的奖励详情
            current_episode_rewards = []
            
            while step_count < max_steps:
                # 验证和归一化输入
                state_tensor = solver._prepare_state_tensor(state)
                normalized_state = self._validate_and_normalize_input(state_tensor)
                
                # 选择动作
                action = solver.select_action(normalized_state)
                
                # 执行动作
                next_state, reward, done, truncated, info = solver.env.step(action.item())
                
                # 奖励裁剪和验证
                reward_info = {
                    'step': step_count,
                    'value': reward,
                    'type': info.get('reward_type', 'unknown'),
                    'clipped': False
                }
                reward = self._clip_and_validate_rewards(reward, reward_info)
                current_episode_rewards.append(reward_info)
                
                episode_reward += reward
                step_count += 1
                total_env_steps += 1
                
                state = next_state
                
                if done or truncated:
                    break
            
            # 记录episode数据
            episode_rewards.append(episode_reward)
            episode_reward_details.append(current_episode_rewards)
            
            # 分析奖励分布
            reward_stats = self._analyze_reward_distribution(current_episode_rewards, episode)
            
            # 检查成功
            is_success = episode_reward >= 1000
            if is_success:
                success_episodes.append(episode)
            
            # 计算完成率
            total_remaining = sum(np.sum(t.remaining_resources) for t in solver.env.targets)
            total_original = sum(np.sum(t.resources) for t in solver.env.targets)
            completion_rate = 1.0 - (total_remaining / (total_original + 1e-6))
            completion_rates.append(completion_rate)
            
            # 训练网络
            batch_size = getattr(solver, 'batch_size', self.config.BATCH_SIZE)
            if len(solver.memory) > batch_size:
                loss = solver.optimize_model()
                if loss is not None:
                    losses.append(loss)
            
            exploration_rates.append(solver.epsilon)
            
            # 优化的控制台输出 (每轮次)
            if episode % 1 == 0:  # 每轮都输出
                collab_ratio = reward_stats.get('collaboration_ratio', 0)
                penalty_ratio = reward_stats.get('penalty_ratio', 0)
                
                print(f"Episode {episode+1:4d}: "
                      f"步数={step_count:2d}, "
                      f"总奖励={episode_reward:7.1f}, "
                      f"协同占比={collab_ratio:.2%}, "
                      f"惩罚占比={penalty_ratio:.2%}, "
                      f"完成率={completion_rate:.2%}")
                
                # 详细奖励类型统计 (每10轮输出一次)
                if episode % 10 == 0 and reward_stats['reward_types']:
                    print(f"  奖励构成: ", end="")
                    for reward_type, value in reward_stats['reward_types'].items():
                        count = reward_stats['reward_counts'][reward_type]
                        print(f"{reward_type}={value:.1f}({count}次) ", end="")
                    print()
            
            # 检查优化的早停条件
            if episode >= min_episodes:
                early_stop_triggered, early_stop_info = self._optimized_early_stopping_monitor(
                    completion_rates, episode, total_env_steps
                )
                
                if early_stop_triggered:
                    print(f"\n⏹️ 优化早停触发: {early_stop_info['reason']}")
                    print(f"   改进幅度: {early_stop_info['improvement']:.4f} < {early_stop_info['required_improvement']:.4f}")
                    print(f"   监控点数: {early_stop_info['monitor_points']}")
                    print(f"   环境步数: {early_stop_info['env_steps']}")
                    break
            
            # 传统早停检查
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience and episode >= min_episodes * 0.5:
                print(f"\n⏹️ 传统早停触发于第 {episode + 1} 轮")
                break
            
            episode += 1
        
        training_time = time.time() - start_time
        actual_episodes = episode + 1
        
        # 协同奖励占比分析
        avg_collaboration_ratio = np.mean(self.training_stats['collaboration_ratios'][-actual_episodes:]) if self.training_stats['collaboration_ratios'] else 0
        
        print(f"\n{level_name} 优化训练完成:")
        print(f"  实际轮次: {actual_episodes}")
        print(f"  训练时间: {training_time:.1f}秒")
        print(f"  总环境步数: {total_env_steps}")
        print(f"  平均协同奖励占比: {avg_collaboration_ratio:.2%}")
        print(f"  输入验证错误: {self.training_stats['input_validation_errors']}")
        
        # 协同奖励优化建议
        if avg_collaboration_ratio > self.reward_tracking['collaboration_threshold']:
            print(f"⚠️ 协同奖励占比过高 ({avg_collaboration_ratio:.2%} > {self.reward_tracking['collaboration_threshold']:.2%})")
            print(f"   建议: 降低协同奖励权重或增加其他奖励类型的权重")
        
        performance = {
            'episodes': actual_episodes,
            'training_time': training_time,
            'total_env_steps': total_env_steps,
            'total_successes': len(success_episodes),
            'final_success_rate': len(success_episodes) / actual_episodes,
            'final_avg_reward': np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards),
            'final_completion_rate': np.mean(completion_rates[-50:]) if len(completion_rates) >= 50 else np.mean(completion_rates),
            'episode_rewards': episode_rewards,
            'completion_rates': completion_rates,
            'losses': losses,
            'exploration_rates': exploration_rates,
            'best_reward': best_reward,
            'early_stop_info': early_stop_info,
            'collaboration_analysis': {
                'avg_ratio': avg_collaboration_ratio,
                'ratios_history': self.training_stats['collaboration_ratios'][-actual_episodes:],
                'threshold_exceeded': avg_collaboration_ratio > self.reward_tracking['collaboration_threshold']
            },
            'input_validation': {
                'errors': self.training_stats['input_validation_errors'],
                'normalization_enabled': self.normalization_config['enable_input_validation']
            }
        }
        
        return performance
    
    def train_optimized_curriculum(self, base_episodes_per_level=200, success_threshold=0.8):
        """执行优化的课程学习训练"""
        print("=" * 80)
        print("开始优化版高级课程学习训练")
        print("=" * 80)
        
        set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
        curriculum_scenarios = get_curriculum_scenarios()
        
        solver = None
        previous_model_path = None
        
        for level_idx, (scenario_func, level_name, description) in enumerate(curriculum_scenarios):
            print(f"\n{'='*60}")
            print(f"训练级别 {level_idx + 1}: {level_name}")
            print(f"场景描述: {description}")
            print(f"{'='*60}")
            
            # 获取场景
            uavs, targets, obstacles = scenario_func(self.config.OBSTACLE_TOLERANCE)
            
            # 创建新求解器
            solver = self._create_optimized_solver(uavs, targets, obstacles, level_idx, level_name, previous_model_path)
            
            # 动态确定训练轮次
            dynamic_episodes = self._determine_dynamic_episodes(level_idx, level_name, base_episodes_per_level)
            
            # 获取早停配置
            early_stopping_config = self._get_flexible_early_stopping_config(level_name)
            
            # 训练当前级别
            level_performance = self._train_level_optimized(
                solver, level_name, dynamic_episodes, success_threshold, early_stopping_config
            )
            
            # 保存模型
            model_path = os.path.join(self.output_dir, f"model_level_{level_idx + 1}_{level_name}.pth")
            torch.save(solver.policy_net.state_dict(), model_path)
            previous_model_path = model_path
            
            print(f"模型已保存: {model_path}")
        
        print(f"\n{'='*80}")
        print("优化版课程学习训练完成！")
        print(f"详细报告保存在: {self.output_dir}")
        print(f"{'='*80}")
        
        return solver
    
    def _create_optimized_solver(self, uavs, targets, obstacles, level_idx, level_name, previous_model_path):
        """创建优化的求解器"""
        print(f"🔄 创建优化求解器...")
        
        graph = DirectedGraph(uavs, targets, len(uavs[0].resources), obstacles, self.config)
        i_dim = len(uavs) * len(targets) * len(uavs[0].resources)
        h_dim = [256, 128]
        o_dim = len(uavs) * len(targets) * len(uavs[0].resources)
        
        solver = GraphRLSolver(
            uavs=uavs, targets=targets, graph=graph, obstacles=obstacles,
            i_dim=i_dim, h_dim=h_dim, o_dim=o_dim, config=self.config,
            obs_mode="graph", network_type="ZeroShotGNN"
        )
        
        # 权重迁移
        if previous_model_path and os.path.exists(previous_model_path):
            try:
                previous_state_dict = torch.load(previous_model_path, map_location=solver.device)
                current_state_dict = solver.policy_net.state_dict()
                loaded_keys = []
                
                for key, value in previous_state_dict.items():
                    if key in current_state_dict and current_state_dict[key].shape == value.shape:
                        current_state_dict[key] = value
                        loaded_keys.append(key)
                
                solver.policy_net.load_state_dict(current_state_dict)
                solver.target_net.load_state_dict(current_state_dict)
                
                print(f"   ✅ 成功加载 {len(loaded_keys)} 个权重")
                
            except Exception as e:
                print(f"   ⚠️ 权重加载失败: {e}")
        
        print(f"   求解器创建完成 (动作空间: {o_dim})")
        return solver
    
    def _determine_dynamic_episodes(self, level_idx, level_name, base_episodes):
        """动态确定训练轮次"""
        difficulty_multipliers = {
            'easy': 0.7, 'simple': 0.8, 'medium': 1.0, 'hard': 1.3, 'expert': 1.5
        }
        
        difficulty = 'medium'
        for diff_key in difficulty_multipliers.keys():
            if diff_key.lower() in level_name.lower():
                difficulty = diff_key
                break
        
        multiplier = difficulty_multipliers[difficulty]
        
        min_episodes = max(self.dynamic_config['min_episodes_per_level'], int(base_episodes * multiplier * 0.5))
        max_episodes = min(self.dynamic_config['max_episodes_per_level'], int(base_episodes * multiplier * 1.5))
        
        return {
            'min': min_episodes,
            'max': max_episodes,
            'base': base_episodes,
            'difficulty': difficulty,
            'multiplier': multiplier
        }
    
    def _get_flexible_early_stopping_config(self, level_name):
        """获取灵活早停配置"""
        difficulty = 'medium'
        for diff_key in self.early_stopping_config['patience_multiplier'].keys():
            if diff_key.lower() in level_name.lower():
                difficulty = diff_key
                break
        
        multiplier = self.early_stopping_config['patience_multiplier'][difficulty]
        patience = int(self.early_stopping_config['base_patience'] * multiplier)
        
        return {
            'patience': patience,
            'difficulty': difficulty,
            'multiplier': multiplier,
            'min_improvement': self.early_stopping_config['min_improvement']
        }


def main():
    """主函数"""
    print("优化版高级课程学习训练系统")
    print("=" * 50)
    
    config = Config()
    trainer = OptimizedAdvancedCurriculumTrainer(config)
    
    final_solver = trainer.train_optimized_curriculum(
        base_episodes_per_level=200,
        success_threshold=0.8
    )
    
    print("优化版训练完成！")
    return final_solver


if __name__ == "__main__":
    main()
