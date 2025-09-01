#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强训练器
实现优化的训练过程，包括动态epsilon调度和模型管理
"""

import numpy as np
import torch
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

from enhanced_training_config import EnhancedTrainingConfig
from model_manager import ModelManager
from baseline_config import BaselineConfig

class EnhancedTrainer:
    """增强训练器"""
    
    def __init__(self, solver, config, output_dir):
        """
        初始化增强训练器
        
        Args:
            solver: 求解器对象
            config: 配置对象
            output_dir: 输出目录
        """
        self.solver = solver
        self.config = config
        self.output_dir = output_dir
        
        # 创建模型管理器
        model_save_dir = os.path.join(output_dir, "saved_models")
        self.model_manager = ModelManager(
            save_dir=model_save_dir,
            max_models=EnhancedTrainingConfig.SAVE_TOP_N_MODELS
        )
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_scores': [],
            'epsilon_history': [],
            'loss_history': [],
            'model_saves': [],
            'phase_transitions': []
        }
        
        print(f"增强训练器初始化完成")
        print(f"输出目录: {output_dir}")
        print(f"模型保存目录: {model_save_dir}")
    
    def train_enhanced(self, episodes, use_baseline=False):
        """
        增强训练主函数
        
        Args:
            episodes: 训练轮次
            use_baseline: 是否使用基线配置
            
        Returns:
            dict: 训练结果
        """
        print(f"\n🚀 开始增强训练")
        print(f"训练轮次: {episodes}")
        print(f"使用配置: {'基线配置' if use_baseline else '增强配置'}")
        
        # 选择配置
        config_class = BaselineConfig if use_baseline else EnhancedTrainingConfig
        
        start_time = time.time()
        best_score = float('-inf')
        
        # 计算阶段转换点
        exploration_episodes = int(episodes * config_class.EXPLORATION_PHASE_RATIO)
        
        print(f"探索阶段: 0-{exploration_episodes} 轮")
        print(f"利用阶段: {exploration_episodes}-{episodes} 轮")
        
        for episode in range(episodes):
            # 动态调整epsilon
            if not use_baseline:
                epsilon = config_class.get_epsilon_schedule(episode, episodes)
                self.solver.epsilon = epsilon
            else:
                # 基线配置使用原有的epsilon衰减
                self.solver.epsilon = max(
                    config_class.EPSILON_END,
                    self.solver.epsilon * config_class.EPSILON_DECAY
                )
            
            # 记录阶段转换
            if episode == exploration_episodes and not use_baseline:
                self.training_stats['phase_transitions'].append({
                    'episode': episode,
                    'phase': 'exploitation',
                    'epsilon': self.solver.epsilon
                })
                print(f"\n🎯 进入利用阶段 (Episode {episode})")
                print(f"   当前Epsilon: {self.solver.epsilon:.6f}")
            
            # 执行一个episode
            episode_reward, episode_score = self._run_episode(episode)
            
            # 记录统计信息
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_scores'].append(episode_score)
            self.training_stats['epsilon_history'].append(self.solver.epsilon)
            
            # 训练网络
            if len(self.solver.memory) > self.config.BATCH_SIZE:
                loss = self.solver.optimize_model()
                if loss is not None:
                    self.training_stats['loss_history'].append(loss)
            
            # 更新最佳分数
            if episode_score > best_score:
                best_score = episode_score
            
            # 模型保存检查
            if not use_baseline:  # 只在增强模式下保存多个模型
                if self.model_manager.should_save_model(
                    episode, episode_score,
                    min_episodes=config_class.MIN_EPISODES_FOR_SAVE,
                    save_interval=config_class.MODEL_SAVE_INTERVAL
                ):
                    filepath = self.model_manager.save_model(
                        self.solver.policy_net, episode, episode_score,
                        additional_info={
                            'epsilon': self.solver.epsilon,
                            'total_episodes': episodes,
                            'config_type': 'enhanced'
                        }
                    )
                    self.training_stats['model_saves'].append({
                        'episode': episode,
                        'score': episode_score,
                        'filepath': filepath
                    })
            
            # 定期输出进度
            if (episode + 1) % 100 == 0:
                self._print_progress(episode + 1, episodes, episode_reward, episode_score)
        
        training_time = time.time() - start_time
        
        # 生成训练报告
        results = self._generate_training_report(episodes, training_time, use_baseline)
        
        # 保存最终模型（基线模式）
        if use_baseline:
            final_model_path = os.path.join(self.output_dir, "final_baseline_model.pth")
            torch.save({
                'model_state_dict': self.solver.policy_net.state_dict(),
                'episode': episodes,
                'score': best_score,
                'config_type': 'baseline',
                'training_time': training_time
            }, final_model_path)
            print(f"基线模型已保存: {final_model_path}")
        
        return results
    
    def _run_episode(self, episode):
        """
        运行单个episode
        
        Args:
            episode: 当前轮次
            
        Returns:
            tuple: (episode_reward, episode_score)
        """
        state = self.solver.env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 50
        
        while step_count < max_steps:
            # 选择动作
            state_tensor = self.solver._prepare_state_tensor(state)
            action = self.solver.select_action(state_tensor)
            
            # 执行动作
            next_state, reward, done, truncated, info = self.solver.env.step(action.item())
            
            episode_reward += reward
            step_count += 1
            
            # 存储经验
            self.solver.memory.push(state, action, reward, next_state, done or truncated)
            
            state = next_state
            
            if done or truncated:
                break
        
        # 计算episode分数（可以是奖励、完成率等的组合）
        episode_score = self._calculate_episode_score(episode_reward, info, step_count)
        
        return episode_reward, episode_score
    
    def _calculate_episode_score(self, reward, info, steps):
        """
        计算episode分数
        
        Args:
            reward: 总奖励
            info: 最后一步的信息
            steps: 步数
            
        Returns:
            float: episode分数
        """
        # 综合考虑奖励、效率等因素
        base_score = reward
        
        # 效率奖励（步数越少越好）
        efficiency_bonus = max(0, (50 - steps) * 2)
        
        # 成功完成奖励
        success_bonus = 100 if info.get('done', False) else 0
        
        total_score = base_score + efficiency_bonus + success_bonus
        return total_score
    
    def _print_progress(self, episode, total_episodes, reward, score):
        """打印训练进度"""
        recent_rewards = self.training_stats['episode_rewards'][-100:]
        recent_scores = self.training_stats['episode_scores'][-100:]
        
        avg_reward = np.mean(recent_rewards)
        avg_score = np.mean(recent_scores)
        
        print(f"Episode {episode}/{total_episodes}: "
              f"Reward={reward:.1f}, Score={score:.1f}, "
              f"Avg_Reward={avg_reward:.1f}, Avg_Score={avg_score:.1f}, "
              f"Epsilon={self.solver.epsilon:.6f}")
    
    def _generate_training_report(self, episodes, training_time, use_baseline):
        """生成训练报告"""
        results = {
            'episodes': episodes,
            'training_time': training_time,
            'config_type': 'baseline' if use_baseline else 'enhanced',
            'final_epsilon': self.solver.epsilon,
            'best_score': max(self.training_stats['episode_scores']),
            'avg_final_score': np.mean(self.training_stats['episode_scores'][-100:]),
            'model_saves_count': len(self.training_stats['model_saves']),
            'training_stats': self.training_stats
        }
        
        print(f"\n📊 训练完成报告:")
        print(f"   训练时间: {training_time:.1f}秒")
        print(f"   最佳分数: {results['best_score']:.1f}")
        print(f"   最终100轮平均分数: {results['avg_final_score']:.1f}")
        print(f"   最终Epsilon: {results['final_epsilon']:.6f}")
        
        if not use_baseline:
            print(f"   保存的模型数量: {results['model_saves_count']}")
            print(self.model_manager.get_model_summary())
        
        # 保存训练曲线
        self._save_training_curves(use_baseline)
        
        return results
    
    def _save_training_curves(self, use_baseline):
        """保存训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        axes[0, 0].plot(self.training_stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 分数曲线
        axes[0, 1].plot(self.training_stats['episode_scores'])
        axes[0, 1].set_title('Episode Scores')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        
        # Epsilon曲线
        axes[1, 0].plot(self.training_stats['epsilon_history'])
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_yscale('log')
        
        # 损失曲线
        if self.training_stats['loss_history']:
            axes[1, 1].plot(self.training_stats['loss_history'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        
        # 保存图片
        config_type = 'baseline' if use_baseline else 'enhanced'
        curve_path = os.path.join(self.output_dir, f"training_curves_{config_type}.png")
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: {curve_path}")