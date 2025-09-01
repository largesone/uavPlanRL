#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试奖励学习问题 - 分析为什么1000分奖励没有被学习到
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict, deque
import pickle
import os
from datetime import datetime

class RewardLearningDebugger:
    def __init__(self):
        self.episode_rewards = []
        self.step_rewards = []
        self.q_values = []
        self.target_completion_rewards = []
        self.synergy_rewards = []
        self.final_success_rewards = []
        self.learning_rates = []
        self.losses = []
        
    def log_episode_data(self, episode, total_reward, step_rewards, q_values=None, 
                        target_rewards=None, synergy_rewards=None, final_reward=None,
                        learning_rate=None, loss=None):
        """记录每个episode的详细数据"""
        self.episode_rewards.append((episode, total_reward))
        self.step_rewards.extend([(episode, step, reward) for step, reward in enumerate(step_rewards)])
        
        if q_values is not None:
            self.q_values.extend([(episode, step, q_val) for step, q_val in enumerate(q_values)])
        
        if target_rewards:
            self.target_completion_rewards.extend([(episode, reward) for reward in target_rewards])
        
        if synergy_rewards:
            self.synergy_rewards.extend([(episode, reward) for reward in synergy_rewards])
            
        if final_reward is not None:
            self.final_success_rewards.append((episode, final_reward))
            
        if learning_rate is not None:
            self.learning_rates.append((episode, learning_rate))
            
        if loss is not None:
            self.losses.append((episode, loss))
    
    def analyze_reward_distribution(self):
        """分析奖励分布"""
        print("=== 奖励分布分析 ===")
        
        # 分析episode总奖励
        if self.episode_rewards:
            rewards = [r for _, r in self.episode_rewards]
            print(f"Episode总奖励统计:")
            print(f"  平均值: {np.mean(rewards):.2f}")
            print(f"  中位数: {np.median(rewards):.2f}")
            print(f"  最大值: {np.max(rewards):.2f}")
            print(f"  最小值: {np.min(rewards):.2f}")
            print(f"  标准差: {np.std(rewards):.2f}")
            
            # 统计高奖励episode
            high_reward_episodes = [(ep, r) for ep, r in self.episode_rewards if r > 800]
            print(f"  高奖励(>800)episode数量: {len(high_reward_episodes)}")
            if high_reward_episodes:
                print(f"  高奖励episode: {high_reward_episodes[:10]}")  # 显示前10个
        
        # 分析最终成功奖励
        if self.final_success_rewards:
            final_rewards = [r for _, r in self.final_success_rewards]
            print(f"\n最终成功奖励统计:")
            print(f"  获得1000分奖励的episode数: {len([r for r in final_rewards if r >= 1000])}")
            print(f"  总的最终奖励次数: {len(final_rewards)}")
            print(f"  最终奖励分布: {set(final_rewards)}")
        
        # 分析协同奖励
        if self.synergy_rewards:
            synergy_vals = [r for _, r in self.synergy_rewards]
            print(f"\n协同奖励统计:")
            print(f"  协同奖励次数: {len(synergy_vals)}")
            print(f"  平均协同奖励: {np.mean(synergy_vals):.2f}")
            print(f"  协同奖励分布: {set(synergy_vals)}")
    
    def analyze_learning_progress(self):
        """分析学习进度"""
        print("\n=== 学习进度分析 ===")
        
        if self.q_values:
            # 分析Q值变化
            episodes = [ep for ep, _, _ in self.q_values]
            q_vals = [q for _, _, q in self.q_values]
            
            # 按episode分组计算平均Q值
            episode_q_means = defaultdict(list)
            for ep, _, q in self.q_values:
                episode_q_means[ep].append(q)
            
            avg_q_by_episode = [(ep, np.mean(qs)) for ep, qs in episode_q_means.items()]
            avg_q_by_episode.sort()
            
            if len(avg_q_by_episode) > 10:
                print(f"Q值变化趋势 (前10个episode):")
                for ep, avg_q in avg_q_by_episode[:10]:
                    print(f"  Episode {ep}: 平均Q值 = {avg_q:.3f}")
                
                print(f"Q值变化趋势 (后10个episode):")
                for ep, avg_q in avg_q_by_episode[-10:]:
                    print(f"  Episode {ep}: 平均Q值 = {avg_q:.3f}")
        
        if self.losses:
            recent_losses = self.losses[-20:] if len(self.losses) > 20 else self.losses
            avg_recent_loss = np.mean([loss for _, loss in recent_losses])
            print(f"\n最近损失值: {avg_recent_loss:.6f}")
    
    def detect_learning_issues(self):
        """检测学习问题"""
        print("\n=== 学习问题检测 ===")
        
        issues = []
        
        # 检测奖励方差问题
        if self.episode_rewards:
            rewards = [r for _, r in self.episode_rewards]
            if len(rewards) > 10:
                recent_rewards = rewards[-20:] if len(rewards) > 20 else rewards
                reward_std = np.std(recent_rewards)
                reward_mean = np.mean(recent_rewards)
                
                if reward_std > reward_mean * 0.5:
                    issues.append(f"奖励方差过大: std={reward_std:.2f}, mean={reward_mean:.2f}")
        
        # 检测Q值爆炸/消失
        if self.q_values:
            q_vals = [q for _, _, q in self.q_values]
            max_q = np.max(q_vals)
            min_q = np.min(q_vals)
            
            if max_q > 1000:
                issues.append(f"Q值可能爆炸: max_q={max_q:.2f}")
            if abs(min_q) > 1000:
                issues.append(f"Q值可能爆炸: min_q={min_q:.2f}")
            if max_q - min_q < 0.01:
                issues.append(f"Q值可能消失: range={max_q - min_q:.6f}")
        
        # 检测学习停滞
        if len(self.episode_rewards) > 50:
            recent_50 = [r for _, r in self.episode_rewards[-50:]]
            early_50 = [r for _, r in self.episode_rewards[:50]]
            
            if abs(np.mean(recent_50) - np.mean(early_50)) < 10:
                issues.append("学习可能停滞: 最近50轮与前50轮平均奖励差异很小")
        
        if issues:
            print("发现的问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("未发现明显学习问题")
    
    def generate_debug_plots(self, save_dir="debug_output"):
        """生成调试图表"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 奖励趋势图
        if self.episode_rewards:
            plt.figure(figsize=(15, 10))
            
            # 子图1: Episode总奖励
            plt.subplot(2, 3, 1)
            episodes, rewards = zip(*self.episode_rewards)
            plt.plot(episodes, rewards, alpha=0.7, linewidth=1)
            
            # 添加移动平均
            if len(rewards) > 10:
                window = min(20, len(rewards) // 5)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}轮移动平均')
                plt.legend()
            
            plt.title('Episode总奖励趋势')
            plt.xlabel('Episode')
            plt.ylabel('总奖励')
            plt.grid(True, alpha=0.3)
            
            # 子图2: 奖励分布直方图
            plt.subplot(2, 3, 2)
            plt.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
            plt.title('奖励分布直方图')
            plt.xlabel('奖励值')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
            
            # 子图3: 最终成功奖励
            if self.final_success_rewards:
                plt.subplot(2, 3, 3)
                final_eps, final_rewards = zip(*self.final_success_rewards)
                plt.scatter(final_eps, final_rewards, alpha=0.7, s=20)
                plt.title('最终成功奖励分布')
                plt.xlabel('Episode')
                plt.ylabel('最终奖励')
                plt.grid(True, alpha=0.3)
            
            # 子图4: Q值变化
            if self.q_values:
                plt.subplot(2, 3, 4)
                # 计算每个episode的平均Q值
                episode_q_means = defaultdict(list)
                for ep, _, q in self.q_values:
                    episode_q_means[ep].append(q)
                
                avg_q_episodes = []
                avg_q_values = []
                for ep in sorted(episode_q_means.keys()):
                    avg_q_episodes.append(ep)
                    avg_q_values.append(np.mean(episode_q_means[ep]))
                
                plt.plot(avg_q_episodes, avg_q_values, 'g-', alpha=0.7)
                plt.title('平均Q值变化')
                plt.xlabel('Episode')
                plt.ylabel('平均Q值')
                plt.grid(True, alpha=0.3)
            
            # 子图5: 损失变化
            if self.losses:
                plt.subplot(2, 3, 5)
                loss_episodes, loss_values = zip(*self.losses)
                plt.plot(loss_episodes, loss_values, 'orange', alpha=0.7)
                plt.title('训练损失变化')
                plt.xlabel('Episode')
                plt.ylabel('损失值')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
            
            # 子图6: 学习率变化
            if self.learning_rates:
                plt.subplot(2, 3, 6)
                lr_episodes, lr_values = zip(*self.learning_rates)
                plt.plot(lr_episodes, lr_values, 'purple', alpha=0.7)
                plt.title('学习率变化')
                plt.xlabel('Episode')
                plt.ylabel('学习率')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/reward_learning_debug.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"调试图表已保存至: {save_dir}/reward_learning_debug.png")
    
    def save_debug_data(self, save_dir="debug_output"):
        """保存调试数据"""
        os.makedirs(save_dir, exist_ok=True)
        
        debug_data = {
            'episode_rewards': self.episode_rewards,
            'step_rewards': self.step_rewards,
            'q_values': self.q_values,
            'target_completion_rewards': self.target_completion_rewards,
            'synergy_rewards': self.synergy_rewards,
            'final_success_rewards': self.final_success_rewards,
            'learning_rates': self.learning_rates,
            'losses': self.losses,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存为pickle文件
        with open(f"{save_dir}/debug_data.pkl", 'wb') as f:
            pickle.dump(debug_data, f)
        
        # 保存为JSON文件（部分数据）
        json_data = {
            'episode_rewards': self.episode_rewards[-100:],  # 只保存最近100个
            'final_success_rewards': self.final_success_rewards,
            'synergy_rewards': self.synergy_rewards[-50:],  # 只保存最近50个
            'summary': {
                'total_episodes': len(self.episode_rewards),
                'high_reward_episodes': len([r for _, r in self.episode_rewards if r > 800]),
                'final_success_count': len(self.final_success_rewards),
                'synergy_reward_count': len(self.synergy_rewards)
            }
        }
        
        with open(f"{save_dir}/debug_summary.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"调试数据已保存至: {save_dir}/")
    
    def generate_report(self):
        """生成完整的调试报告"""
        print("=" * 60)
        print("奖励学习调试报告")
        print("=" * 60)
        
        self.analyze_reward_distribution()
        self.analyze_learning_progress()
        self.detect_learning_issues()
        
        print("\n=== 建议的解决方案 ===")
        
        # 基于分析结果给出建议
        if self.final_success_rewards:
            success_count = len([r for _, r in self.final_success_rewards if r >= 1000])
            total_episodes = len(self.episode_rewards) if self.episode_rewards else 0
            
            if success_count > 0 and total_episodes > 0:
                success_rate = success_count / total_episodes
                print(f"1000分奖励获得率: {success_rate:.3f} ({success_count}/{total_episodes})")
                
                if success_rate < 0.1:
                    print("建议:")
                    print("  1. 检查奖励函数是否正确计算和传递")
                    print("  2. 增加奖励的权重或缩放")
                    print("  3. 调整探索策略，增加获得高奖励的机会")
                    print("  4. 检查经验回放缓冲区是否正确存储高奖励经验")
                elif success_rate < 0.3:
                    print("建议:")
                    print("  1. 优化网络结构以更好地学习高价值状态")
                    print("  2. 调整学习率和批次大小")
                    print("  3. 增加高奖励经验的采样权重")
        
        print("\n" + "=" * 60)

# 使用示例和集成建议
def integrate_with_training_loop():
    """展示如何集成到训练循环中"""
    print("\n=== 集成建议 ===")
    print("""
在训练循环中添加以下代码来使用调试器:

# 在训练开始前初始化
debugger = RewardLearningDebugger()

# 在每个episode结束后记录数据
debugger.log_episode_data(
    episode=episode,
    total_reward=episode_reward,
    step_rewards=step_rewards_list,
    q_values=q_values_list,
    target_rewards=target_completion_rewards,
    synergy_rewards=synergy_rewards_list,
    final_reward=final_success_reward,
    learning_rate=current_lr,
    loss=current_loss
)

# 定期生成报告（比如每100个episode）
if episode % 100 == 0:
    debugger.generate_report()
    debugger.generate_debug_plots()
    debugger.save_debug_data()
    """)

if __name__ == "__main__":
    # 创建示例调试器并展示功能
    debugger = RewardLearningDebugger()
    
    # 模拟一些数据来演示功能
    print("创建示例调试数据...")
    np.random.seed(42)
    
    for episode in range(100):
        # 模拟episode奖励
        base_reward = np.random.normal(100, 50)
        if episode > 50 and np.random.random() < 0.2:  # 20%概率获得高奖励
            base_reward += 900  # 模拟获得1000分奖励
        
        step_rewards = np.random.normal(5, 2, size=20)
        q_values = np.random.normal(base_reward/10, 10, size=20)
        
        # 模拟最终成功奖励
        final_reward = None
        if base_reward > 800:
            final_reward = 1000.0
        
        # 模拟协同奖励
        synergy_rewards = []
        if np.random.random() < 0.3:
            synergy_rewards = [300.0] * np.random.randint(1, 4)
        
        debugger.log_episode_data(
            episode=episode,
            total_reward=base_reward,
            step_rewards=step_rewards,
            q_values=q_values,
            synergy_rewards=synergy_rewards,
            final_reward=final_reward,
            learning_rate=1e-4 * (0.99 ** episode),
            loss=np.random.exponential(0.01)
        )
    
    # 生成完整报告
    debugger.generate_report()
    debugger.generate_debug_plots()
    debugger.save_debug_data()
    
    # 显示集成建议
    integrate_with_training_loop()
