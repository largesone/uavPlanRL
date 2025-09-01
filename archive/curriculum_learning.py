#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课程学习训练脚本
从易到难的场景序列训练，提高学习效率和最终性能
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import pickle
from datetime import datetime
from typing import List, Dict, Tuple

from main import GraphRLSolver, set_chinese_font
from config import Config
from scenarios import get_curriculum_scenarios
from environment import DirectedGraph
from comprehensive_debug_analysis import ComprehensiveDebugAnalyzer
from comprehensive_debug_analysis import ComprehensiveDebugAnalyzer
from scenarios import test_large_curriculum_scenarios

class CurriculumLearningTrainer:
    """课程学习训练器"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.config.ENABLE_REWARD_DEBUG = False  # 减少输出
        
        # 训练历史记录
        self.training_history = {
            'levels': [],
            'performance': [],
            'convergence_data': [],
            'training_config': {
                'episodes_per_level': None,
                'success_threshold': None,
                'early_stopping': True,
                'patience': getattr(self.config, 'PATIENCE', 100),
                'learning_rate': getattr(self.config, 'LEARNING_RATE', 1e-5),
                'batch_size': getattr(self.config, 'BATCH_SIZE', 16),
                'memory_size': getattr(self.config, 'MEMORY_SIZE', 15000),
                'network_type': 'ZeroShotGNN',
                'optimizer': 'AdamW',
                'gamma': getattr(self.config, 'GAMMA', 0.99),
                'epsilon_start': getattr(self.config, 'EPSILON_START', 0.9),
                'epsilon_end': getattr(self.config, 'EPSILON_END', 0.1),
                'epsilon_decay': getattr(self.config, 'EPSILON_DECAY', 0.9995),
                'target_update_freq': getattr(self.config, 'TARGET_UPDATE_FREQ', 20),
                'use_prioritized_replay': getattr(self.config.training_config, 'use_prioritized_replay', True),
                'per_alpha': getattr(self.config.training_config, 'per_alpha', 0.6),
                'per_beta_start': getattr(self.config.training_config, 'per_beta_start', 0.4),
                'gradient_clipping': getattr(self.config.training_config, 'use_gradient_clipping', True),
                'max_grad_norm': getattr(self.config.training_config, 'max_grad_norm', 1.0),
                'reward_normalization': getattr(self.config, 'REWARD_NORMALIZATION', True),
                'reward_scale': getattr(self.config, 'REWARD_SCALE', 0.3)
            }
        }
        
        # 输出目录
        self.output_dir = f"output/curriculum_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"课程学习训练器初始化完成")
        print(f"输出目录: {self.output_dir}")
        print(f"训练配置: 学习率={self.training_history['training_config']['learning_rate']}, "
              f"批次大小={self.training_history['training_config']['batch_size']}")
    
    def train_curriculum(self, episodes_per_level=200, success_threshold=0.8):
        """
        执行课程学习训练
        
        Args:
            episodes_per_level: 每个难度级别的训练轮次
            success_threshold: 进入下一级别的成功率阈值
        """
        print("=" * 80)
        print("开始课程学习训练")
        print("=" * 80)
        
        # 更新训练配置记录
        self.training_history['training_config']['episodes_per_level'] = episodes_per_level
        self.training_history['training_config']['success_threshold'] = success_threshold
        
        # 记录实际使用的配置参数
        if hasattr(self, 'config'):
            self.training_history['training_config'].update({
                'actual_learning_rate': getattr(self.config, 'LEARNING_RATE', 1e-5),
                'actual_batch_size': getattr(self.config, 'BATCH_SIZE', 16),
                'actual_memory_size': getattr(self.config, 'MEMORY_SIZE', 15000),
                'network_architecture': getattr(self.config, 'NETWORK_TYPE', 'ZeroShotGNN'),
                'training_mode': getattr(self.config, 'TRAINING_MODE', 'zero_shot_train'),
                'use_phrrt': getattr(self.config, 'USE_PHRRT_DURING_TRAINING', True),
                'obstacle_tolerance': getattr(self.config, 'OBSTACLE_TOLERANCE', 50.0),
                'map_size': getattr(self.config, 'MAP_SIZE', 1000.0)
            })
        
        # 设置字体
        set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
        
        # 获取课程场景
        curriculum_scenarios = get_curriculum_scenarios()
        
        # 初始化求解器（使用第一个场景）
        solver = None
        
        for level_idx, (scenario_func, level_name, description) in enumerate(curriculum_scenarios):
            print(f"\n{'='*60}")
            print(f"训练级别 {level_idx + 1}: {level_name}")
            print(f"场景描述: {description}")
            print(f"{'='*60}")
            
            # 获取当前级别的场景
            uavs, targets, obstacles = scenario_func(self.config.OBSTACLE_TOLERANCE)
            
            # 创建或更新求解器
            if solver is None:
                # 首次创建求解器
                graph = DirectedGraph(uavs, targets, len(uavs[0].resources), obstacles, self.config)
                i_dim = len(uavs) * len(targets) * len(uavs[0].resources)
                h_dim = [256, 128]
                o_dim = len(uavs) * len(targets) * len(uavs[0].resources)
                
                solver = GraphRLSolver(
                    uavs=uavs, targets=targets, graph=graph, obstacles=obstacles,
                    i_dim=i_dim, h_dim=h_dim, o_dim=o_dim, config=self.config,
                    obs_mode="graph", network_type="ZeroShotGNN"
                )
                print(f"求解器初始化完成")
            else:
                # 更新求解器的环境
                solver.env.uavs = uavs
                solver.env.targets = targets
                solver.env.obstacles = obstacles
                solver.env.graph = DirectedGraph(uavs, targets, len(uavs[0].resources), obstacles, self.config)
                print(f"求解器环境已更新")
            
            # 训练当前级别
            level_performance = self._train_level(
                solver, level_name, episodes_per_level, success_threshold
            )
            
            # 记录训练历史
            self.training_history['levels'].append({
                'level': level_idx + 1,
                'name': level_name,
                'description': description,
                'episodes': episodes_per_level,
                'performance': level_performance
            })
            
            # 检查是否达到成功阈值
            final_success_rate = level_performance.get('final_success_rate', 0)
            if final_success_rate < success_threshold:
                print(f"⚠️ 级别 {level_idx + 1} 未达到成功阈值 ({final_success_rate:.2%} < {success_threshold:.2%})")
                print(f"建议增加训练轮次或调整网络参数")
            else:
                print(f"✅ 级别 {level_idx + 1} 训练成功 ({final_success_rate:.2%} >= {success_threshold:.2%})")
            
            # 保存当前级别的最优模型
            best_model_path = os.path.join(self.output_dir, f"best_model_level_{level_idx + 1}_{level_name}.pth")
            
            # 检查是否有最佳模型保存路径
            if hasattr(solver, '_last_saved_model_path') and solver._last_saved_model_path:
                # 复制最佳模型到课程学习目录
                import shutil
                try:
                    shutil.copy2(solver._last_saved_model_path, best_model_path)
                    print(f"✅ 最优模型已保存: {best_model_path}")
                except Exception as e:
                    # 如果复制失败，保存当前模型
                    torch.save(solver.policy_net.state_dict(), best_model_path)
                    print(f"⚠️ 复制最优模型失败，保存当前模型: {best_model_path}")
            else:
                # 保存当前模型
                torch.save(solver.policy_net.state_dict(), best_model_path)
                print(f"📁 当前模型已保存: {best_model_path}")
            
            # 记录模型路径到性能数据中
            level_performance['best_model_path'] = best_model_path
        
        # 生成最终报告
        self._generate_final_report()
        
        print(f"\n{'='*80}")
        print("课程学习训练完成！")
        print(f"详细报告保存在: {self.output_dir}")
        print(f"{'='*80}")
        
        return solver
    
    def _train_level(self, solver, level_name, episodes, success_threshold):
        """训练单个难度级别 - 增强版，输出每轮奖励变化详情"""
        print(f"\n开始训练 {level_name}...")
        print(f"目标轮次: {episodes}, 成功阈值: {success_threshold:.2%}")
        
        # 初始化分析器
        analyzer = ComprehensiveDebugAnalyzer(
            os.path.join(self.output_dir, f"debug_{level_name}")
        )
        
        # 训练统计
        episode_rewards = []
        success_episodes = []
        completion_rates = []
        losses = []  # 记录损失值
        exploration_rates = []  # 记录探索率
        
        # 新增：详细奖励分解记录
        detailed_reward_history = []  # 记录每个episode的详细奖励分解
        step_reward_history = []      # 记录每步的奖励分解
        
        # 早停机制
        best_reward = float('-inf')
        patience_counter = 0
        patience = 100  # 早停耐心值
        
        start_time = time.time()
        
        # 进度条初始化
        print(f"\n🚀 开始训练 {level_name}")
        print(f"训练进度: [{'':50}] 0/{episodes} (0.0%)")
        last_progress_update = 0
        
        # 训练阶段标记
        training_phases = {
            int(episodes * 0.25): "探索阶段",
            int(episodes * 0.50): "学习阶段", 
            int(episodes * 0.75): "优化阶段",
            int(episodes * 0.90): "收敛阶段"
        }
        
        for episode in range(episodes):
            # 重置环境
            state = solver.env.reset()
            episode_reward = 0
            step_count = 0
            max_steps = 50  # 限制最大步数
            
            # 当前episode的奖励分解记录
            episode_reward_breakdown = {
                'total_reward': 0.0,
                'step_rewards': [],
                'step_breakdowns': [],
                'pbrs_info': [],
                'final_success': False,
                'completion_rate': 0.0
            }
            
            while step_count < max_steps:
                # 选择动作
                state_tensor = solver._prepare_state_tensor(state)
                action = solver.select_action(state_tensor)
                
                # 执行动作
                next_state, reward, done, truncated, info = solver.env.step(action.item())
                
                episode_reward += reward
                step_count += 1
                
                # 记录当前步骤的详细奖励信息
                step_reward_info = {
                    'step': step_count,
                    'action': action.item(),
                    'reward': reward,
                    'base_reward': info.get('base_reward', 0.0),
                    'shaping_reward': info.get('shaping_reward', 0.0),
                    'reward_breakdown': info.get('reward_breakdown', {}),
                    'target_id': info.get('target_id', -1),
                    'uav_id': info.get('uav_id', -1),
                    'contribution': info.get('actual_contribution', 0.0),
                    'path_length': info.get('path_length', 0.0),
                    'done': done
                }
                
                episode_reward_breakdown['step_rewards'].append(reward)
                episode_reward_breakdown['step_breakdowns'].append(step_reward_info)
                
                # 记录PBRS信息
                if info.get('pbrs_enabled', False):
                    pbrs_info = {
                        'potential_before': info.get('potential_before', 0.0),
                        'potential_after': info.get('potential_after', 0.0),
                        'shaping_reward': info.get('shaping_reward', 0.0)
                    }
                    episode_reward_breakdown['pbrs_info'].append(pbrs_info)
                
                # 记录数据到分析器
                if episode % 20 == 0:  # 每20个episode记录一次详细数据
                    analyzer.log_step_data(episode, step_count, {
                        'action': action.item(),
                        'reward': reward,
                        'is_valid': not info.get('invalid_action', False),
                        'reward_breakdown': info.get('reward_breakdown', {})
                    })
                
                state = next_state
                
                if done or truncated:
                    break
            
            # 完善episode奖励分解记录
            episode_reward_breakdown['total_reward'] = episode_reward
            episode_reward_breakdown['final_success'] = episode_reward >= 1000
            
            # 计算完成率
            total_remaining = sum(np.sum(t.remaining_resources) for t in solver.env.targets)
            total_original = sum(np.sum(t.resources) for t in solver.env.targets)
            completion_rate = 1.0 - (total_remaining / (total_original + 1e-6))
            episode_reward_breakdown['completion_rate'] = completion_rate
            
            # 记录episode数据
            episode_rewards.append(episode_reward)
            completion_rates.append(completion_rate)
            detailed_reward_history.append(episode_reward_breakdown)
            
            # 检查是否成功（获得1000分奖励）
            if episode_reward >= 1000:
                success_episodes.append(episode)
            
            # 输出每轮的详细奖励变化（每10轮输出一次详细信息）
            if episode % 10 == 0 or episode_reward >= 1000:
                self._print_episode_reward_details(episode, episode_reward_breakdown, level_name)
            
            # 记录到分析器
            if episode % 20 == 0:
                analyzer.log_episode_data(episode, {
                    'total_reward': episode_reward,
                    'completion_rate': completion_rate,
                    'final_success': episode_reward >= 1000,
                    'step_count': step_count,
                    'detailed_breakdown': episode_reward_breakdown
                })
                
                analyzer.log_resource_allocation(episode, solver.env.uavs, solver.env.targets)
            
            # 训练网络并记录损失
            if len(solver.memory) > solver.batch_size:
                loss = solver.optimize_model()
                if loss is not None:
                    losses.append(loss)
            
            # 记录探索率
            exploration_rates.append(solver.epsilon)
            
            # 早停检查
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 更新进度条
            progress = (episode + 1) / episodes
            if progress - last_progress_update >= 0.02 or episode == episodes - 1:  # 每2%更新一次
                filled = int(50 * progress)
                bar = '█' * filled + '░' * (50 - filled)
                print(f"\r训练进度: [{bar}] {episode + 1}/{episodes} ({progress:.1%})", end='', flush=True)
                last_progress_update = progress
            
            # 训练阶段提示
            if (episode + 1) in training_phases:
                phase_name = training_phases[episode + 1]
                print(f"\n🎯 进入{phase_name} (Episode {episode + 1})")
            
            # 定期输出详细进度
            if (episode + 1) % 50 == 0:
                recent_rewards = episode_rewards[-50:]
                recent_success_rate = len([r for r in recent_rewards if r >= 1000]) / len(recent_rewards)
                recent_completion = np.mean(completion_rates[-50:])
                avg_loss = np.mean(losses[-50:]) if losses else 0
                
                # 计算训练趋势
                if len(episode_rewards) >= 100:
                    trend = "📈" if np.mean(episode_rewards[-50:]) > np.mean(episode_rewards[-100:-50]) else "📉"
                else:
                    trend = "📊"
                
                print(f"\n{trend} Episode {episode + 1}/{episodes}: "
                      f"平均奖励={np.mean(recent_rewards):.1f}, "
                      f"成功率={recent_success_rate:.2%}, "
                      f"完成率={recent_completion:.2%}, "
                      f"损失={avg_loss:.4f}, "
                      f"探索率={solver.epsilon:.3f}")
                
                # GPU内存信息（如果可用）
                if hasattr(solver, 'get_gpu_memory_info'):
                    print(f"   {solver.get_gpu_memory_info()}")
            
            # 早停检查
            if patience_counter >= patience and episode > episodes * 0.3:
                print(f"\n早停触发于第 {episode + 1} 轮 (连续{patience}轮无改进)")
                break
        
        training_time = time.time() - start_time
        
        # 计算最终性能指标
        final_success_rate = len(success_episodes) / episodes
        final_avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        final_completion_rate = np.mean(completion_rates[-50:]) if len(completion_rates) >= 50 else np.mean(completion_rates)
        
        performance = {
            'episodes': episode + 1,  # 实际训练轮次
            'training_time': training_time,
            'total_successes': len(success_episodes),
            'final_success_rate': final_success_rate,
            'final_avg_reward': final_avg_reward,
            'final_completion_rate': final_completion_rate,
            'episode_rewards': episode_rewards,
            'completion_rates': completion_rates,
            'losses': losses,
            'exploration_rates': exploration_rates,
            'best_reward': best_reward,
            'early_stopped': patience_counter >= patience,
            'detailed_reward_history': detailed_reward_history,  # 新增：详细奖励分解历史
            'convergence_data': {
                'reward_trend': np.polyfit(range(len(episode_rewards)), episode_rewards, 1)[0] if len(episode_rewards) > 1 else 0,
                'reward_stability': np.std(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.std(episode_rewards),
                'final_exploration_rate': solver.epsilon
            }
        }
        
        print(f"\n{level_name} 训练完成:")
        print(f"  训练时间: {training_time:.1f}秒")
        print(f"  成功次数: {len(success_episodes)}/{episodes}")
        print(f"  最终成功率: {final_success_rate:.2%}")
        print(f"  最终平均奖励: {final_avg_reward:.1f}")
        print(f"  最终完成率: {final_completion_rate:.2%}")
        
        # 生成级别分析报告和收敛曲线
        if episode % 20 == 0:  # 只有记录了数据才生成报告
            analyzer.generate_comprehensive_plots()
            analyzer.generate_detailed_report()
        
        # 生成详细奖励分析报告
        self._generate_reward_analysis_report(level_name, detailed_reward_history)
        
        # 保存训练收敛曲线
        self._save_convergence_curves(level_name, performance)
        
        # 保存训练历史数据（与main.py格式一致）
        training_data = {
            'episode_rewards': episode_rewards,
            'completion_rates': completion_rates,
            'losses': losses,
            'exploration_rates': exploration_rates,
            'training_time': training_time,
            'detailed_reward_history': detailed_reward_history,  # 新增：详细奖励分解历史
            'config': {
                'learning_rate': getattr(solver, 'learning_rate', self.config.LEARNING_RATE),
                'batch_size': getattr(solver, 'batch_size', self.config.BATCH_SIZE),
                'memory_size': len(solver.memory),
                'network_type': getattr(solver, 'network_type', 'ZeroShotGNN'),
                'episodes': episode + 1,
                'early_stopping': getattr(self.config, 'PATIENCE', 100),
                'target_update_freq': getattr(self.config, 'TARGET_UPDATE_FREQ', 20),
                'gamma': getattr(self.config, 'GAMMA', 0.99),
                'epsilon_start': getattr(self.config, 'EPSILON_START', 0.9),
                'epsilon_end': getattr(self.config, 'EPSILON_END', 0.1),
                'epsilon_decay': getattr(self.config, 'EPSILON_DECAY', 0.995),
                'use_prioritized_replay': getattr(self.config.training_config, 'use_prioritized_replay', True),
                'optimizer': 'AdamW' if getattr(solver, 'network_type', '') == 'ZeroShotGNN' else 'Adam'
            },
            'performance_metrics': {
                'final_success_rate': len([r for r in episode_rewards if r >= 1000]) / len(episode_rewards),
                'final_avg_reward': np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards),
                'final_completion_rate': np.mean(completion_rates[-50:]) if len(completion_rates) >= 50 else np.mean(completion_rates),
                'convergence_episode': self._find_convergence_point(episode_rewards),
                'training_stability': np.std(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.std(episode_rewards)
            }
        }
        
        # 保存为pickle文件（与main.py一致）
        history_path = os.path.join(self.output_dir, f"training_history_{level_name}.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"\n训练数据已保存: {history_path}")
        
        return performance
    
    def _find_convergence_point(self, rewards):
        """
        检测训练收敛点
        
        Args:
            rewards: 奖励序列
            
        Returns:
            int: 收敛的episode编号，如果未收敛返回-1
        """
        if len(rewards) < 100:
            return -1
        
        # 使用滑动窗口检测收敛
        window_size = 50
        threshold = 0.1  # 变化阈值
        
        for i in range(window_size, len(rewards) - window_size):
            # 计算前后窗口的平均值
            before_avg = np.mean(rewards[i-window_size:i])
            after_avg = np.mean(rewards[i:i+window_size])
            
            # 如果变化小于阈值，认为收敛
            if abs(after_avg - before_avg) / (abs(before_avg) + 1e-6) < threshold:
                return i
        
        return -1
    
    def _print_episode_reward_details(self, episode, reward_breakdown, level_name):
        """
        打印每轮的详细奖励分解信息
        
        Args:
            episode: 当前轮次
            reward_breakdown: 奖励分解字典
            level_name: 级别名称
        """
        print(f"\n📊 [{level_name}] Episode {episode + 1} 奖励详情:")
        print(f"   总奖励: {reward_breakdown['total_reward']:.2f}")
        print(f"   完成率: {reward_breakdown['completion_rate']:.2%}")
        print(f"   最终成功: {'✅' if reward_breakdown['final_success'] else '❌'}")
        print(f"   总步数: {len(reward_breakdown['step_rewards'])}")
        
        # 统计步骤奖励分布
        step_rewards = reward_breakdown['step_rewards']
        if step_rewards:
            print(f"   步骤奖励统计:")
            print(f"     - 平均步骤奖励: {np.mean(step_rewards):.2f}")
            print(f"     - 最大步骤奖励: {np.max(step_rewards):.2f}")
            print(f"     - 最小步骤奖励: {np.min(step_rewards):.2f}")
            print(f"     - 正奖励步数: {len([r for r in step_rewards if r > 0])}")
            print(f"     - 负奖励步数: {len([r for r in step_rewards if r < 0])}")
        
        # 显示关键步骤的奖励分解
        step_breakdowns = reward_breakdown['step_breakdowns']
        if step_breakdowns:
            # 找出奖励最高的步骤
            max_reward_step = max(step_breakdowns, key=lambda x: x['reward'])
            print(f"   🏆 最高奖励步骤 (Step {max_reward_step['step']}):")
            print(f"     - 奖励: {max_reward_step['reward']:.2f}")
            print(f"     - 基础奖励: {max_reward_step['base_reward']:.2f}")
            print(f"     - 塑形奖励: {max_reward_step['shaping_reward']:.2f}")
            print(f"     - 目标ID: {max_reward_step['target_id']}, UAV ID: {max_reward_step['uav_id']}")
            print(f"     - 贡献量: {max_reward_step['contribution']:.2f}")
            print(f"     - 路径长度: {max_reward_step['path_length']:.2f}")
            
            # 显示详细的奖励分解（如果有）
            if max_reward_step['reward_breakdown']:
                breakdown = max_reward_step['reward_breakdown']
                print(f"     - 详细分解:")
                
                # 处理不同类型的奖励分解
                if 'simple_breakdown' in breakdown:
                    # 简单奖励分解
                    simple_bd = breakdown['simple_breakdown']
                    for key, value in simple_bd.items():
                        if value != 0:
                            print(f"       * {key}: {value:.2f}")
                
                elif 'layer1_breakdown' in breakdown and 'layer2_breakdown' in breakdown:
                    # 双层奖励分解
                    print(f"       * 第一层奖励: {breakdown['layer1_total']:.2f}")
                    layer1_bd = breakdown['layer1_breakdown']
                    for key, value in layer1_bd.items():
                        if value != 0:
                            print(f"         - {key}: {value:.2f}")
                    
                    print(f"       * 第二层奖励: {breakdown['layer2_total']:.2f}")
                    layer2_bd = breakdown['layer2_breakdown']
                    for key, value in layer2_bd.items():
                        if value != 0:
                            print(f"         - {key}: {value:.2f}")
        
        # PBRS信息统计
        pbrs_info = reward_breakdown['pbrs_info']
        if pbrs_info:
            total_shaping = sum(info['shaping_reward'] for info in pbrs_info)
            avg_potential_change = np.mean([info['potential_after'] - info['potential_before'] for info in pbrs_info])
            print(f"   🔄 PBRS信息:")
            print(f"     - 总塑形奖励: {total_shaping:.2f}")
            print(f"     - 平均势能变化: {avg_potential_change:.2f}")
        
        print("   " + "="*50)
    
    def _generate_reward_analysis_report(self, level_name, detailed_reward_history):
        """
        生成详细的奖励分析报告
        
        Args:
            level_name: 级别名称
            detailed_reward_history: 详细奖励历史数据
        """
        if not detailed_reward_history:
            return
        
        print(f"\n📈 [{level_name}] 奖励分析报告:")
        
        # 统计总体奖励分布
        total_rewards = [episode['total_reward'] for episode in detailed_reward_history]
        completion_rates = [episode['completion_rate'] for episode in detailed_reward_history]
        success_episodes = [i for i, episode in enumerate(detailed_reward_history) if episode['final_success']]
        
        print(f"   总体统计:")
        print(f"     - 平均总奖励: {np.mean(total_rewards):.2f}")
        print(f"     - 最高总奖励: {np.max(total_rewards):.2f}")
        print(f"     - 最低总奖励: {np.min(total_rewards):.2f}")
        print(f"     - 奖励标准差: {np.std(total_rewards):.2f}")
        print(f"     - 成功轮次: {len(success_episodes)}/{len(detailed_reward_history)} ({len(success_episodes)/len(detailed_reward_history):.2%})")
        print(f"     - 平均完成率: {np.mean(completion_rates):.2%}")
        
        # 分析奖励组成
        if detailed_reward_history:
            # 统计PBRS使用情况
            pbrs_episodes = [episode for episode in detailed_reward_history if episode['pbrs_info']]
            if pbrs_episodes:
                total_shaping_rewards = []
                for episode in pbrs_episodes:
                    episode_shaping = sum(info['shaping_reward'] for info in episode['pbrs_info'])
                    total_shaping_rewards.append(episode_shaping)
                
                print(f"   PBRS统计:")
                print(f"     - 使用PBRS的轮次: {len(pbrs_episodes)}/{len(detailed_reward_history)}")
                print(f"     - 平均塑形奖励: {np.mean(total_shaping_rewards):.2f}")
                print(f"     - 塑形奖励范围: [{np.min(total_shaping_rewards):.2f}, {np.max(total_shaping_rewards):.2f}]")
            
            # 分析步骤奖励模式
            all_step_rewards = []
            for episode in detailed_reward_history:
                all_step_rewards.extend(episode['step_rewards'])
            
            if all_step_rewards:
                positive_rewards = [r for r in all_step_rewards if r > 0]
                negative_rewards = [r for r in all_step_rewards if r < 0]
                zero_rewards = [r for r in all_step_rewards if r == 0]
                
                print(f"   步骤奖励分析:")
                print(f"     - 总步数: {len(all_step_rewards)}")
                print(f"     - 正奖励步数: {len(positive_rewards)} ({len(positive_rewards)/len(all_step_rewards):.2%})")
                print(f"     - 负奖励步数: {len(negative_rewards)} ({len(negative_rewards)/len(all_step_rewards):.2%})")
                print(f"     - 零奖励步数: {len(zero_rewards)} ({len(zero_rewards)/len(all_step_rewards):.2%})")
                
                if positive_rewards:
                    print(f"     - 平均正奖励: {np.mean(positive_rewards):.2f}")
                if negative_rewards:
                    print(f"     - 平均负奖励: {np.mean(negative_rewards):.2f}")
        
        # 保存详细分析到文件
        report_path = os.path.join(self.output_dir, f"reward_analysis_{level_name}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"奖励分析报告 - {level_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write("总体统计:\n")
            f.write(f"  平均总奖励: {np.mean(total_rewards):.2f}\n")
            f.write(f"  最高总奖励: {np.max(total_rewards):.2f}\n")
            f.write(f"  最低总奖励: {np.min(total_rewards):.2f}\n")
            f.write(f"  奖励标准差: {np.std(total_rewards):.2f}\n")
            f.write(f"  成功轮次: {len(success_episodes)}/{len(detailed_reward_history)} ({len(success_episodes)/len(detailed_reward_history):.2%})\n")
            f.write(f"  平均完成率: {np.mean(completion_rates):.2%}\n\n")
            
            # 详细的每轮奖励分解
            f.write("详细轮次分解:\n")
            for i, episode in enumerate(detailed_reward_history):
                if i % 10 == 0 or episode['final_success']:  # 每10轮或成功轮次记录详细信息
                    f.write(f"\nEpisode {i+1}:\n")
                    f.write(f"  总奖励: {episode['total_reward']:.2f}\n")
                    f.write(f"  完成率: {episode['completion_rate']:.2%}\n")
                    f.write(f"  最终成功: {episode['final_success']}\n")
                    f.write(f"  步数: {len(episode['step_rewards'])}\n")
                    
                    if episode['step_rewards']:
                        f.write(f"  步骤奖励统计: 平均={np.mean(episode['step_rewards']):.2f}, "
                               f"最大={np.max(episode['step_rewards']):.2f}, "
                               f"最小={np.min(episode['step_rewards']):.2f}\n")
        
        print(f"   详细分析报告已保存: {report_path}")
    
    def _generate_final_report(self):
        """生成最终的课程学习报告"""
        print(f"\n生成最终课程学习报告...")
        
        # 设置字体 - 增强版本
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
        
        # 创建综合性能图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        levels = [level['level'] for level in self.training_history['levels']]
        level_names = [level['name'] for level in self.training_history['levels']]
        success_rates = [level['performance']['final_success_rate'] for level in self.training_history['levels']]
        avg_rewards = [level['performance']['final_avg_reward'] for level in self.training_history['levels']]
        completion_rates = [level['performance']['final_completion_rate'] for level in self.training_history['levels']]
        training_times = [level['performance']['training_time'] for level in self.training_history['levels']]
        
        # 1. 成功率变化
        ax1.bar(levels, success_rates, alpha=0.7, color='green')
        ax1.set_title('各级别最终成功率')
        ax1.set_xlabel('训练级别')
        ax1.set_ylabel('成功率')
        ax1.set_xticks(levels)
        ax1.set_xticklabels([f"L{i}" for i in levels])
        ax1.grid(True, alpha=0.3)
        
        # 2. 平均奖励变化
        ax2.plot(levels, avg_rewards, 'o-', color='blue', linewidth=2, markersize=8)
        ax2.set_title('各级别最终平均奖励')
        ax2.set_xlabel('训练级别')
        ax2.set_ylabel('平均奖励')
        ax2.set_xticks(levels)
        ax2.set_xticklabels([f"L{i}" for i in levels])
        ax2.grid(True, alpha=0.3)
        
        # 3. 完成率变化
        ax3.bar(levels, completion_rates, alpha=0.7, color='orange')
        ax3.set_title('各级别最终完成率')
        ax3.set_xlabel('训练级别')
        ax3.set_ylabel('完成率')
        ax3.set_xticks(levels)
        ax3.set_xticklabels([f"L{i}" for i in levels])
        ax3.grid(True, alpha=0.3)
        
        # 4. 训练时间
        ax4.bar(levels, training_times, alpha=0.7, color='red')
        ax4.set_title('各级别训练时间')
        ax4.set_xlabel('训练级别')
        ax4.set_ylabel('训练时间(秒)')
        ax4.set_xticks(levels)
        ax4.set_xticklabels([f"L{i}" for i in levels])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(self.output_dir, "curriculum_learning_summary.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成文本报告
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("课程学习训练总结报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"训练级别数: {len(self.training_history['levels'])}")
        report_lines.append("")
        
        # 各级别详细信息
        for level in self.training_history['levels']:
            perf = level['performance']
            report_lines.append(f"级别 {level['level']}: {level['name']}")
            report_lines.append(f"  场景描述: {level['description']}")
            report_lines.append(f"  训练轮次: {perf['episodes']}")
            report_lines.append(f"  训练时间: {perf['training_time']:.1f}秒")
            report_lines.append(f"  成功次数: {perf['total_successes']}")
            report_lines.append(f"  最终成功率: {perf['final_success_rate']:.2%}")
            report_lines.append(f"  最终平均奖励: {perf['final_avg_reward']:.1f}")
            report_lines.append(f"  最终完成率: {perf['final_completion_rate']:.2%}")
            report_lines.append("")
        
        # 总体分析
        report_lines.append("总体分析:")
        report_lines.append("-" * 40)
        
        total_time = sum(training_times)
        avg_success_rate = np.mean(success_rates)
        final_level_success = success_rates[-1] if success_rates else 0
        
        report_lines.append(f"总训练时间: {total_time:.1f}秒")
        report_lines.append(f"平均成功率: {avg_success_rate:.2%}")
        report_lines.append(f"最终级别成功率: {final_level_success:.2%}")
        
        if final_level_success >= 0.8:
            report_lines.append("✅ 课程学习成功！模型已具备处理复杂场景的能力")
        else:
            report_lines.append("⚠️ 课程学习需要改进，建议增加训练轮次或调整参数")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "curriculum_learning_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 保存训练历史
        history_path = os.path.join(self.output_dir, "training_history.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        print(f"最终报告已保存:")
        print(f"  图表: {chart_path}")
        print(f"  报告: {report_path}")
        print(f"  历史: {history_path}")
    def train_large_scale_curriculum(self, episodes_per_level=300, success_threshold=0.7):
        """
        执行大规模课程学习训练（20UAV-15Target）
        
        Args:
            episodes_per_level: 每个难度级别的训练轮次（大规模场景需要更多轮次）
            success_threshold: 进入下一级别的成功率阈值
        """
        print("=" * 80)
        print("开始大规模课程学习训练 (20UAV-15Target)")
        print("=" * 80)
        
        # 设置字体
        set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
        
        # 获取大规模课程场景 - 使用优化版本
        try:
            from large_curriculum_scenarios_optimized import get_large_curriculum_scenarios_optimized
            curriculum_scenarios = get_large_curriculum_scenarios_optimized()
            print("✅ 使用优化版本的大规模课程场景")
        except ImportError:
            from scenarios import get_large_curriculum_scenarios
            curriculum_scenarios = get_large_curriculum_scenarios()
            print("⚠️ 使用原版大规模课程场景")
        
        # 调整配置以适应大规模场景
        self.config.BATCH_SIZE = 32  # 增加批次大小
        self.config.MEMORY_SIZE = 20000  # 增加经验回放缓冲区
        self.config.TARGET_UPDATE = 200  # 调整目标网络更新频率
        
        print(f"大规模训练配置:")
        print(f"  批次大小: {self.config.BATCH_SIZE}")
        print(f"  经验缓冲区: {self.config.MEMORY_SIZE}")
        print(f"  每级别轮次: {episodes_per_level}")
        
        # 使用原有的训练逻辑，但传入大规模场景
        return self._train_curriculum_with_scenarios(
            curriculum_scenarios, episodes_per_level, success_threshold, "Large_Scale"
        )

    def _train_curriculum_with_scenarios(self, scenarios, episodes_per_level, success_threshold, prefix=""):
        """
        使用指定场景进行课程学习训练
        """
        solver = None
        
        for level_idx, (scenario_func, level_name, description) in enumerate(scenarios):
            print(f"\n{'='*60}")
            print(f"训练级别 {level_idx + 1}: {level_name}")
            print(f"场景描述: {description}")
            print(f"{'='*60}")
            
            # 获取当前级别的场景
            uavs, targets, obstacles = scenario_func(self.config.OBSTACLE_TOLERANCE)
            
            # 创建或更新求解器
            if solver is None:
                # 首次创建求解器
                graph = DirectedGraph(uavs, targets, len(uavs[0].resources), obstacles, self.config)
                i_dim = len(uavs) * len(targets) * len(uavs[0].resources)
                h_dim = [512, 256, 128]  # 大规模场景使用更深的网络
                o_dim = len(uavs) * len(targets) * len(uavs[0].resources)
                
                solver = GraphRLSolver(
                    uavs=uavs, targets=targets, graph=graph, obstacles=obstacles,
                    i_dim=i_dim, h_dim=h_dim, o_dim=o_dim, config=self.config,
                    obs_mode="graph", network_type="ZeroShotGNN"
                )
                print(f"大规模求解器初始化完成 (动作空间: {o_dim})")
            else:
                # 更新求解器的环境
                solver.env.uavs = uavs
                solver.env.targets = targets
                solver.env.obstacles = obstacles
                solver.env.graph = DirectedGraph(uavs, targets, len(uavs[0].resources), obstacles, self.config)
                print(f"求解器环境已更新")
            
            # 训练当前级别
            level_performance = self._train_level(
                solver, f"{prefix}_{level_name}" if prefix else level_name, 
                episodes_per_level, success_threshold
            )
            
            # 记录训练历史
            self.training_history['levels'].append({
                'level': level_idx + 1,
                'name': level_name,
                'description': description,
                'episodes': episodes_per_level,
                'performance': level_performance,
                'scale': prefix or "Standard"
            })
            
            # 检查成功阈值
            final_success_rate = level_performance.get('final_success_rate', 0)
            if final_success_rate < success_threshold:
                print(f"⚠️ 级别 {level_idx + 1} 未达到成功阈值 ({final_success_rate:.2%} < {success_threshold:.2%})")
            else:
                print(f"✅ 级别 {level_idx + 1} 训练成功 ({final_success_rate:.2%} >= {success_threshold:.2%})")
            
            # 保存模型
            model_path = os.path.join(self.output_dir, f"model_{prefix}_level_{level_idx + 1}_{level_name}.pth")
            torch.save(solver.policy_net.state_dict(), model_path)
            print(f"模型已保存: {model_path}")
        
        # 生成最终报告
        self._generate_final_report()
        
        return solver


def main():
    """主函数"""
    print("课程学习训练系统")
    print("=" * 50)
    
    # 初始化配置
    config = Config()
    
    # 创建训练器
    trainer = CurriculumLearningTrainer(config)
    
    # 开始课程学习训练
    # final_solver = trainer.train_curriculum(
    #     episodes_per_level=200,  # 每级别200轮
    #     success_threshold=0.6    # 60%成功率阈值
    # )
    final_solver = trainer.train_large_scale_curriculum(episodes_per_level=300, success_threshold=0.7)

    print("课程学习训练完成！")
    
    return final_solver

    def _save_convergence_curves(self, level_name, performance):
        """保存训练收敛曲线（与main.py格式一致）"""
        try:
            # 设置中文字体 - 增强版本
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            episodes = range(1, len(performance['episode_rewards']) + 1)
            
            # 1. 奖励收敛曲线
            ax1.plot(episodes, performance['episode_rewards'], alpha=0.6, linewidth=1, label='原始奖励')
            if len(performance['episode_rewards']) > 20:
                window = min(50, len(performance['episode_rewards']) // 5)
                moving_avg = np.convolve(performance['episode_rewards'], np.ones(window)/window, mode='valid')
                ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}轮移动平均')
            ax1.set_title(f'{level_name} - 奖励收敛曲线')
            ax1.set_xlabel('训练轮次')
            ax1.set_ylabel('奖励值')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 完成率曲线
            ax2.plot(episodes, performance['completion_rates'], 'g-', alpha=0.7, label='完成率')
            ax2.set_title(f'{level_name} - 任务完成率')
            ax2.set_xlabel('训练轮次')
            ax2.set_ylabel('完成率')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 损失曲线
            if performance['losses']:
                loss_episodes = np.linspace(1, len(episodes), len(performance['losses']))
                ax3.plot(loss_episodes, performance['losses'], 'orange', alpha=0.7, label='训练损失')
                ax3.set_title(f'{level_name} - 训练损失')
                ax3.set_xlabel('训练轮次')
                ax3.set_ylabel('损失值')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, '无损失数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f'{level_name} - 训练损失')
            
            # 4. 探索率衰减
            if performance['exploration_rates']:
                ax4.plot(episodes, performance['exploration_rates'], 'm-', alpha=0.7, label='探索率')
                ax4.set_title(f'{level_name} - 探索率衰减')
                ax4.set_xlabel('训练轮次')
                ax4.set_ylabel('探索率')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '无探索率数据', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title(f'{level_name} - 探索率衰减')
            
            plt.tight_layout()
            
            # 保存图表
            curve_path = os.path.join(self.output_dir, f"convergence_curves_{level_name}.png")
            plt.savefig(curve_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"收敛曲线已保存: {curve_path}")
            
        except Exception as e:
            print(f"保存收敛曲线时出错: {e}")


def main():
    """主函数"""
    print("课程学习训练系统")
    print("=" * 50)
    
    # 初始化配置
    config = Config()
    
    # 创建训练器
    trainer = CurriculumLearningTrainer(config)
    
    # 开始课程学习训练
    # final_solver = trainer.train_curriculum(
    #     episodes_per_level=200,  # 每级别200轮
    #     success_threshold=0.6    # 60%成功率阈值
    # )
    final_solver = trainer.train_large_scale_curriculum(episodes_per_level=300, success_threshold=0.7)

    print("课程学习训练完成！")
    
    return final_solver


if __name__ == "__main__":
    main()