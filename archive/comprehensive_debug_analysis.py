#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合调试分析脚本 - 专门检查奖励收敛和资源分配问题
重点分析：无人机有资源、目标有需求，但未参与任务分配的情况
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import os
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns
from pathlib import Path

# 导入字体设置函数
from main import set_chinese_font

# 设置中文字体，避免字体警告
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 抑制matplotlib字体警告
import logging
import warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 彻底解决字体警告问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 关键：禁用Unicode减号
plt.rcParams['font.family'] = 'sans-serif'

# 调用字体设置函数
set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')

class ComprehensiveDebugAnalyzer:
    """综合调试分析器 - 深度分析训练过程中的各种问题"""
    
    def __init__(self, output_dir="debug_analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.episode_data = []
        self.step_data = []
        self.resource_allocation_data = []
        self.reward_breakdown_data = []
        self.convergence_data = []
        
        # 问题检测计数器
        self.issue_counters = {
            'unused_resources': 0,
            'unmet_demands': 0,
            'inefficient_allocation': 0,
            'reward_anomalies': 0,
            'convergence_issues': 0
        }
        
        # 分析结果存储
        self.analysis_results = {}
        
    def log_episode_data(self, episode: int, episode_info: Dict[str, Any]):
        """记录每个episode的详细数据"""
        episode_record = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'total_reward': episode_info.get('total_reward', 0),
            'completion_rate': episode_info.get('completion_rate', 0),
            'resource_utilization': episode_info.get('resource_utilization', 0),
            'num_actions': episode_info.get('num_actions', 0),
            'invalid_actions': episode_info.get('invalid_actions', 0),
            'final_success': episode_info.get('final_success', False),
            'synergy_attacks': episode_info.get('synergy_attacks', 0),
            'exploration_rate': episode_info.get('exploration_rate', 0),
            'learning_rate': episode_info.get('learning_rate', 0),
            'loss': episode_info.get('loss', 0)
        }
        
        self.episode_data.append(episode_record)
        
        # 实时问题检测
        self._detect_episode_issues(episode_record)
    
    def log_step_data(self, episode: int, step: int, step_info: Dict[str, Any]):
        """记录每个step的详细数据"""
        step_record = {
            'episode': episode,
            'step': step,
            'action': step_info.get('action', None),
            'reward': step_info.get('reward', 0),
            'state_before': step_info.get('state_before', None),
            'state_after': step_info.get('state_after', None),
            'q_values': step_info.get('q_values', None),
            'target_id': step_info.get('target_id', None),
            'uav_id': step_info.get('uav_id', None),
            'contribution': step_info.get('contribution', 0),
            'path_length': step_info.get('path_length', 0),
            'is_valid': step_info.get('is_valid', True)
        }
        
        self.step_data.append(step_record)
    
    def log_resource_allocation(self, episode: int, uavs: List, targets: List):
        """记录资源分配状态"""
        allocation_record = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'uav_states': [],
            'target_states': [],
            'allocation_matrix': [],
            'efficiency_metrics': {}
        }
        
        # 记录UAV状态
        for uav in uavs:
            uav_state = {
                'id': uav.id,
                'position': list(uav.current_position),
                'resources': list(uav.resources),
                'max_resources': getattr(uav, 'initial_resources', list(uav.resources)),
                'is_active': np.any(uav.resources > 0),
                'task_count': len(getattr(uav, 'task_sequence', [])),
                'utilization_rate': self._calculate_uav_utilization(uav)
            }
            allocation_record['uav_states'].append(uav_state)
        
        # 记录目标状态
        for target in targets:
            target_state = {
                'id': target.id,
                'position': list(target.position),
                'resources': list(target.resources),
                'remaining_resources': list(target.remaining_resources),
                'allocated_uavs': list(getattr(target, 'allocated_uavs', [])),
                'is_satisfied': np.all(target.remaining_resources <= 0),
                'satisfaction_rate': self._calculate_target_satisfaction(target)
            }
            allocation_record['target_states'].append(target_state)
        
        # 计算分配矩阵
        allocation_record['allocation_matrix'] = self._build_allocation_matrix(uavs, targets)
        
        # 计算效率指标
        allocation_record['efficiency_metrics'] = self._calculate_efficiency_metrics(uavs, targets)
        
        self.resource_allocation_data.append(allocation_record)
        
        # 检测资源分配问题
        self._detect_allocation_issues(allocation_record)
    
    def _calculate_uav_utilization(self, uav) -> float:
        """计算UAV资源利用率"""
        initial_resources = getattr(uav, 'initial_resources', uav.resources)
        if np.sum(initial_resources) == 0:
            return 0.0
        
        used_resources = np.array(initial_resources) - np.array(uav.resources)
        return np.sum(used_resources) / np.sum(initial_resources)
    
    def _calculate_target_satisfaction(self, target) -> float:
        """计算目标满足率"""
        if np.sum(target.resources) == 0:
            return 1.0
        
        satisfied_resources = np.array(target.resources) - np.array(target.remaining_resources)
        return np.sum(satisfied_resources) / np.sum(target.resources)
    
    def _build_allocation_matrix(self, uavs: List, targets: List) -> List[List[float]]:
        """构建分配矩阵"""
        matrix = []
        for uav in uavs:
            uav_row = []
            for target in targets:
                # 检查UAV是否分配给了这个目标
                allocation_score = 0.0
                if hasattr(target, 'allocated_uavs'):
                    for uav_id, phi_idx in target.allocated_uavs:
                        if uav_id == uav.id:
                            allocation_score = 1.0
                            break
                uav_row.append(allocation_score)
            matrix.append(uav_row)
        return matrix
    
    def _calculate_efficiency_metrics(self, uavs: List, targets: List) -> Dict[str, float]:
        """计算效率指标"""
        metrics = {}
        
        # 总体资源利用率
        total_initial_resources = sum(np.sum(getattr(uav, 'initial_resources', uav.resources)) for uav in uavs)
        total_remaining_resources = sum(np.sum(uav.resources) for uav in uavs)
        metrics['overall_utilization'] = (total_initial_resources - total_remaining_resources) / max(total_initial_resources, 1e-6)
        
        # 目标完成率
        completed_targets = sum(1 for target in targets if np.all(target.remaining_resources <= 0))
        metrics['completion_rate'] = completed_targets / len(targets)
        
        # 资源匹配效率
        total_demand = sum(np.sum(target.remaining_resources) for target in targets)
        total_supply = sum(np.sum(uav.resources) for uav in uavs)
        metrics['supply_demand_ratio'] = total_supply / max(total_demand, 1e-6)
        
        # 负载均衡度
        uav_utilizations = [self._calculate_uav_utilization(uav) for uav in uavs]
        if len(uav_utilizations) > 1:
            metrics['load_balance'] = 1.0 - np.std(uav_utilizations) / max(np.mean(uav_utilizations), 1e-6)
        else:
            metrics['load_balance'] = 1.0
        
        return metrics
    
    def _detect_episode_issues(self, episode_record: Dict[str, Any]):
        """检测episode级别的问题"""
        # 检测奖励异常
        if abs(episode_record['total_reward']) > 10000 or np.isnan(episode_record['total_reward']):
            self.issue_counters['reward_anomalies'] += 1
        
        # 检测收敛问题
        if len(self.episode_data) > 50:
            recent_rewards = [ep['total_reward'] for ep in self.episode_data[-50:]]
            if np.std(recent_rewards) > np.mean(recent_rewards) * 2:
                self.issue_counters['convergence_issues'] += 1
    
    def _detect_allocation_issues(self, allocation_record: Dict[str, Any]):
        """检测资源分配问题"""
        # 检测未使用的资源
        active_uavs = [uav for uav in allocation_record['uav_states'] if uav['is_active']]
        unallocated_uavs = [uav for uav in active_uavs if uav['task_count'] == 0]
        
        if unallocated_uavs:
            self.issue_counters['unused_resources'] += len(unallocated_uavs)
        
        # 检测未满足的需求
        unsatisfied_targets = [target for target in allocation_record['target_states'] 
                             if not target['is_satisfied'] and target['satisfaction_rate'] < 0.1]
        
        if unsatisfied_targets and active_uavs:
            self.issue_counters['unmet_demands'] += len(unsatisfied_targets)
        
        # 检测低效分配
        if allocation_record['efficiency_metrics']['overall_utilization'] < 0.3:
            self.issue_counters['inefficient_allocation'] += 1
    
    def analyze_reward_convergence(self) -> Dict[str, Any]:
        """分析奖励收敛情况"""
        if not self.episode_data:
            return {"error": "没有episode数据"}
        
        rewards = [ep['total_reward'] for ep in self.episode_data]
        episodes = [ep['episode'] for ep in self.episode_data]
        
        analysis = {
            'total_episodes': len(rewards),
            'reward_statistics': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards)
            },
            'convergence_metrics': {},
            'trend_analysis': {},
            'stability_analysis': {}
        }
        
        # 收敛性分析 - 修复版本
        if len(rewards) > 10:  # 降低阈值，确保有足够数据进行分析
            # 计算移动平均
            window_size = min(20, max(5, len(rewards) // 4))  # 确保窗口大小合理
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            
            # 趋势分析 - 总是执行
            if len(moving_avg) > 2:  # 只需要至少3个点就能计算趋势
                trend_slope = np.polyfit(range(len(moving_avg)), moving_avg, 1)[0]
                analysis['trend_analysis'] = {
                    'slope': trend_slope,
                    'is_improving': trend_slope > 0,
                    'improvement_rate': trend_slope * 100  # 每100轮的改进量
                }
            else:
                # 如果移动平均点太少，直接用原始数据
                trend_slope = np.polyfit(range(len(rewards)), rewards, 1)[0]
                analysis['trend_analysis'] = {
                    'slope': trend_slope,
                    'is_improving': trend_slope > 0,
                    'improvement_rate': trend_slope * 100
                }
            
            # 稳定性分析
            if len(rewards) > 20:
                recent_size = min(50, len(rewards) // 2)
                early_size = min(50, len(rewards) // 2)
                recent_rewards = rewards[-recent_size:]
                early_rewards = rewards[:early_size]
            else:
                # 数据较少时，简单分为前后两半
                mid = len(rewards) // 2
                recent_rewards = rewards[mid:]
                early_rewards = rewards[:mid] if mid > 0 else rewards
            
            analysis['stability_analysis'] = {
                'recent_std': np.std(recent_rewards),
                'early_std': np.std(early_rewards),
                'stability_improvement': np.std(early_rewards) - np.std(recent_rewards),
                'coefficient_of_variation': np.std(recent_rewards) / max(abs(np.mean(recent_rewards)), 1e-6)
            }
        else:
            # 数据太少时的默认值
            analysis['trend_analysis'] = {
                'slope': 0.0,
                'is_improving': False,
                'improvement_rate': 0.0
            }
            analysis['stability_analysis'] = {
                'recent_std': np.std(rewards),
                'early_std': np.std(rewards),
                'stability_improvement': 0.0,
                'coefficient_of_variation': np.std(rewards) / max(abs(np.mean(rewards)), 1e-6)
            }
        
        # 1000分奖励分析
        high_reward_episodes = [ep for ep in self.episode_data if ep.get('final_success', False)]
        analysis['high_reward_analysis'] = {
            'count': len(high_reward_episodes),
            'rate': len(high_reward_episodes) / len(self.episode_data),
            'episodes': [ep['episode'] for ep in high_reward_episodes[:10]]  # 前10个
        }
        
        return analysis
    
    def analyze_resource_allocation_patterns(self) -> Dict[str, Any]:
        """分析资源分配模式"""
        if not self.resource_allocation_data:
            return {"error": "没有资源分配数据"}
        
        analysis = {
            'allocation_efficiency': {},
            'resource_utilization_trends': {},
            'problematic_patterns': {},
            'recommendations': []
        }
        
        # 效率趋势分析
        efficiency_metrics = [record['efficiency_metrics'] for record in self.resource_allocation_data]
        
        if efficiency_metrics:
            for metric_name in efficiency_metrics[0].keys():
                values = [metrics[metric_name] for metrics in efficiency_metrics]
                analysis['allocation_efficiency'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0,
                    'latest': values[-1] if values else 0
                }
        
        # 问题模式检测
        unused_resource_episodes = []
        unmet_demand_episodes = []
        
        for record in self.resource_allocation_data:
            episode = record['episode']
            
            # 检测有资源但未分配的情况
            active_uavs = [uav for uav in record['uav_states'] if uav['is_active']]
            unallocated_uavs = [uav for uav in active_uavs if uav['task_count'] == 0]
            unsatisfied_targets = [target for target in record['target_states'] 
                                 if not target['is_satisfied']]
            
            if unallocated_uavs and unsatisfied_targets:
                unused_resource_episodes.append({
                    'episode': episode,
                    'unused_uavs': len(unallocated_uavs),
                    'unmet_targets': len(unsatisfied_targets),
                    'efficiency': record['efficiency_metrics']['overall_utilization']
                })
        
        analysis['problematic_patterns'] = {
            'unused_resources_with_unmet_demands': {
                'count': len(unused_resource_episodes),
                'rate': len(unused_resource_episodes) / len(self.resource_allocation_data),
                'examples': unused_resource_episodes[:5]  # 前5个例子
            }
        }
        
        # 生成建议
        if len(unused_resource_episodes) > len(self.resource_allocation_data) * 0.3:
            analysis['recommendations'].append("检测到大量未使用资源与未满足需求并存的情况，建议优化动作选择策略")
        
        if analysis['allocation_efficiency'].get('overall_utilization', {}).get('mean', 0) < 0.5:
            analysis['recommendations'].append("整体资源利用率较低，建议增加资源贡献奖励权重")
        
        return analysis
    
    def generate_comprehensive_plots(self):
        """生成综合分析图表"""
        if not self.episode_data:
            print("没有数据可供绘图")
            return
        
        # 确保字体设置生效
        set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
        
        # 创建大图表
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 奖励收敛曲线
        ax1 = plt.subplot(3, 3, 1)
        episodes = [ep['episode'] for ep in self.episode_data]
        rewards = [ep['total_reward'] for ep in self.episode_data]
        
        plt.plot(episodes, rewards, alpha=0.6, linewidth=1, label='原始奖励')
        
        # 添加移动平均
        if len(rewards) > 20:
            window = min(50, len(rewards) // 5)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}轮移动平均')
        
        plt.title('奖励收敛曲线')
        plt.xlabel('Episode')
        plt.ylabel('总奖励')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 1000分奖励获得情况
        ax2 = plt.subplot(3, 3, 2)
        final_success_episodes = [ep['episode'] for ep in self.episode_data if ep.get('final_success', False)]
        final_success_rewards = [1000] * len(final_success_episodes)
        
        if final_success_episodes:
            plt.scatter(final_success_episodes, final_success_rewards, color='gold', s=50, alpha=0.8)
            plt.title(f'1000分奖励获得情况 (共{len(final_success_episodes)}次)')
        else:
            plt.text(0.5, 0.5, '未获得1000分奖励', ha='center', va='center', transform=ax2.transAxes)
            plt.title('1000分奖励获得情况 (0次)')
        
        plt.xlabel('Episode')
        plt.ylabel('奖励值')
        plt.grid(True, alpha=0.3)
        
        # 3. 完成率趋势
        ax3 = plt.subplot(3, 3, 3)
        completion_rates = [ep.get('completion_rate', 0) for ep in self.episode_data]
        plt.plot(episodes, completion_rates, 'g-', alpha=0.7)
        plt.title('任务完成率趋势')
        plt.xlabel('Episode')
        plt.ylabel('完成率')
        plt.grid(True, alpha=0.3)
        
        # 4. 资源利用率分析
        if self.resource_allocation_data:
            ax4 = plt.subplot(3, 3, 4)
            allocation_episodes = [record['episode'] for record in self.resource_allocation_data]
            utilization_rates = [record['efficiency_metrics']['overall_utilization'] 
                               for record in self.resource_allocation_data]
            
            plt.plot(allocation_episodes, utilization_rates, 'b-', alpha=0.7)
            plt.title('资源利用率趋势')
            plt.xlabel('Episode')
            plt.ylabel('利用率')
            plt.grid(True, alpha=0.3)
        
        # 5. 探索率衰减
        ax5 = plt.subplot(3, 3, 5)
        exploration_rates = [ep.get('exploration_rate', 0) for ep in self.episode_data]
        if any(rate > 0 for rate in exploration_rates):
            plt.plot(episodes, exploration_rates, 'm-', alpha=0.7)
            plt.title('探索率衰减')
            plt.xlabel('Episode')
            plt.ylabel('探索率')
            plt.grid(True, alpha=0.3)
        
        # 6. 学习率变化
        ax6 = plt.subplot(3, 3, 6)
        learning_rates = [ep.get('learning_rate', 0) for ep in self.episode_data]
        if any(rate > 0 for rate in learning_rates):
            plt.semilogy(episodes, learning_rates, 'c-', alpha=0.7)
            plt.title('学习率变化')
            plt.xlabel('Episode')
            plt.ylabel('学习率 (log scale)')
            plt.grid(True, alpha=0.3)
        
        # 7. 损失函数变化
        ax7 = plt.subplot(3, 3, 7)
        losses = [ep.get('loss', 0) for ep in self.episode_data if ep.get('loss', 0) > 0]
        loss_episodes = [ep['episode'] for ep in self.episode_data if ep.get('loss', 0) > 0]
        if losses:
            plt.semilogy(loss_episodes, losses, 'orange', alpha=0.7)
            plt.title('训练损失变化')
            plt.xlabel('Episode')
            plt.ylabel('损失值 (log scale)')
            plt.grid(True, alpha=0.3)
        
        # 8. 问题统计
        ax8 = plt.subplot(3, 3, 8)
        issue_names = list(self.issue_counters.keys())
        issue_counts = list(self.issue_counters.values())
        
        bars = plt.bar(range(len(issue_names)), issue_counts, alpha=0.7)
        plt.title('检测到的问题统计')
        plt.xlabel('问题类型')
        plt.ylabel('次数')
        plt.xticks(range(len(issue_names)), [name.replace('_', '\n') for name in issue_names], rotation=45)
        
        # 为每个柱子添加数值标签
        for bar, count in zip(bars, issue_counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
        
        # 9. 协同攻击统计
        ax9 = plt.subplot(3, 3, 9)
        synergy_attacks = [ep.get('synergy_attacks', 0) for ep in self.episode_data]
        if any(attacks > 0 for attacks in synergy_attacks):
            plt.plot(episodes, synergy_attacks, 'purple', alpha=0.7, marker='o', markersize=3)
            plt.title('协同攻击次数')
            plt.xlabel('Episode')
            plt.ylabel('协同攻击次数')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '无协同攻击记录', ha='center', va='center', transform=ax9.transAxes)
            plt.title('协同攻击次数 (0次)')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.output_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"综合分析图表已保存至: {plot_path}")
        
        return plot_path
    
    def generate_detailed_report(self) -> str:
        """生成详细的分析报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("综合调试分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"分析数据: {len(self.episode_data)} episodes, {len(self.step_data)} steps")
        report_lines.append("")
        
        # 1. 奖励收敛分析
        reward_analysis = self.analyze_reward_convergence()
        report_lines.append("1. 奖励收敛分析")
        report_lines.append("-" * 40)
        
        if 'error' not in reward_analysis:
            stats = reward_analysis['reward_statistics']
            report_lines.append(f"   总episode数: {reward_analysis['total_episodes']}")
            report_lines.append(f"   奖励统计: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}")
            report_lines.append(f"   奖励范围: [{stats['min']:.2f}, {stats['max']:.2f}]")
            
            high_reward = reward_analysis['high_reward_analysis']
            report_lines.append(f"   1000分奖励: {high_reward['count']}次 ({high_reward['rate']:.1%})")
            
            if 'trend_analysis' in reward_analysis:
                trend = reward_analysis['trend_analysis']
                trend_desc = "上升" if trend['is_improving'] else "下降"
                report_lines.append(f"   收敛趋势: {trend_desc} (斜率={trend['slope']:.4f})")
            
            if 'stability_analysis' in reward_analysis:
                stability = reward_analysis['stability_analysis']
                report_lines.append(f"   稳定性: 变异系数={stability['coefficient_of_variation']:.3f}")
        
        report_lines.append("")
        
        # 2. 资源分配分析
        allocation_analysis = self.analyze_resource_allocation_patterns()
        report_lines.append("2. 资源分配分析")
        report_lines.append("-" * 40)
        
        if 'error' not in allocation_analysis:
            if 'allocation_efficiency' in allocation_analysis:
                efficiency = allocation_analysis['allocation_efficiency']
                if 'overall_utilization' in efficiency:
                    util = efficiency['overall_utilization']
                    report_lines.append(f"   整体资源利用率: {util['mean']:.3f} ± {util['std']:.3f}")
                
                if 'completion_rate' in efficiency:
                    comp = efficiency['completion_rate']
                    report_lines.append(f"   任务完成率: {comp['mean']:.3f} ± {comp['std']:.3f}")
            
            # 问题模式
            if 'problematic_patterns' in allocation_analysis:
                patterns = allocation_analysis['problematic_patterns']
                unused_pattern = patterns.get('unused_resources_with_unmet_demands', {})
                if unused_pattern.get('count', 0) > 0:
                    report_lines.append(f"   ⚠️  资源浪费问题: {unused_pattern['count']}次 ({unused_pattern['rate']:.1%})")
                    report_lines.append("      -> 存在有资源的UAV未分配，同时有目标未满足的情况")
            
            # 建议
            recommendations = allocation_analysis.get('recommendations', [])
            if recommendations:
                report_lines.append("   建议:")
                for rec in recommendations:
                    report_lines.append(f"      • {rec}")
        
        report_lines.append("")
        
        # 3. 问题统计
        report_lines.append("3. 问题检测统计")
        report_lines.append("-" * 40)
        
        total_issues = sum(self.issue_counters.values())
        if total_issues > 0:
            for issue_type, count in self.issue_counters.items():
                if count > 0:
                    percentage = count / len(self.episode_data) * 100 if self.episode_data else 0
                    report_lines.append(f"   {issue_type.replace('_', ' ').title()}: {count}次 ({percentage:.1f}%)")
        else:
            report_lines.append("   未检测到明显问题")
        
        report_lines.append("")
        
        # 4. 关键发现和建议
        report_lines.append("4. 关键发现和建议")
        report_lines.append("-" * 40)
        
        # 基于分析结果生成建议
        if 'error' not in reward_analysis:
            high_reward_rate = reward_analysis['high_reward_analysis']['rate']
            if high_reward_rate < 0.1:
                report_lines.append("   🔍 发现: 1000分奖励获得率过低")
                report_lines.append("      建议: 1) 增加成功奖励权重 2) 优化探索策略 3) 检查奖励函数设计")
            
            if reward_analysis['reward_statistics']['std'] > reward_analysis['reward_statistics']['mean']:
                report_lines.append("   🔍 发现: 奖励方差过大，训练不稳定")
                report_lines.append("      建议: 1) 启用奖励标准化 2) 调整学习率 3) 增加梯度裁剪")
        
        if self.issue_counters['unused_resources'] > len(self.episode_data) * 0.2:
            report_lines.append("   🔍 发现: 大量资源未被有效利用")
            report_lines.append("      建议: 1) 优化动作选择策略 2) 增加资源利用奖励 3) 改进状态表示")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"详细分析报告已保存至: {report_path}")
        
        return report_content
    
    def export_data(self):
        """导出所有分析数据"""
        export_data = {
            'episode_data': self.episode_data,
            'step_data': self.step_data[-1000:],  # 只保存最近1000步
            'resource_allocation_data': self.resource_allocation_data,
            'issue_counters': self.issue_counters,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # 保存为JSON
        json_path = self.output_dir / f"debug_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存为pickle
        pickle_path = self.output_dir / f"debug_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(export_data, f)
        
        print(f"调试数据已导出至:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pickle_path}")
        
        return json_path, pickle_path

def integrate_with_training_example():
    """展示如何集成到训练循环中的示例"""
    example_code = '''
# 在训练开始前初始化分析器
analyzer = ComprehensiveDebugAnalyzer("debug_analysis_output")

# 在训练循环中
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    episode_info = {
        'synergy_attacks': 0,
        'invalid_actions': 0,
        'num_actions': 0
    }
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        # 记录step数据
        analyzer.log_step_data(episode, step, {
            'action': action,
            'reward': reward,
            'state_before': state,
            'state_after': next_state,
            'target_id': info.get('target_id'),
            'uav_id': info.get('uav_id'),
            'contribution': info.get('actual_contribution', 0),
            'path_length': info.get('path_length', 0),
            'is_valid': not info.get('invalid_action', False)
        })
        
        episode_reward += reward
        episode_info['num_actions'] += 1
        if info.get('invalid_action'):
            episode_info['invalid_actions'] += 1
        if 'synergy_attacks' in info:
            episode_info['synergy_attacks'] += 1
        
        state = next_state
        step += 1
    
    # 记录episode数据
    episode_info.update({
        'total_reward': episode_reward,
        'completion_rate': calculate_completion_rate(env),
        'resource_utilization': calculate_resource_utilization(env),
        'final_success': episode_reward > 900,
        'exploration_rate': agent.epsilon,
        'learning_rate': agent.optimizer.param_groups[0]['lr'],
        'loss': last_loss
    })
    
    analyzer.log_episode_data(episode, episode_info)
    
    # 记录资源分配状态
    analyzer.log_resource_allocation(episode, env.uavs, env.targets)
    
    # 定期生成分析报告
    if episode % 100 == 0 and episode > 0:
        analyzer.generate_comprehensive_plots()
        analyzer.generate_detailed_report()
        analyzer.export_data()

# 训练结束后生成最终报告
analyzer.generate_comprehensive_plots()
final_report = analyzer.generate_detailed_report()
analyzer.export_data()

print("\\n" + "="*60)
print("最终分析报告:")
print("="*60)
print(final_report)
'''
    
    return example_code

if __name__ == "__main__":
    print("=" * 60)
    print("综合调试分析器")
    print("=" * 60)
    
    # 创建示例分析器
    analyzer = ComprehensiveDebugAnalyzer()
    
    # 生成一些示例数据进行演示
    print("生成示例数据进行演示...")
    
    np.random.seed(42)
    for episode in range(200):
        # 模拟episode数据
        base_reward = np.random.normal(150, 80)
        
        # 模拟1000分奖励（10%概率）
        final_success = False
        if episode > 50 and np.random.random() < 0.1:
            base_reward = 1000
            final_success = True
        
        episode_info = {
            'total_reward': base_reward,
            'completion_rate': np.random.beta(2, 3),
            'resource_utilization': np.random.beta(3, 2),
            'num_actions': np.random.randint(10, 50),
            'invalid_actions': np.random.randint(0, 5),
            'final_success': final_success,
            'synergy_attacks': np.random.randint(0, 3) if final_success else 0,
            'exploration_rate': 0.9 * (0.995 ** episode),
            'learning_rate': 1e-4 * (0.99 ** (episode // 10)),
            'loss': np.random.exponential(0.01)
        }
        
        analyzer.log_episode_data(episode, episode_info)
        
        # 模拟资源分配数据
        if episode % 10 == 0:
            # 创建模拟的UAV和目标对象
            class MockUAV:
                def __init__(self, uav_id):
                    self.id = uav_id
                    self.current_position = np.random.rand(2) * 1000
                    self.resources = np.random.rand(2) * 100
                    self.initial_resources = self.resources + np.random.rand(2) * 50
                    self.task_sequence = []
                    if np.random.random() < 0.7:  # 70%概率有任务
                        self.task_sequence = list(range(np.random.randint(1, 4)))
            
            class MockTarget:
                def __init__(self, target_id):
                    self.id = target_id
                    self.position = np.random.rand(2) * 1000
                    self.resources = np.random.rand(2) * 100 + 50
                    self.remaining_resources = self.resources * np.random.rand(2)
                    self.allocated_uavs = []
                    if np.random.random() < 0.6:  # 60%概率有分配
                        self.allocated_uavs = [(np.random.randint(0, 4), np.random.randint(0, 6))]
            
            mock_uavs = [MockUAV(i) for i in range(4)]
            mock_targets = [MockTarget(i) for i in range(2)]
            
            analyzer.log_resource_allocation(episode, mock_uavs, mock_targets)
    
    # 生成分析结果
    print("\n生成综合分析...")
    analyzer.generate_comprehensive_plots()
    report = analyzer.generate_detailed_report()
    analyzer.export_data()
    
    print("\n" + "="*60)
    print("示例分析报告:")
    print("="*60)
    print(report)
    
    print("\n" + "="*60)
    print("集成示例:")
    print("="*60)
    print(integrate_with_training_example())
