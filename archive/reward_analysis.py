#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励构成分析工具
分析训练过程中各个奖励组件的作用和效果
"""

import numpy as np
import matplotlib.pyplot as plt
from main import GraphRLSolver, set_chinese_font
from config import Config
from scenarios import get_small_scenario
from environment import DirectedGraph
from comprehensive_debug_analysis import ComprehensiveDebugAnalyzer

class RewardAnalyzer:
    """奖励分析器"""
    
    def __init__(self):
        self.reward_components = []
        self.waste_penalties = []
        self.episode_data = []
        
    def analyze_reward_components(self, episodes=50):
        """分析奖励组件的构成和效果"""
        print("=" * 60)
        print("奖励构成分析")
        print("=" * 60)
        
        # 初始化环境
        config = Config()
        config.ENABLE_REWARD_DEBUG = True
        
        uavs, targets, obstacles = get_small_scenario(config.OBSTACLE_TOLERANCE)
        graph = DirectedGraph(uavs, targets, len(uavs[0].resources), obstacles, config)
        
        # 计算维度
        i_dim = len(uavs) * len(targets) * len(uavs[0].resources)
        h_dim = [256, 128]
        o_dim = len(uavs) * len(targets) * len(uavs[0].resources)
        
        solver = GraphRLSolver(
            uavs=uavs, targets=targets, graph=graph, obstacles=obstacles,
            i_dim=i_dim, h_dim=h_dim, o_dim=o_dim, config=config,
            obs_mode="graph", network_type="ZeroShotGNN"
        )
        
        # 收集奖励数据
        total_waste_penalty = 0
        total_completion_reward = 0
        total_resource_reward = 0
        total_progress_reward = 0
        total_path_penalty = 0
        
        idle_uav_counts = []
        unmet_target_counts = []
        
        for episode in range(episodes):
            state = solver.env.reset()
            episode_waste_penalty = 0
            episode_completion_reward = 0
            episode_resource_reward = 0
            episode_progress_reward = 0
            episode_path_penalty = 0
            
            step_count = 0
            while step_count < 30:  # 限制步数
                state_tensor = solver._prepare_state_tensor(state)
                action = solver.select_action(state_tensor)
                next_state, reward, done, truncated, info = solver.env.step(action.item())
                
                # 分析奖励组件（需要修改环境以返回详细信息）
                if hasattr(solver.env, '_last_reward_breakdown'):
                    breakdown = solver.env._last_reward_breakdown
                    episode_completion_reward += breakdown.get('completion_reward', 0)
                    episode_resource_reward += breakdown.get('resource_reward', 0)
                    episode_progress_reward += breakdown.get('progress_reward', 0)
                    episode_path_penalty += breakdown.get('path_penalty', 0)
                    episode_waste_penalty += breakdown.get('waste_penalty', 0)
                
                # 统计空闲UAV和未满足目标
                idle_uavs = sum(1 for uav in solver.env.uavs 
                               if np.any(uav.resources > 0) and 
                               not any(uav.id in [uav_info[0] for uav_info in target.allocated_uavs] 
                                      for target in solver.env.targets))
                unmet_targets = sum(1 for target in solver.env.targets 
                                   if np.any(target.remaining_resources > 0))
                
                idle_uav_counts.append(idle_uavs)
                unmet_target_counts.append(unmet_targets)
                
                state = next_state
                step_count += 1
                
                if done or truncated:
                    break
            
            total_waste_penalty += episode_waste_penalty
            total_completion_reward += episode_completion_reward
            total_resource_reward += episode_resource_reward
            total_progress_reward += episode_progress_reward
            total_path_penalty += episode_path_penalty
            
            if episode % 10 == 0:
                print(f"Episode {episode}: 浪费惩罚={episode_waste_penalty:.1f}, "
                      f"完成奖励={episode_completion_reward:.1f}")
        
        # 分析结果
        print(f"\n奖励组件分析结果 (共{episodes}个episodes):")
        print(f"  完成奖励总计: {total_completion_reward:.1f}")
        print(f"  资源奖励总计: {total_resource_reward:.1f}")
        print(f"  进度奖励总计: {total_progress_reward:.1f}")
        print(f"  路径惩罚总计: {total_path_penalty:.1f}")
        print(f"  浪费惩罚总计: {total_waste_penalty:.1f}")
        
        print(f"\n资源浪费情况:")
        print(f"  平均空闲UAV数量: {np.mean(idle_uav_counts):.2f}")
        print(f"  平均未满足目标数量: {np.mean(unmet_target_counts):.2f}")
        print(f"  浪费惩罚占总奖励比例: {abs(total_waste_penalty)/(total_completion_reward + total_resource_reward + total_progress_reward + 1e-6):.2%}")
        
        # 生成可视化
        self._plot_reward_analysis(idle_uav_counts, unmet_target_counts)
        
        return {
            'completion_reward': total_completion_reward,
            'resource_reward': total_resource_reward,
            'progress_reward': total_progress_reward,
            'path_penalty': total_path_penalty,
            'waste_penalty': total_waste_penalty,
            'avg_idle_uavs': np.mean(idle_uav_counts),
            'avg_unmet_targets': np.mean(unmet_target_counts)
        }
    
    def _plot_reward_analysis(self, idle_uav_counts, unmet_target_counts):
        """绘制奖励分析图表"""
        set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 空闲UAV数量变化
        ax1.plot(idle_uav_counts, alpha=0.7, label='空闲UAV数量')
        ax1.set_title('空闲UAV数量变化')
        ax1.set_xlabel('步数')
        ax1.set_ylabel('空闲UAV数量')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 未满足目标数量变化
        ax2.plot(unmet_target_counts, alpha=0.7, color='red', label='未满足目标数量')
        ax2.set_title('未满足目标数量变化')
        ax2.set_xlabel('步数')
        ax2.set_ylabel('未满足目标数量')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('reward_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"奖励分析图表已保存至: reward_analysis.png")

if __name__ == "__main__":
    analyzer = RewardAnalyzer()
    results = analyzer.analyze_reward_components()