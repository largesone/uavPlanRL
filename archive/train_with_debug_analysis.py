#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成comprehensive_debug_analysis的训练脚本
专门用于分析奖励收敛和资源分配问题
"""

import sys
import os
import numpy as np
import torch
import time
from datetime import datetime
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置matplotlib字体，避免字体警告
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 尝试设置中文字体
try:
    import matplotlib.font_manager as fm
    # 查找系统中的中文字体
    font_list = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    
    for font in chinese_fonts:
        if font in font_list:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
    
    print(f"✅ 字体设置完成: {plt.rcParams['font.sans-serif'][0]}")
except Exception as e:
    print(f"⚠️  字体设置警告: {e}")
    # 使用默认字体，禁用unicode minus以避免警告
    plt.rcParams['axes.unicode_minus'] = False

# 导入主要模块
from main import *
from comprehensive_debug_analysis import ComprehensiveDebugAnalyzer
from scenarios import get_new_experimental_scenario
from config import Config

def calculate_completion_rate(env):
    """计算任务完成率"""
    completed = sum(1 for target in env.targets if np.all(target.remaining_resources <= 0))
    return completed / len(env.targets)

def calculate_resource_utilization(env):
    """计算资源利用率"""
    total_initial = sum(np.sum(getattr(uav, 'initial_resources', uav.resources)) for uav in env.uavs)
    total_remaining = sum(np.sum(uav.resources) for uav in env.uavs)
    if total_initial == 0:
        return 0.0
    return (total_initial - total_remaining) / total_initial

def run_training_with_analysis():
    """运行带有综合分析的训练"""
    
    print("=" * 80)
    print("开始带有综合调试分析的训练")
    print("=" * 80)
    
    # 初始化配置
    config = Config()
    config.EPISODES = 300  # 减少episode数量以便快速分析
    config.NETWORK_TYPE = 'ZeroShotGNN'
    
    # 初始化分析器
    analyzer = ComprehensiveDebugAnalyzer("debug_analysis_output")
    
    # 设置场景
    uavs, targets, obstacles = get_new_experimental_scenario(config.OBSTACLE_TOLERANCE)
    
    # 设置字体
    set_chinese_font(manual_font_path="C:/Windows/Fonts/simhei.ttf")
    
    # 创建有向图
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    
    # 计算网络维度
    if config.NETWORK_TYPE == 'ZeroShotGNN':
        obs_mode = "graph"
        i_dim = 64  # 占位值，图模式不使用
        h_dim = 128
        o_dim = len(targets) * len(uavs) * config.GRAPH_N_PHI
    else:
        obs_mode = "flat"
        env_temp = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
        state_temp = env_temp.reset()
        i_dim = len(state_temp)
        h_dim = 256
        o_dim = len(targets) * len(uavs) * config.GRAPH_N_PHI
    
    print(f"网络配置: {config.NETWORK_TYPE}, 观测模式: {obs_mode}")
    print(f"维度: 输入={i_dim}, 隐藏={h_dim}, 输出={o_dim}")
    
    # 显示GPU信息
    if torch.cuda.is_available():
        print(f"GPU信息:")
        print(f"  - 设备数量: {torch.cuda.device_count()}")
        print(f"  - 当前设备: {torch.cuda.current_device()}")
        print(f"  - 设备名称: {torch.cuda.get_device_name()}")
        print(f"  - CUDA版本: {torch.version.cuda}")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/experimental场景_{config.NETWORK_TYPE}"
    tensorboard_dir = f"{output_dir}/tensorboard/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 初始化求解器
    solver = GraphRLSolver(
        uavs, targets, graph, obstacles, 
        i_dim, h_dim, o_dim, config, 
        network_type=config.NETWORK_TYPE,
        tensorboard_dir=tensorboard_dir,
        obs_mode=obs_mode
    )
    
    print(f"开始训练 {config.NETWORK_TYPE} 网络，训练轮数: {config.EPISODES}")
    
    # 训练统计
    episode_rewards = []
    best_avg_reward = float('-inf')
    patience_counter = 0
    last_loss = 0.0
    
    # 训练循环
    for episode in range(config.EPISODES):
        state = solver.env.reset()
        episode_reward = 0
        step = 0
        done = False
        truncated = False
        
        # Episode信息收集
        episode_info = {
            'synergy_attacks': 0,
            'invalid_actions': 0,
            'num_actions': 0,
            'step_rewards': [],
            'contributions': []
        }
        
        # 记录初始资源状态
        initial_uav_resources = [np.sum(uav.resources) for uav in solver.env.uavs]
        initial_target_demands = [np.sum(target.remaining_resources) for target in solver.env.targets]
        
        while not done and not truncated:
            # 准备状态张量
            state_tensor = solver._prepare_state_tensor(state)
            
            # 选择动作
            action = solver.select_action(state_tensor)
            action_idx = action.item()
            
            # 执行动作
            next_state, reward, done, truncated, info = solver.env.step(action_idx)
            
            # 记录step数据
            analyzer.log_step_data(episode, step, {
                'action': action_idx,
                'reward': reward,
                'state_before': state,
                'state_after': next_state,
                'target_id': info.get('target_id'),
                'uav_id': info.get('uav_id'),
                'contribution': info.get('actual_contribution', 0),
                'path_length': info.get('path_length', 0),
                'is_valid': not info.get('invalid_action', False)
            })
            
            # 存储经验
            next_state_tensor = solver._prepare_state_tensor(next_state)
            reward_tensor = torch.tensor([reward], device=solver.device)
            done_tensor = torch.tensor([done], device=solver.device, dtype=torch.bool)
            
            if solver.use_per:
                solver.memory.push(state_tensor, action, reward_tensor, next_state_tensor, done_tensor)
            else:
                solver.memory.append((state_tensor, action, reward_tensor, next_state_tensor, done_tensor))
            
            # 优化模型
            if len(solver.memory) > config.BATCH_SIZE:
                loss_result = solver.optimize_model()
                if loss_result is not None:
                    last_loss = loss_result
            
            # 更新统计
            episode_reward += reward
            episode_info['num_actions'] += 1
            episode_info['step_rewards'].append(reward)
            episode_info['contributions'].append(info.get('actual_contribution', 0))
            
            if info.get('invalid_action'):
                episode_info['invalid_actions'] += 1
            
            # 检测协同攻击
            if 'synergy_attacks' in str(reward) or reward > 800:  # 简单的协同检测
                episode_info['synergy_attacks'] += 1
            
            state = next_state
            step += 1
        
        # 更新探索率
        solver.epsilon = max(solver.epsilon_min, solver.epsilon * solver.epsilon_decay)
        
        # 记录episode数据
        episode_rewards.append(episode_reward)
        
        # 计算完成率和资源利用率
        completion_rate = calculate_completion_rate(solver.env)
        resource_utilization = calculate_resource_utilization(solver.env)
        
        # 检测最终成功（1000分奖励）
        final_success = episode_reward > 900  # 接近1000分就算成功
        
        # 准备episode信息
        episode_data = {
            'total_reward': episode_reward,
            'completion_rate': completion_rate,
            'resource_utilization': resource_utilization,
            'num_actions': episode_info['num_actions'],
            'invalid_actions': episode_info['invalid_actions'],
            'final_success': final_success,
            'synergy_attacks': episode_info['synergy_attacks'],
            'exploration_rate': solver.epsilon,
            'learning_rate': solver.optimizer.param_groups[0]['lr'],
            'loss': last_loss,
            'step_count': step
        }
        
        # 记录到分析器
        analyzer.log_episode_data(episode, episode_data)
        
        # 记录资源分配状态
        analyzer.log_resource_allocation(episode, solver.env.uavs, solver.env.targets)
        
        # 计算移动平均奖励
        if len(episode_rewards) >= 20:
            avg_reward = np.mean(episode_rewards[-20:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                patience_counter = 0
                
                # 保存最佳模型
                best_model_path = os.path.join(output_dir, f"best_{config.NETWORK_TYPE}_model.pth")
                torch.save(solver.policy_net.state_dict(), best_model_path)
                print(f"Episode {episode}: 新的最佳平均奖励 {avg_reward:.2f} (最近20轮)")
            else:
                patience_counter += 1
        
        # 定期输出进度
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            gpu_info = solver.get_gpu_memory_info() if hasattr(solver, 'get_gpu_memory_info') else ""
            print(f"Episode {episode}: 奖励={episode_reward:.2f}, 平均奖励={avg_reward:.2f}, "
                  f"完成率={completion_rate:.3f}, 资源利用率={resource_utilization:.3f}, "
                  f"探索率={solver.epsilon:.3f}")
            if gpu_info:
                print(f"  {gpu_info}")
            
            # 定期清理GPU内存
            if torch.cuda.is_available() and episode % 100 == 0:
                torch.cuda.empty_cache()
        
        # 定期生成分析报告
        if episode % 50 == 0 and episode > 0:
            print(f"\n=== Episode {episode} 中期分析报告 ===")
            analyzer.generate_comprehensive_plots()
            report = analyzer.generate_detailed_report()
            print("中期分析完成\n")
        
        # 早停检查
        if patience_counter >= config.PATIENCE:
            print(f"早停触发于第 {episode} 回合: 奖励无改进超过 {config.PATIENCE} 轮")
            break
    
    # 训练结束，生成最终分析报告
    print("\n" + "=" * 80)
    print("训练完成，生成最终综合分析报告")
    print("=" * 80)
    
    # 生成最终图表和报告
    plot_path = analyzer.generate_comprehensive_plots()
    final_report = analyzer.generate_detailed_report()
    json_path, pickle_path = analyzer.export_data()
    
    # 输出最终统计
    print(f"\n训练统计:")
    print(f"总回合数: {len(episode_rewards)}")
    print(f"最佳单轮奖励: {max(episode_rewards):.2f}")
    print(f"最佳平均奖励: {best_avg_reward:.2f}")
    print(f"最终探索率: {solver.epsilon:.4f}")
    
    # 1000分奖励分析
    high_reward_episodes = [i for i, r in enumerate(episode_rewards) if r > 900]
    print(f"高奖励(>900)episode数: {len(high_reward_episodes)}")
    print(f"高奖励获得率: {len(high_reward_episodes)/len(episode_rewards):.1%}")
    
    if high_reward_episodes:
        print(f"高奖励episode: {high_reward_episodes[:10]}")  # 显示前10个
    
    # 输出分析报告
    print("\n" + "=" * 80)
    print("最终分析报告:")
    print("=" * 80)
    print(final_report)
    
    # 输出文件路径
    print(f"\n生成的文件:")
    print(f"  - 综合分析图表: {plot_path}")
    print(f"  - 调试数据(JSON): {json_path}")
    print(f"  - 调试数据(Pickle): {pickle_path}")
    
    return analyzer, episode_rewards, solver

if __name__ == "__main__":
    try:
        analyzer, rewards, solver = run_training_with_analysis()
        
        print("\n" + "=" * 80)
        print("训练和分析完成！")
        print("=" * 80)
        
        # 额外的问题检测
        print(f"\n问题检测统计:")
        for issue_type, count in analyzer.issue_counters.items():
            if count > 0:
                print(f"  - {issue_type.replace('_', ' ').title()}: {count}次")
        
        # 给出具体建议
        print(f"\n具体建议:")
        if analyzer.issue_counters['unused_resources'] > len(rewards) * 0.2:
            print("  1. 检测到大量未使用资源，建议优化动作选择策略")
        
        if len([r for r in rewards if r > 900]) < len(rewards) * 0.1:
            print("  2. 1000分奖励获得率过低，建议调整奖励函数权重")
        
        if np.std(rewards) > np.mean(rewards):
            print("  3. 奖励方差过大，建议启用奖励标准化")
        
        print("\n分析数据已保存，可用于进一步研究。")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
