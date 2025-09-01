#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强训练使用示例
展示如何使用新的训练功能
"""

import os
import sys
from datetime import datetime

from config import Config
from scenarios import get_curriculum_scenarios
from environment import DirectedGraph, UAVTaskEnv
from main import GraphRLSolver
from enhanced_trainer import EnhancedTrainer
from enhanced_training_config import EnhancedTrainingConfig
from baseline_config import BaselineConfig
from model_manager import EnsembleInference

def main():
    """主函数"""
    print("🚀 增强训练系统示例")
    print("="*60)
    
    # 1. 创建配置
    config = Config()
    
    # 应用基线配置（可选）
    # BaselineConfig.apply_to_config(config)
    
    # 或者手动设置增强配置
    config.NETWORK_TYPE = 'ZeroShotGNN'
    config.ENABLE_PBRS = True
    config.ENABLE_REWARD_DEBUG = False  # 训练时关闭以提高性能
    
    # 2. 创建场景和环境
    curriculum_scenarios = get_curriculum_scenarios()
    scenario_func, level_name, description = curriculum_scenarios[0]
    scenario = scenario_func()
    
    if isinstance(scenario, tuple):
        uavs, targets, obstacles = scenario
    else:
        uavs = scenario['uavs']
        targets = scenario['targets']
        obstacles = scenario.get('obstacles', [])
    
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="graph")
    
    # 3. 创建求解器
    i_dim = len(uavs) + len(targets)
    h_dim = 128
    o_dim = len(uavs) * len(targets) * len(uavs[0].resources)
    
    solver = GraphRLSolver(
        uavs=uavs, targets=targets, graph=graph, obstacles=obstacles,
        i_dim=i_dim, h_dim=[h_dim], o_dim=o_dim, config=config,
        obs_mode="graph"
    )
    
    # 4. 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"enhanced_training_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. 创建增强训练器
    trainer = EnhancedTrainer(solver, config, output_dir)
    
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 训练场景: {level_name}")
    
    # 6. 选择训练模式
    print(f"\n请选择训练模式:")
    print(f"1. 基线训练 (稳定版本)")
    print(f"2. 增强训练 (优化版本)")
    print(f"3. 对比训练 (两种模式)")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 基线训练
        print(f"\n🔧 执行基线训练...")
        results = trainer.train_enhanced(episodes=500, use_baseline=True)
        
    elif choice == "2":
        # 增强训练
        print(f"\n🚀 执行增强训练...")
        results = trainer.train_enhanced(episodes=1000, use_baseline=False)
        
        # 演示集成推理
        if results['model_saves_count'] > 0:
            print(f"\n🎯 演示集成推理...")
            
            ensemble = EnsembleInference(
                trainer.model_manager,
                type(solver.policy_net),
                temperature=EnhancedTrainingConfig.SOFTMAX_TEMPERATURE
            )
            
            ensemble.load_ensemble_models(
                top_n=min(EnhancedTrainingConfig.ENSEMBLE_SIZE, results['model_saves_count']),
                i_dim=i_dim, h_dim=[h_dim], o_dim=o_dim
            )
            
            # 测试几步推理
            state = env.reset()
            for step in range(5):
                state_tensor = solver._prepare_state_tensor(state)
                action = ensemble.predict(state_tensor, method='weighted_softmax')
                
                next_state, reward, done, truncated, info = env.step(action)
                print(f"  Step {step+1}: Action={action}, Reward={reward:.2f}")
                
                if done or truncated:
                    break
                state = next_state
        
    elif choice == "3":
        # 对比训练
        print(f"\n📊 执行对比训练...")
        
        print(f"阶段1: 基线训练")
        baseline_results = trainer.train_enhanced(episodes=300, use_baseline=True)
        
        # 重置求解器状态
        solver.epsilon = EnhancedTrainingConfig.EPSILON_START
        
        print(f"\n阶段2: 增强训练")
        enhanced_results = trainer.train_enhanced(episodes=500, use_baseline=False)
        
        # 对比结果
        print(f"\n📈 对比结果:")
        print(f"基线训练 - 最佳分数: {baseline_results['best_score']:.1f}, "
              f"最终Epsilon: {baseline_results['final_epsilon']:.6f}")
        print(f"增强训练 - 最佳分数: {enhanced_results['best_score']:.1f}, "
              f"最终Epsilon: {enhanced_results['final_epsilon']:.6f}, "
              f"保存模型: {enhanced_results['model_saves_count']}")
    
    else:
        print(f"无效选择，使用默认增强训练")
        results = trainer.train_enhanced(episodes=500, use_baseline=False)
    
    print(f"\n🎉 训练完成!")
    print(f"📁 所有结果保存在: {output_dir}")
    
    # 显示配置信息
    print(f"\n📋 使用的配置:")
    if hasattr(results, 'config_type'):
        if results['config_type'] == 'baseline':
            print("基线配置参数:")
            baseline_config = BaselineConfig.get_baseline_config()
            for key, value in baseline_config.items():
                print(f"  {key}: {value}")
        else:
            print("增强配置参数:")
            print(f"  EPSILON_START: {EnhancedTrainingConfig.EPSILON_START}")
            print(f"  EPSILON_END: {EnhancedTrainingConfig.EPSILON_END}")
            print(f"  EPSILON_DECAY: {EnhancedTrainingConfig.EPSILON_DECAY}")
            print(f"  SAVE_TOP_N_MODELS: {EnhancedTrainingConfig.SAVE_TOP_N_MODELS}")
            print(f"  ENSEMBLE_SIZE: {EnhancedTrainingConfig.ENSEMBLE_SIZE}")

if __name__ == "__main__":
    main()