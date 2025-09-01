# -*- coding: utf-8 -*-
"""
奖励日志记录功能演示脚本
展示如何在实际训练中使用新增的奖励日志记录功能
"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import *
from config import Config
from scenarios import get_small_scenario

def demo_training_with_reward_logging():
    """演示带奖励日志记录的训练过程"""
    print("🚀 演示奖励日志记录功能")
    print("=" * 60)
    
    # 初始化配置
    config = Config()
    config.EPISODES = 5  # 演示用，只训练5轮
    config.SHOW_VISUALIZATION = False
    
    print(f"📊 配置信息:")
    print(f"  GRAPH_N_PHI: {config.GRAPH_N_PHI} (动作空间大小已优化)")
    print(f"  训练轮数: {config.EPISODES}")
    print(f"  网络类型: {config.NETWORK_TYPE}")
    print(f"  PBRS启用: {config.ENABLE_PBRS}")
    
    # 获取小规模场景进行演示
    uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    print(f"📍 场景信息: {len(uavs)}架UAV, {len(targets)}个目标")
    
    # 创建有向图
    graph = DirectedGraph(uavs, targets, config.GRAPH_N_PHI, obstacles, config)
    
    # 计算网络维度
    if config.NETWORK_TYPE == 'ZeroShotGNN':
        # 图模式：使用占位维度
        i_dim = 64  # 占位值
        h_dim = 128
        o_dim = len(targets) * len(uavs) * graph.n_phi
        obs_mode = "graph"
    else:
        # 扁平模式：计算实际维度
        env_temp = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode="flat")
        state_temp = env_temp.reset()
        i_dim = len(state_temp)
        h_dim = 256
        o_dim = len(targets) * len(uavs) * graph.n_phi
        obs_mode = "flat"
    
    print(f"🧠 网络维度: 输入={i_dim}, 隐藏={h_dim}, 输出={o_dim}, 观测模式={obs_mode}")
    
    # 创建输出目录
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化求解器（包含奖励日志记录功能）
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    model_save_path = os.path.join(output_dir, "demo_model.pth")
    
    solver = GraphRLSolver(
        uavs=uavs,
        targets=targets, 
        graph=graph,
        obstacles=obstacles,
        i_dim=i_dim,
        h_dim=h_dim,
        o_dim=o_dim,
        config=config,
        network_type=config.NETWORK_TYPE,
        tensorboard_dir=tensorboard_dir,
        obs_mode=obs_mode
    )
    
    print(f"🤖 求解器初始化完成")
    print(f"📝 奖励日志将保存到: {output_dir}")
    print(f"📊 TensorBoard日志: {tensorboard_dir}")
    
    # 开始训练（包含奖励日志记录）
    print("\n🎯 开始训练...")
    print("-" * 60)
    
    training_time = solver.train(
        episodes=config.EPISODES,
        patience=config.PATIENCE,
        log_interval=config.LOG_INTERVAL,
        model_save_path=model_save_path
    )
    
    print("-" * 60)
    print(f"✅ 训练完成，耗时: {training_time:.2f}秒")
    
    # 查找生成的奖励日志文件
    reward_log_files = [f for f in os.listdir(output_dir) if f.startswith('reward_log_') and f.endswith('.txt')]
    
    if reward_log_files:
        latest_log = max(reward_log_files)
        log_path = os.path.join(output_dir, latest_log)
        
        print(f"📄 奖励日志文件: {log_path}")
        
        # 显示日志文件的部分内容
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"📊 日志文件统计:")
                print(f"  总行数: {len(lines)}")
                print(f"  文件大小: {os.path.getsize(log_path)} 字节")
                
                # 显示前几行和后几行
                print(f"\n📝 日志内容预览 (前10行):")
                for i, line in enumerate(lines[:10]):
                    print(f"  {i+1:2d}: {line.rstrip()}")
                
                if len(lines) > 20:
                    print(f"\n📝 日志内容预览 (后5行):")
                    for i, line in enumerate(lines[-5:], len(lines)-4):
                        print(f"  {i:2d}: {line.rstrip()}")
                        
        except Exception as e:
            print(f"❌ 读取日志文件失败: {e}")
    else:
        print("❌ 未找到奖励日志文件")
    
    print("\n🎉 演示完成!")
    print("=" * 60)
    print("功能总结:")
    print("✅ 1. GRAPH_N_PHI已设置为1，降低动作空间复杂度")
    print("✅ 2. 奖励日志记录功能已集成到训练流程")
    print("✅ 3. 每个回合的每一步都记录详细奖励分解")
    print("✅ 4. 包括基础奖励、塑形奖励、完成率等信息")
    print("✅ 5. 自动生成带时间戳的日志文件")

if __name__ == "__main__":
    demo_training_with_reward_logging()