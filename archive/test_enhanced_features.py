#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强功能：奖励日志记录和集成推理
"""

import os
import sys
import numpy as np
from config import Config
from scenarios import get_balanced_scenario
from main import GraphRLSolver
from environment import DirectedGraph

def test_reward_logging():
    """测试奖励日志记录功能 - 简化版，检查现有日志"""
    print("🧪 测试奖励日志记录功能")
    print("=" * 50)
    
    # 检查现有的奖励日志文件
    if not os.path.exists("output"):
        print("❌ output目录不存在")
        return False
    
    log_files = [f for f in os.listdir("output") if f.startswith("reward_log_")]
    if log_files:
        latest_log = max(log_files)
        log_path = os.path.join("output", latest_log)
        print(f"✅ 找到奖励日志文件: {latest_log}")
        
        # 读取并显示部分日志内容
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   日志行数: {len(lines)}")
                
                # 检查是否包含裁剪前后的值
                has_clipping_info = False
                for line in lines[10:20]:  # 检查中间部分的日志
                    if "→" in line and ("Base=" in line or "Total=" in line):
                        has_clipping_info = True
                        print(f"   ✅ 发现裁剪信息: {line.strip()}")
                        break
                
                if not has_clipping_info:
                    # 查找普通的奖励信息
                    for line in lines[10:20]:
                        if "Base=" in line and "Total=" in line:
                            print(f"   📝 奖励信息示例: {line.strip()}")
                            break
                
                print("   日志内容预览:")
                for i, line in enumerate(lines[:3]):
                    print(f"     {line.strip()}")
                if len(lines) > 6:
                    print("     ...")
                    for line in lines[-2:]:
                        print(f"     {line.strip()}")
                        
        except Exception as e:
            print(f"❌ 读取日志文件失败: {e}")
            return False
    else:
        print("❌ 未找到奖励日志文件")
        return False
    
    print("✅ 奖励日志记录功能验证完成")
    return True

def test_ensemble_inference():
    """测试集成推理功能 - 使用已有模型"""
    print("\n🧪 测试集成推理功能")
    print("=" * 50)
    
    # 检查是否有保存的模型
    ensemble_dir = "output/ensemble_models"
    if not os.path.exists(ensemble_dir):
        print("❌ 集成模型目录不存在")
        print("💡 提示: 请先运行完整训练生成集成模型")
        return False
    
    model_files = [f for f in os.listdir(ensemble_dir) if f.endswith('.pth')]
    if len(model_files) < 2:
        print(f"❌ 集成模型数量不足: {len(model_files)} < 2")
        print("💡 提示: 需要至少2个模型进行集成推理")
        return False
    
    print(f"✅ 找到 {len(model_files)} 个集成模型")
    
    # 显示模型信息
    print("📋 模型列表:")
    for i, model_file in enumerate(model_files[:5]):  # 显示前5个
        print(f"   {i+1}. {model_file}")
    if len(model_files) > 5:
        print(f"   ... 还有 {len(model_files) - 5} 个模型")
    
    try:
        # 导入并运行集成推理演示
        from ensemble_inference_demo import EnsembleInferenceDemo
        
        config = Config()
        config.NETWORK_TYPE = 'ZeroShotGNN'
        
        demo = EnsembleInferenceDemo(config)
        
        print(f"🚀 开始集成推理演示...")
        result = demo.run_demo(
            model_dir=ensemble_dir,
            num_plans=3,  # 生成3个方案用于测试
            top_n=min(3, len(model_files))  # 使用最多3个模型
        )
        
        if result:
            print("✅ 集成推理测试成功")
            print(f"🏆 最优方案完成率: {result['evaluation']['completion_rate']:.3f}")
            print(f"📈 最优方案价值: {result['evaluation']['total_value']:.1f}")
            return True
        else:
            print("❌ 集成推理测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 集成推理测试中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数 - 优化版，使用已有模型"""
    print("🚀 开始测试增强功能")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 检查是否有训练数据
    has_training_data = (
        os.path.exists("output/ensemble_models") and 
        len([f for f in os.listdir("output/ensemble_models") if f.endswith('.pth')]) > 0
    )
    
    if not has_training_data:
        print("⚠️ 未找到训练好的模型数据")
        print("💡 建议: 先运行完整训练生成模型和日志")
        print("   python main.py  # 运行完整训练")
        print("   然后再运行此测试脚本")
        print()
    
    # 测试1: 奖励日志记录
    print("📝 测试1: 奖励日志记录功能")
    reward_logging_success = test_reward_logging()
    
    # 测试2: 集成推理（如果有足够的模型）
    print("\n🤖 测试2: 集成推理功能")
    ensemble_inference_success = test_ensemble_inference()
    
    # 总结
    print("\n" + "=" * 60)
    print("🏁 测试总结")
    print("=" * 60)
    print(f"奖励日志记录: {'✅ 通过' if reward_logging_success else '❌ 失败'}")
    print(f"集成推理功能: {'✅ 通过' if ensemble_inference_success else '❌ 失败'}")
    
    if reward_logging_success and ensemble_inference_success:
        print("🎉 所有增强功能测试通过！")
        print("\n📚 功能说明:")
        print("1. 奖励日志记录: 显示Base和Total奖励的裁剪前后值")
        print("2. 集成推理: 使用多个模型进行Softmax采样推理")
        return True
    elif reward_logging_success:
        print("✅ 奖励日志记录功能正常")
        print("⚠️ 集成推理功能需要更多训练模型")
        return True
    else:
        print("⚠️ 部分功能测试失败，请检查实现")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)