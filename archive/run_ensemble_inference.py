#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的集成推理演示脚本
直接使用已训练的模型进行集成Softmax采样推理
"""

import os
import sys
from config import Config
from ensemble_inference_demo import EnsembleInferenceDemo

def main():
    """主函数"""
    print("🚀 集成Softmax采样推理演示")
    print("=" * 60)
    
    # 检查模型目录
    ensemble_dir = "output/ensemble_models"
    if not os.path.exists(ensemble_dir):
        print("❌ 集成模型目录不存在")
        print("💡 请先运行训练生成集成模型:")
        print("   python main.py")
        return False
    
    model_files = [f for f in os.listdir(ensemble_dir) if f.endswith('.pth')]
    if len(model_files) < 2:
        print(f"❌ 集成模型数量不足: {len(model_files)} < 2")
        print("💡 需要至少2个模型进行集成推理")
        return False
    
    print(f"✅ 找到 {len(model_files)} 个集成模型")
    
    # 创建配置
    config = Config()
    config.NETWORK_TYPE = 'ZeroShotGNN'
    
    # 创建演示实例
    demo = EnsembleInferenceDemo(config)
    
    # 运行演示
    print(f"🎯 开始集成推理演示...")
    result = demo.run_demo(
        model_dir=ensemble_dir,
        num_plans=3,  # 生成3个方案（减少数量提高效率）
        top_n=min(3, len(model_files))  # 使用前3个最优模型
    )
    
    if result:
        print(f"\n🎉 集成推理演示成功完成！")
        print(f"🏆 最优方案ID: {result['plan_id']}")
        print(f"📈 完成率: {result['evaluation']['completion_rate']:.3f}")
        print(f"💰 总价值: {result['evaluation']['total_value']:.1f}")
        print(f"🔢 步数: {result['steps']}")
        
        print(f"\n💡 集成推理优势:")
        print(f"   - 使用了 {len(demo.ensemble.models)} 个训练好的模型")
        print(f"   - 低温度(0.1)Softmax采样增加决策多样性")
        print(f"   - 通过评估选择最优方案")
        print(f"   - 提高了决策的鲁棒性和性能")
        
        return True
    else:
        print("❌ 集成推理演示失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)