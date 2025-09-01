#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU性能测试脚本 - 验证GPU加速效果
"""

import torch
import time
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gpu_availability():
    """测试GPU可用性"""
    print("=" * 60)
    print("GPU可用性测试")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - 显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - 计算能力: {props.major}.{props.minor}")
            print(f"  - 多处理器数量: {props.multi_processor_count}")
        
        # 测试GPU内存
        device = torch.device("cuda")
        print(f"\n当前GPU内存使用:")
        print(f"  - 已分配: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"  - 已缓存: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
        
        return True
    else:
        print("❌ GPU不可用")
        return False

def test_gpu_performance():
    """测试GPU性能"""
    print("\n" + "=" * 60)
    print("GPU vs CPU性能对比测试")
    print("=" * 60)
    
    # 测试参数
    batch_size = 64
    input_size = 512
    hidden_size = 256
    output_size = 48
    iterations = 1000
    
    # 创建测试网络
    class TestNetwork(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # CPU测试
    print("测试CPU性能...")
    cpu_device = torch.device("cpu")
    cpu_model = TestNetwork(input_size, hidden_size, output_size).to(cpu_device)
    cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=0.001)
    cpu_criterion = torch.nn.MSELoss()
    
    # 预热
    for _ in range(10):
        x = torch.randn(batch_size, input_size).to(cpu_device)
        y = torch.randn(batch_size, output_size).to(cpu_device)
        output = cpu_model(x)
        loss = cpu_criterion(output, y)
        cpu_optimizer.zero_grad()
        loss.backward()
        cpu_optimizer.step()
    
    # CPU计时
    start_time = time.time()
    for i in range(iterations):
        x = torch.randn(batch_size, input_size).to(cpu_device)
        y = torch.randn(batch_size, output_size).to(cpu_device)
        output = cpu_model(x)
        loss = cpu_criterion(output, y)
        cpu_optimizer.zero_grad()
        loss.backward()
        cpu_optimizer.step()
    cpu_time = time.time() - start_time
    
    print(f"CPU训练时间: {cpu_time:.2f}秒 ({iterations}次迭代)")
    print(f"CPU平均每次迭代: {cpu_time/iterations*1000:.2f}ms")
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n测试GPU性能...")
        gpu_device = torch.device("cuda")
        gpu_model = TestNetwork(input_size, hidden_size, output_size).to(gpu_device)
        gpu_optimizer = torch.optim.Adam(gpu_model.parameters(), lr=0.001)
        gpu_criterion = torch.nn.MSELoss()
        
        # 预热
        for _ in range(10):
            x = torch.randn(batch_size, input_size).to(gpu_device)
            y = torch.randn(batch_size, output_size).to(gpu_device)
            output = gpu_model(x)
            loss = gpu_criterion(output, y)
            gpu_optimizer.zero_grad()
            loss.backward()
            gpu_optimizer.step()
        
        torch.cuda.synchronize()  # 确保GPU操作完成
        
        # GPU计时
        start_time = time.time()
        for i in range(iterations):
            x = torch.randn(batch_size, input_size).to(gpu_device)
            y = torch.randn(batch_size, output_size).to(gpu_device)
            output = gpu_model(x)
            loss = gpu_criterion(output, y)
            gpu_optimizer.zero_grad()
            loss.backward()
            gpu_optimizer.step()
        
        torch.cuda.synchronize()  # 确保GPU操作完成
        gpu_time = time.time() - start_time
        
        print(f"GPU训练时间: {gpu_time:.2f}秒 ({iterations}次迭代)")
        print(f"GPU平均每次迭代: {gpu_time/iterations*1000:.2f}ms")
        
        # 计算加速比
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU加速比: {speedup:.2f}x")
        
        if speedup > 2:
            print("✅ GPU加速效果显著")
        elif speedup > 1.2:
            print("⚡ GPU有一定加速效果")
        else:
            print("⚠️  GPU加速效果不明显，可能需要更大的模型或批次")
        
        return speedup
    else:
        print("❌ 无法进行GPU性能测试")
        return 0

def test_mixed_precision():
    """测试混合精度训练"""
    if not torch.cuda.is_available():
        print("❌ GPU不可用，无法测试混合精度")
        return
    
    print("\n" + "=" * 60)
    print("混合精度训练测试")
    print("=" * 60)
    
    device = torch.device("cuda")
    
    # 检查是否支持混合精度
    if torch.cuda.get_device_capability()[0] >= 7:
        print("✅ 支持Tensor Core混合精度训练")
        
        # 创建测试模型
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 48)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()
        criterion = torch.nn.MSELoss()
        
        batch_size = 128
        iterations = 500
        
        # 测试FP32训练
        print("测试FP32训练...")
        start_time = time.time()
        for i in range(iterations):
            x = torch.randn(batch_size, 512).to(device)
            y = torch.randn(batch_size, 48).to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        fp32_time = time.time() - start_time
        
        # 测试混合精度训练
        print("测试混合精度训练...")
        start_time = time.time()
        for i in range(iterations):
            x = torch.randn(batch_size, 512).to(device)
            y = torch.randn(batch_size, 48).to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        amp_time = time.time() - start_time
        
        speedup = fp32_time / amp_time
        print(f"FP32训练时间: {fp32_time:.2f}秒")
        print(f"混合精度训练时间: {amp_time:.2f}秒")
        print(f"混合精度加速比: {speedup:.2f}x")
        
        if speedup > 1.3:
            print("✅ 混合精度训练效果显著")
        else:
            print("⚡ 混合精度训练有一定效果")
            
    else:
        print("⚠️  当前GPU不支持Tensor Core，混合精度效果有限")

def main():
    """主函数"""
    print("GPU性能测试开始...")
    
    # 测试GPU可用性
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        # 测试性能
        speedup = test_gpu_performance()
        
        # 测试混合精度
        test_mixed_precision()
        
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print("✅ GPU环境配置正常")
        print(f"🚀 建议使用GPU进行训练以获得 {speedup:.1f}x 加速")
        
        if torch.cuda.get_device_capability()[0] >= 7:
            print("💡 建议启用混合精度训练以进一步提升性能")
        
    else:
        print("\n❌ GPU不可用，建议:")
        print("1. 检查CUDA驱动是否正确安装")
        print("2. 检查PyTorch是否为CUDA版本")
        print("3. 重新安装CUDA版本的PyTorch")

if __name__ == "__main__":
    main()
