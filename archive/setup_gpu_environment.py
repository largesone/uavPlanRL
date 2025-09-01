#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU环境设置脚本 - 帮助配置GPU训练环境
"""

import torch
import subprocess
import sys
import os

def check_cuda_installation():
    """检查CUDA安装情况"""
    print("=" * 60)
    print("CUDA安装检查")
    print("=" * 60)
    
    try:
        # 检查nvidia-smi命令
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA驱动已安装")
            print("GPU信息:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi命令失败")
            return False
    except FileNotFoundError:
        print("❌ 未找到nvidia-smi命令")
        print("请安装NVIDIA驱动程序")
        return False

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    print("\n" + "=" * 60)
    print("PyTorch CUDA支持检查")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        return True
    else:
        print("❌ PyTorch不支持CUDA")
        return False

def get_installation_commands():
    """获取安装命令"""
    print("\n" + "=" * 60)
    print("GPU环境安装指南")
    print("=" * 60)
    
    print("1. 安装NVIDIA驱动 (如果未安装):")
    print("   - 访问: https://www.nvidia.com/drivers/")
    print("   - 下载并安装适合您显卡的驱动")
    
    print("\n2. 安装CUDA Toolkit (推荐版本: 11.8 或 12.1):")
    print("   - 访问: https://developer.nvidia.com/cuda-toolkit")
    print("   - 下载并安装CUDA Toolkit")
    
    print("\n3. 安装支持CUDA的PyTorch:")
    print("   对于CUDA 11.8:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n   对于CUDA 12.1:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n4. 验证安装:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")

def create_gpu_config():
    """创建GPU配置文件"""
    gpu_config = {
        "use_gpu": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mixed_precision": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 if torch.cuda.is_available() else False,
        "memory_fraction": 0.8,
        "benchmark_mode": True
    }
    
    config_path = "gpu_config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump(gpu_config, f, indent=2)
    
    print(f"\n✅ GPU配置已保存至: {config_path}")
    return gpu_config

def main():
    """主函数"""
    print("GPU环境设置助手")
    
    # 检查CUDA
    cuda_available = check_cuda_installation()
    
    # 检查PyTorch
    pytorch_cuda = check_pytorch_cuda()
    
    # 创建配置
    config = create_gpu_config()
    
    print("\n" + "=" * 60)
    print("设置总结")
    print("=" * 60)
    
    if cuda_available and pytorch_cuda:
        print("🎉 GPU环境配置完成！")
        print("✅ 可以使用GPU进行训练")
        if config["mixed_precision"]:
            print("✅ 支持混合精度训练")
    elif cuda_available and not pytorch_cuda:
        print("⚠️  CUDA已安装，但PyTorch不支持CUDA")
        print("请重新安装支持CUDA的PyTorch版本")
        get_installation_commands()
    else:
        print("❌ GPU环境未配置")
        get_installation_commands()

if __name__ == "__main__":
    main()
