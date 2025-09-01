#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时文件清理脚本
删除项目中的临时文件、测试文件和调试文件，保持核心代码结构清晰
"""

import os
import shutil
from pathlib import Path

def cleanup_temp_files():
    """清理临时文件和文件夹"""
    
    # 要删除的临时目录列表
    temp_dirs = [
        'temp',
        'temp_files', 
        'temp_scripts',
        'temp_docs',
        'test_files',
        'migration_temp',
        '__pycache__',
        '.pytest_cache'
    ]
    
    # 要删除的临时文件模式
    temp_file_patterns = [
        '*.tmp',
        '*.temp',
        '*.log',
        '*.bak',
        '*.old',
        '*_test.py',
        '*_debug.py',
        '*_fix.py',
        '*_temp.py',
        '*_backup.py'
    ]
    
    # 要保留的核心文件（即使匹配临时文件模式）
    core_files = {
        'main.py',
        'config.py', 
        'environment.py',
        'solvers.py',
        'trainer.py',
        'networks.py',
        'evaluator.py',
        'scenarios.py',
        'entities.py',
        'path_planning.py',
        'distance_service.py',
        'model_manager.py',
        'adaptive_curriculum.py',
        'ensemble_inference_manager.py',
        'scenario_viewer.py',
        'requirements.txt',
        'README.md'
    }
    
    print("🧹 开始清理临时文件和文件夹...")
    
    # 1. 删除临时目录
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"✅ 已删除临时目录: {temp_dir}")
                else:
                    os.remove(temp_dir)
                    print(f"✅ 已删除临时文件: {temp_dir}")
            except Exception as e:
                print(f"⚠️ 删除 {temp_dir} 时出错: {e}")
    
    # 2. 删除根目录下的临时文件
    root_dir = Path('.')
    deleted_count = 0
    
    for file_pattern in temp_file_patterns:
        for file_path in root_dir.glob(file_pattern):
            # 跳过核心文件
            if file_path.name in core_files:
                continue
                
            # 跳过.git目录
            if '.git' in str(file_path):
                continue
                
            try:
                if file_path.is_file():
                    file_path.unlink()
                    print(f"✅ 已删除临时文件: {file_path.name}")
                    deleted_count += 1
            except Exception as e:
                print(f"⚠️ 删除 {file_path.name} 时出错: {e}")
    
    # 3. 清理output目录中的临时文件
    output_dir = Path('output')
    if output_dir.exists():
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                # 删除临时日志文件
                for log_file in subdir.glob('*.log'):
                    try:
                        log_file.unlink()
                        print(f"✅ 已删除日志文件: {log_file.name}")
                    except Exception as e:
                        print(f"⚠️ 删除 {log_file.name} 时出错: {e}")
    
    # 4. 清理Python缓存文件
    for root, dirs, files in os.walk('.'):
        # 跳过.git目录
        if '.git' in root:
            continue
            
        # 删除__pycache__目录
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ 已删除缓存目录: {cache_dir}")
            except Exception as e:
                print(f"⚠️ 删除 {cache_dir} 时出错: {e}")
        
        # 删除.pyc文件
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                try:
                    os.remove(pyc_file)
                    print(f"✅ 已删除缓存文件: {file}")
                except Exception as e:
                    print(f"⚠️ 删除 {file} 时出错: {e}")
    
    print(f"\n🎉 清理完成！共删除 {deleted_count} 个临时文件")
    print("\n📁 当前项目结构:")
    print_project_structure()

def print_project_structure():
    """打印当前项目结构"""
    core_dirs = ['output', 'scenarios', 'docs', 'archive', 'analysis_reports']
    
    print("\n根目录文件:")
    for item in sorted(os.listdir('.')):
        if os.path.isfile(item) and not item.startswith('.'):
            print(f"  📄 {item}")
    
    print("\n核心目录:")
    for dir_name in core_dirs:
        if os.path.exists(dir_name):
            print(f"  📁 {dir_name}/")
            # 显示子目录
            try:
                subdirs = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
                for subdir in sorted(subdirs)[:5]:  # 只显示前5个
                    print(f"    └── {subdir}/")
                if len(subdirs) > 5:
                    print(f"    └── ... 还有 {len(subdirs) - 5} 个子目录")
            except Exception:
                pass

if __name__ == "__main__":
    cleanup_temp_files()
