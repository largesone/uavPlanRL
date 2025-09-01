#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复main.py中train方法和相关方法的缩进问题
"""

def fix_main_py_indentation():
    """修复main.py的缩进问题"""
    
    # 读取文件内容
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到需要修复的方法
    in_train_method = False
    in_get_convergence_method = False
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # 检测train方法开始
        if 'def train(self, episodes, patience, log_interval, model_save_path):' in line:
            in_train_method = True
            fixed_lines.append(line)
            continue
        
        # 检测get_convergence_metrics方法开始
        if 'def get_convergence_metrics(self):' in line:
            in_get_convergence_method = True
            # 修复缩进
            fixed_lines.append('    def get_convergence_metrics(self):\n')
            continue
        
        # 检测方法结束
        if in_train_method and line.strip().startswith('def ') and 'train' not in line:
            in_train_method = False
        
        if in_get_convergence_method and line.strip().startswith('def ') and 'get_convergence_metrics' not in line:
            in_get_convergence_method = False
        
        # 修复train方法内的缩进
        if in_train_method:
            # 如果行以8个或更多空格开始，减少到4个空格
            if line.startswith('        '):
                # 计算当前缩进级别
                indent_level = (len(line) - len(line.lstrip())) // 4
                if indent_level >= 2:
                    # 减少一个缩进级别
                    new_indent = '    ' * (indent_level - 1)
                    fixed_lines.append(new_indent + line.lstrip())
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # 修复get_convergence_metrics方法内的缩进
        elif in_get_convergence_method:
            # 如果行以8个或更多空格开始，减少到8个空格
            if line.startswith('        '):
                # 保持正确的方法内缩进
                fixed_lines.append(line)
            elif line.startswith('    ') and line.strip():
                # 确保方法内容有正确的缩进
                fixed_lines.append('        ' + line.lstrip())
            else:
                fixed_lines.append(line)
        
        else:
            fixed_lines.append(line)
    
    # 写回文件
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ main.py缩进修复完成")

if __name__ == "__main__":
    fix_main_py_indentation()