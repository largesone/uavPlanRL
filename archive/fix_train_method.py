#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确修复train方法的缩进问题
"""

def fix_train_method():
    """修复train方法的缩进"""
    
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到train方法的开始和结束
    train_start = content.find('def train(self, episodes, patience, log_interval, model_save_path):')
    if train_start == -1:
        print("❌ 找不到train方法")
        return False
    
    # 找到下一个方法的开始（train方法的结束）
    next_method_start = content.find('\n    def ', train_start + 1)
    if next_method_start == -1:
        # 如果没有找到下一个方法，找到类的结束
        next_method_start = len(content)
    
    # 提取train方法的内容
    train_method = content[train_start:next_method_start]
    
    # 修复缩进：将所有8个空格的缩进改为4个空格（相对于方法定义）
    lines = train_method.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if i == 0:  # 方法定义行
            fixed_lines.append(line)
        elif line.strip() == '':  # 空行
            fixed_lines.append(line)
        elif line.startswith('        """') or '"""' in line:  # docstring
            fixed_lines.append(line)
        elif line.startswith('                '):  # 16个空格 -> 12个空格
            fixed_lines.append('            ' + line[16:])
        elif line.startswith('            '):  # 12个空格 -> 8个空格
            fixed_lines.append('        ' + line[12:])
        elif line.startswith('        '):  # 8个空格保持不变
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # 重新组合内容
    fixed_train_method = '\n'.join(fixed_lines)
    new_content = content[:train_start] + fixed_train_method + content[next_method_start:]
    
    # 写回文件
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ train方法缩进修复完成")
    return True

if __name__ == "__main__":
    fix_train_method()