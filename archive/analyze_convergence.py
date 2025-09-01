# -*- coding: utf-8 -*-
"""
收敛性分析脚本
分析ZeroShotGNN网络的训练收敛性，识别NaN问题的根源
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from collections import defaultdict
import seaborn as sns

def analyze_training_history(history_file):
    """分析训练历史数据"""
    print(f"📊 分析训练历史: {history_file}")
    
    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    
    # 提取关键指标
    rewards = history.get('episode_rewards', [])
    losses = history.get('episode_losses', [])
    epsilon_values = history.get('epsilon_values', [])
    completion_rates = history.get('completion_rates', [])
    
    print(f"训练轮数: {len(rewards)}")
    print(f"奖励范围: [{min(rewards):.2f}, {max(rewards):.2f}]")
    
    # 分析损失中的NaN
    valid_losses = [l for l in losses if l is not None and not np.isnan(l) and not np.isinf(l)]
    nan_losses = len(losses) - len(valid_losses)
    
    print(f"有效损失: {len(valid_losses)}/{len(losses)} ({len(valid_losses)/len(losses)*100:.1f}%)")
    print(f"NaN/Inf损失: {nan_losses} ({nan_losses/len(losses)*100:.1f}%)")
    
    if valid_losses:
        print(f"损失范围: [{min(valid_losses):.6f}, {max(valid_losses):.6f}]")
        print(f"损失均值: {np.mean(valid_losses):.6f}")
        print(f"损失标准差: {np.std(valid_losses):.6f}")
    
    return {
        'rewards': rewards,
        'losses': losses,
        'valid_losses': valid_losses,
        'nan_loss_rate': nan_losses/len(losses) if losses else 0,
        'epsilon_values': epsilon_values,
        'completion_rates': completion_rates
    }

def detect_convergence_issues(analysis_data):
    """检测收敛性问题"""
    print("\n🔍 收敛性问题检测:")
    
    rewards = analysis_data['rewards']
    losses = analysis_data['valid_losses']
    nan_loss_rate = analysis_data['nan_loss_rate']
    
    issues = []
    
    # 1. 检查奖励趋势
    if len(rewards) > 50:
        early_rewards = np.mean(rewards[:25])
        late_rewards = np.mean(rewards[-25:])
        
        if late_rewards < early_rewards * 0.8:
            issues.append("⚠️ 奖励显著下降，可能存在灾难性遗忘")
        elif abs(late_rewards - early_rewards) < 0.1 * abs(early_rewards):
            issues.append("⚠️ 奖励停滞，可能陷入局部最优")
    
    # 2. 检查损失稳定性
    if nan_loss_rate > 0.1:
        issues.append(f"❌ 高NaN损失率 ({nan_loss_rate*100:.1f}%)，存在数值不稳定")
    
    if len(losses) > 20:
        recent_losses = losses[-20:]
        loss_variance = np.var(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        if loss_variance > loss_mean * 10:
            issues.append("⚠️ 损失方差过大，训练不稳定")
    
    # 3. 检查探索率衰减
    epsilon_values = analysis_data['epsilon_values']
    if len(epsilon_values) > 50:
        if epsilon_values[-1] > 0.5:
            issues.append("⚠️ 探索率衰减过慢，可能影响收敛")
        elif epsilon_values[-1] < 0.01:
            issues.append("⚠️ 探索率过低，可能陷入局部最优")
    
    # 4. 检查完成率
    completion_rates = analysis_data['completion_rates']
    if len(completion_rates) > 50:
        recent_completion = np.mean(completion_rates[-25:])
        if recent_completion < 0.3:
            issues.append("⚠️ 完成率过低，学习效果不佳")
    
    if not issues:
        print("✅ 未发现明显的收敛性问题")
    else:
        for issue in issues:
            print(f"  {issue}")
    
    return issues

def plot_convergence_analysis(analysis_data, save_path):
    """绘制收敛性分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ZeroShotGNN 收敛性分析', fontsize=16, fontweight='bold')
    
    rewards = analysis_data['rewards']
    losses = analysis_data['valid_losses']
    epsilon_values = analysis_data['epsilon_values']
    completion_rates = analysis_data['completion_rates']
    
    # 1. 奖励趋势
    axes[0, 0].plot(rewards, alpha=0.7, color='blue', linewidth=1)
    if len(rewards) > 20:
        # 添加移动平均
        window = min(20, len(rewards)//5)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, color='red', linewidth=2, label=f'移动平均({window})')
        axes[0, 0].legend()
    axes[0, 0].set_title('奖励收敛趋势')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 损失趋势
    if losses:
        axes[0, 1].plot(losses, alpha=0.7, color='orange', linewidth=1)
        if len(losses) > 20:
            window = min(20, len(losses)//5)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(losses)), moving_avg, color='red', linewidth=2, label=f'移动平均({window})')
            axes[0, 1].legend()
    axes[0, 1].set_title('损失收敛趋势')
    axes[0, 1].set_xlabel('Update Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 探索率衰减
    if epsilon_values:
        axes[0, 2].plot(epsilon_values, color='green', linewidth=2)
    axes[0, 2].set_title('探索率衰减')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Epsilon')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 完成率趋势
    if completion_rates:
        axes[1, 0].plot(completion_rates, alpha=0.7, color='purple', linewidth=1)
        if len(completion_rates) > 20:
            window = min(20, len(completion_rates)//5)
            moving_avg = np.convolve(completion_rates, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(completion_rates)), moving_avg, color='red', linewidth=2, label=f'移动平均({window})')
            axes[1, 0].legend()
    axes[1, 0].set_title('任务完成率')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Completion Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 奖励分布
    if len(rewards) > 10:
        axes[1, 1].hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(rewards):.2f}')
        axes[1, 1].axvline(np.median(rewards), color='orange', linestyle='--', linewidth=2, label=f'中位数: {np.median(rewards):.2f}')
        axes[1, 1].legend()
    axes[1, 1].set_title('奖励分布')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 损失分布
    if len(losses) > 10:
        axes[1, 2].hist(losses, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 2].axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(losses):.6f}')
        axes[1, 2].axvline(np.median(losses), color='orange', linestyle='--', linewidth=2, label=f'中位数: {np.median(losses):.6f}')
        axes[1, 2].legend()
    axes[1, 2].set_title('损失分布')
    axes[1, 2].set_xlabel('Loss')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 收敛性分析图已保存: {save_path}")
    plt.close()

def generate_convergence_report(analysis_data, issues, save_path):
    """生成收敛性报告"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("ZeroShotGNN 收敛性分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本统计
        f.write("📊 基本统计信息:\n")
        f.write(f"  训练轮数: {len(analysis_data['rewards'])}\n")
        f.write(f"  平均奖励: {np.mean(analysis_data['rewards']):.4f}\n")
        f.write(f"  奖励标准差: {np.std(analysis_data['rewards']):.4f}\n")
        f.write(f"  最终奖励: {analysis_data['rewards'][-1]:.4f}\n")
        
        if analysis_data['valid_losses']:
            f.write(f"  平均损失: {np.mean(analysis_data['valid_losses']):.6f}\n")
            f.write(f"  损失标准差: {np.std(analysis_data['valid_losses']):.6f}\n")
        
        f.write(f"  NaN损失率: {analysis_data['nan_loss_rate']*100:.2f}%\n")
        
        if analysis_data['completion_rates']:
            f.write(f"  平均完成率: {np.mean(analysis_data['completion_rates']):.4f}\n")
            f.write(f"  最终完成率: {analysis_data['completion_rates'][-1]:.4f}\n")
        
        # 收敛性问题
        f.write(f"\n🔍 收敛性问题 ({len(issues)}个):\n")
        if issues:
            for i, issue in enumerate(issues, 1):
                f.write(f"  {i}. {issue}\n")
        else:
            f.write("  ✅ 未发现明显问题\n")
        
        # 建议
        f.write(f"\n💡 优化建议:\n")
        if analysis_data['nan_loss_rate'] > 0.05:
            f.write("  - 降低学习率，增强数值稳定性\n")
            f.write("  - 检查奖励函数中的数学运算\n")
            f.write("  - 增加梯度裁剪强度\n")
        
        if len(analysis_data['rewards']) > 50:
            recent_trend = np.mean(analysis_data['rewards'][-25:]) - np.mean(analysis_data['rewards'][-50:-25])
            if recent_trend < 0:
                f.write("  - 考虑调整探索策略\n")
                f.write("  - 检查经验回放缓冲区\n")
        
        f.write(f"\n📅 报告生成时间: {np.datetime64('now')}\n")
    
    print(f"📄 收敛性报告已保存: {save_path}")

def main():
    """主函数"""
    print("🚀 开始收敛性分析...")
    
    # 查找最新的训练历史文件
    history_files = glob.glob("output/*/training_history_*.pkl")
    if not history_files:
        print("❌ 未找到训练历史文件")
        return
    
    # 选择最新的文件
    latest_file = max(history_files, key=os.path.getctime)
    print(f"📁 使用文件: {latest_file}")
    
    # 分析训练历史
    analysis_data = analyze_training_history(latest_file)
    
    # 检测收敛性问题
    issues = detect_convergence_issues(analysis_data)
    
    # 生成分析图
    base_name = os.path.splitext(os.path.basename(latest_file))[0]
    plot_path = f"convergence_analysis_{base_name}.png"
    plot_convergence_analysis(analysis_data, plot_path)
    
    # 生成报告
    report_path = f"convergence_report_{base_name}.txt"
    generate_convergence_report(analysis_data, issues, report_path)
    
    print("\n✅ 收敛性分析完成!")
    print(f"📈 分析图: {plot_path}")
    print(f"📄 报告: {report_path}")

if __name__ == "__main__":
    main()