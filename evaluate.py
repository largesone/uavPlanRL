# -*- coding: utf-8 -*-
# 文件名: evaluate.py
# 描述: 包含用于评估任务规划方案质量的函数。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')

# 全局变量存储各指标最优值
GLOBAL_BEST_METRICS = {
    'completion_rate': 0,
    'satisfied_targets_rate': 0,
    'sync_feasibility_rate': 0,
    'load_balance_score': 0,
    'resource_utilization_rate': 0,
    'total_distance': float('inf')  # 距离越小越好
}

def evaluate_plan(final_plan, uavs, targets, deadlocked_tasks=None, final_uav_states=None) -> dict:
    """评估一个最终方案的综合质量，返回包含各项指标的字典。"""
    # # 预处理任务数据补充resource_cost
    # for uav_id, tasks in final_plan.items():
    #     for task in tasks:
    #         if 'resource_cost' not in task:
    #             target = next((t for t in targets if t.id == task['target_id']), None)
    #             task['resource_cost'] = target.resources.copy() if target else np.zeros_like(uavs[0].initial_resources)

    is_deadlocked = 1 if deadlocked_tasks and any(deadlocked_tasks.values()) else 0
    deadlocked_uav_count = len([uav for uav, tasks in (deadlocked_tasks or {}).items() if tasks])

    if not final_plan or not any(final_plan.values()):
        return {
            'total_reward_score': -2000 if is_deadlocked else -1000, 'is_deadlocked': is_deadlocked,
            'deadlocked_uav_count': deadlocked_uav_count, 'completion_rate': 0,
            'satisfied_targets_count': 0, 
            'total_targets': len(targets),
            'satisfied_targets_rate': 0, 'resource_penalty': 1,
            'sync_feasibility_rate': 0, 'load_balance_score': 0, 'total_distance': 0,
            'resource_utilization_rate': 0, 'total_demand': '[]', 'total_contribution': '[]'
        }

    total_demand = np.sum([t.resources for t in targets], axis=0)
    total_demand_safe = np.maximum(total_demand, 1e-6)

    target_contributions = defaultdict(lambda: np.zeros_like(total_demand, dtype=float))
    all_tasks = [task for tasks in final_plan.values() for task in tasks]
    for task in all_tasks:
        # 添加调试信息
        if not isinstance(task, dict):
            print(f"[DEBUG] 警告: task不是字典类型，而是{type(task)}: {task}")
            continue
        target_id = task['target_id']
        target = next((t for t in targets if t.id == target_id), None)
        if target:
            resource_cost = task.get('resource_cost', target.resources) # 如果没有，则默认认为等于需求
        else:
            resource_cost = task.get('resource_cost', np.zeros(len(total_demand)))
        target_contributions[task['target_id']] += resource_cost

    #【修改后的代码】
    # 之前这里直接对 target_contributions 求和，在某些情况下可能导致重复计算
    # 现在改为直接使用 final_uav_states 计算总贡献，这是最准确的数据源
    if final_uav_states:
        total_initial = np.sum([np.array(s['initial_resources']) for s in final_uav_states], axis=0)
        total_final = np.sum([np.array(s['final_resources']) for s in final_uav_states], axis=0)
        total_contribution = total_initial - total_final
    else:
        # 后备方案：如果 final_uav_states 不可用，则使用原有逻辑
        total_contribution = np.sum(list(target_contributions.values()), axis=0)
    
    satisfied_targets_count = sum(1 for t in targets if np.all(target_contributions[t.id] >= t.resources - 1e-5))
    if not targets:
        satisfied_targets_rate = 1.0
    else:
        raw_rate = satisfied_targets_count / len(targets)
        if raw_rate == 1.0:
            satisfied_targets_rate = 1.0  # 全部完成，得满分
        else:
            # 部分完成时，使用平方来放大差距，鼓励完成更多目标
            satisfied_targets_rate = raw_rate ** 2

    total_demand_sum = np.sum(total_demand)
    total_contribution_sum = np.sum(np.minimum(total_contribution, total_demand))
    completion_rate = total_contribution_sum / total_demand_sum if total_demand_sum > 0 else 1.0    
    
    resource_penalty = np.mean(np.maximum(0, total_demand - total_contribution) / total_demand_safe)
    
    if not all_tasks:
        return {
            'total_reward_score': -500, 'is_deadlocked': is_deadlocked, 'deadlocked_uav_count': deadlocked_uav_count,
            'completion_rate': round(completion_rate, 4), 
            'satisfied_targets_count': satisfied_targets_count,
            'total_targets': len(targets),
            'satisfied_targets_rate': round(satisfied_targets_rate, 4), 'resource_penalty': round(resource_penalty, 4),
            'sync_feasibility_rate': 0, 'load_balance_score': 0, 'total_distance': 0, 'resource_utilization_rate': 0,
            'total_demand': np.array2string(total_demand, formatter={'float_kind': lambda x: "%.0f" % x}),
            'total_contribution': np.array2string(total_contribution, formatter={'float_kind': lambda x: "%.1f" % x}),
        }
    
    sync_feasibility_rate = sum(1 for task in all_tasks if task['is_sync_feasible']) / len(all_tasks)
    total_distance = sum(task['distance'] for task in all_tasks if 'distance' in task)

    # 修复资源利用率计算：使用推理完成后的UAV状态
    total_initial_resources = np.zeros(len(uavs[0].initial_resources), dtype=np.float64)
    total_consumed_resources = np.zeros(len(uavs[0].initial_resources), dtype=np.float64)
    
    if final_uav_states:
        # 使用推理结果中的UAV状态
        for uav_state in final_uav_states:
            initial_res = np.array(uav_state['initial_resources'], dtype=np.float64)
            final_res = np.array(uav_state['final_resources'], dtype=np.float64)
            consumed_res = initial_res - final_res
            
            total_initial_resources += initial_res
            total_consumed_resources += consumed_res
    else:
        # 使用当前UAV状态（向后兼容）
        for uav in uavs:
            initial_res = uav.initial_resources.astype(np.float64)
            current_res = uav.resources.astype(np.float64)
            consumed_res = initial_res - current_res
            
            total_initial_resources += initial_res
            total_consumed_resources += consumed_res
    
    # 计算资源利用率：已消耗资源 / 初始总资源
    total_initial_safe = np.maximum(total_initial_resources, 1e-6)
    resource_utilization_rate = np.mean(total_consumed_resources / total_initial_safe)
    
    # 计算负载均衡（基于资源消耗比例）
    uav_utilizations = []
    if final_uav_states:
        for uav_state in final_uav_states:
            initial_res = np.array(uav_state['initial_resources'], dtype=np.float64)
            final_res = np.array(uav_state['final_resources'], dtype=np.float64)
            consumed_res = initial_res - final_res
            
            uav_utilization = np.mean(consumed_res / np.maximum(initial_res, 1e-6))
            uav_utilizations.append(uav_utilization)
    else:
        for uav in uavs:
            initial_res = uav.initial_resources.astype(np.float64)
            current_res = uav.resources.astype(np.float64)
            consumed_res = initial_res - current_res
            
            uav_utilization = np.mean(consumed_res / np.maximum(initial_res, 1e-6))
            uav_utilizations.append(uav_utilization)
    
    load_balance_score = 1 / (1 + np.var(uav_utilizations)) if len(uav_utilizations) > 1 else 1.0

    # 归一化各项指标到0-1之间
    # 更新全局最优值
    GLOBAL_BEST_METRICS['completion_rate'] = max(GLOBAL_BEST_METRICS['completion_rate'], completion_rate)
    GLOBAL_BEST_METRICS['satisfied_targets_rate'] = max(GLOBAL_BEST_METRICS['satisfied_targets_rate'], satisfied_targets_rate)
    GLOBAL_BEST_METRICS['sync_feasibility_rate'] = max(GLOBAL_BEST_METRICS['sync_feasibility_rate'], sync_feasibility_rate)
    GLOBAL_BEST_METRICS['load_balance_score'] = max(GLOBAL_BEST_METRICS['load_balance_score'], load_balance_score)
    GLOBAL_BEST_METRICS['resource_utilization_rate'] = max(GLOBAL_BEST_METRICS['resource_utilization_rate'], resource_utilization_rate)
    GLOBAL_BEST_METRICS['total_distance'] = min(GLOBAL_BEST_METRICS['total_distance'], total_distance) if GLOBAL_BEST_METRICS['total_distance'] != float('inf') else total_distance

    # 相对归一化
    norm_completion = completion_rate / GLOBAL_BEST_METRICS['completion_rate'] if GLOBAL_BEST_METRICS['completion_rate'] > 0 else 0
    norm_satisfied = satisfied_targets_rate / GLOBAL_BEST_METRICS['satisfied_targets_rate'] if GLOBAL_BEST_METRICS['satisfied_targets_rate'] > 0 else 0
    norm_sync = sync_feasibility_rate / GLOBAL_BEST_METRICS['sync_feasibility_rate'] if GLOBAL_BEST_METRICS['sync_feasibility_rate'] > 0 else 0
    norm_load = load_balance_score / GLOBAL_BEST_METRICS['load_balance_score'] if GLOBAL_BEST_METRICS['load_balance_score'] > 0 else 0
    norm_utilization = resource_utilization_rate / GLOBAL_BEST_METRICS['resource_utilization_rate'] if GLOBAL_BEST_METRICS['resource_utilization_rate'] > 0 else 0
    norm_distance = 1 - (total_distance / GLOBAL_BEST_METRICS['total_distance']) if GLOBAL_BEST_METRICS['total_distance'] > 0 else 0

    # 计算总评分（归一化后的加权和）
    total_reward_score = (
        norm_completion * 5.0 + 
        norm_satisfied * 3.0 - 
        resource_penalty * 2.0 +
        norm_sync * 1.0 + 
        norm_load * 0.5 + 
        norm_utilization * 0.5 -
        (1 - norm_distance) * 0.1
    ) * 100  # 乘以100使得分数在合理范围内
    
    if is_deadlocked:
        total_reward_score -= 10 * deadlocked_uav_count

    return {
        'total_reward_score': round(total_reward_score, 2), 'is_deadlocked': is_deadlocked,
        'deadlocked_uav_count': deadlocked_uav_count, 'completion_rate': round(completion_rate, 4),
        'satisfied_targets_count': satisfied_targets_count, 
        'total_targets': len(targets),
        'satisfied_targets_rate': round(satisfied_targets_rate, 4),
        'satisfied_targets_rule': '全满足=1 | 全不满足=0 | 部分满足=平方比例',
        'resource_utilization_rate': round(resource_utilization_rate, 4), 'resource_penalty': round(resource_penalty, 4),
        'sync_feasibility_rate': round(sync_feasibility_rate, 4), 'load_balance_score': round(load_balance_score, 4),
        'total_distance': round(total_distance, 2),
        'total_demand': np.array2string(total_demand, formatter={'float_kind': lambda x: "%.0f" % x}),
        'total_contribution': np.array2string(total_contribution, formatter={'float_kind': lambda x: "%.1f" % x})
    }

def evaluate_score_v3(row):
    """
    根据evaluate.py中的奖励函数设计，计算综合评估分数 (V3版本)。
    
    该评估函数与evaluate.py中的total_reward_score计算逻辑保持一致：
    总评分 = (完成率×5.0 + 目标满足率×3.0 - 资源惩罚×2.0 + 
             同步可行率×1.0 + 负载均衡×0.5 + 资源利用率×0.5 - 
             归一化路径长度×0.1) × 100 - 死锁无人机数×10
    """
    score = 0.0

    # 1. 完成率 (completion_rate) - 权重 5.0
    # 任务完成的百分比，非常重要
    completion_score = row['completion_rate'] * 5.0
    score += completion_score
    
    # 2. 目标满足率 (satisfied_targets_rate) - 权重 3.0
    # 达成目标的比率，与完成率类似，但更侧重于目标本身
    satisfied_score = row['satisfied_targets_rate'] * 3.0
    score += satisfied_score
    
    # 3. 资源惩罚 (resource_penalty) - 权重 -2.0 (负向)
    # 未满足资源需求造成的惩罚
    resource_penalty = row['resource_penalty'] * 2.0
    score -= resource_penalty
    
    # 4. 同步可行率 (sync_feasibility_rate) - 权重 1.0
    # 协同任务中同步完成的可能性
    sync_score = row['sync_feasibility_rate'] * 1.0
    score += sync_score
    
    # 5. 负载均衡得分 (load_balance_score) - 权重 0.5
    # 衡量任务分配的均衡性
    balance_score = row['load_balance_score'] * 0.5
    score += balance_score
    
    # 6. 资源利用率 (resource_utilization_rate) - 权重 0.5
    # 资源有效利用的程度
    utilization_score = row['resource_utilization_rate'] * 0.5
    score += utilization_score
    
    # 7. 路径总长度惩罚 - 权重 -0.1 (负向)
    # 路径越长，成本越高，转换为惩罚项
    # 使用相对惩罚，避免绝对值过大
    if row['total_distance'] > 0:
        # 归一化路径长度惩罚，假设最大路径长度为100000
        normalized_distance = min(row['total_distance'] / 100000.0, 1.0)
        distance_penalty = normalized_distance * 0.1
        score -= distance_penalty
    
    # 8. 死锁惩罚 - 权重 -10 (负向)
    # 死锁是严重问题，直接给予高惩罚
    if row['is_deadlocked'] == 1:
        score -= 10 * row['deadlocked_uav_count']
    
    # 9. 规划时间惩罚 - 额外考虑
    # 规划时间越长，效率越低，但权重较小
    if 'planning_time' in row and row['planning_time'] > 0:
        time_penalty = min(row['planning_time'] / 100.0, 1.0) * 0.5  # 最大0.5分惩罚
        score -= time_penalty

    return score

def calculate_detailed_analysis(df):
    """
    对评估结果进行详细分析，生成各项指标的统计信息
    """
    analysis = {}
    
    # 基础统计
    analysis['total_tests'] = len(df)
    analysis['successful_tests'] = len(df[df['total_reward_score'] > 0])
    analysis['failed_tests'] = len(df[df['total_reward_score'] <= 0])
    
    # 各算法性能统计
    if 'solver' in df.columns:
        solver_stats = df.groupby('solver').agg({
            'total_reward_score': ['mean', 'std', 'min', 'max'],
            'completion_rate': ['mean', 'max'],
            'satisfied_targets_rate': ['mean', 'max'],
            'resource_utilization_rate': ['mean', 'max'],
            'load_balance_score': ['mean', 'max'],
            'total_distance': ['mean', 'min'],
            'planning_time': ['mean', 'min'],
            'is_deadlocked': ['sum']
        }).round(4)
        analysis['solver_performance'] = solver_stats
    
    # 场景难度分析
    if 'scenario' in df.columns:
        scenario_stats = df.groupby('scenario').agg({
            'total_reward_score': ['mean', 'std'],
            'completion_rate': ['mean', 'min'],
            'is_deadlocked': ['sum']
        }).round(4)
        analysis['scenario_difficulty'] = scenario_stats
    
    # 障碍物影响分析
    if 'obstacle_mode' in df.columns:
        obstacle_stats = df.groupby('obstacle_mode').agg({
            'total_reward_score': ['mean', 'std'],
            'completion_rate': ['mean'],
            'total_distance': ['mean'],
            'is_deadlocked': ['sum']
        }).round(4)
        analysis['obstacle_impact'] = obstacle_stats
    
    return analysis

def generate_evaluation_report(df, output_dir):
    """
    生成详细的评估报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算新的评估分数
    df['evaluation_score_v3'] = df.apply(evaluate_score_v3, axis=1)
    
    # 详细分析
    analysis = calculate_detailed_analysis(df)
    
    # 生成报告
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("UAV任务规划算法综合评估报告")
    report_lines.append("="*60)
    report_lines.append("")
    
    # 1. 总体统计
    report_lines.append("1. 总体统计")
    report_lines.append("-"*30)
    report_lines.append(f"总测试数: {analysis['total_tests']}")
    report_lines.append(f"成功测试数: {analysis['successful_tests']}")
    report_lines.append(f"失败测试数: {analysis['failed_tests']}")
    report_lines.append(f"成功率: {analysis['successful_tests']/analysis['total_tests']*100:.2f}%")
    report_lines.append("")
    
    # 2. 算法性能排名
    if 'solver_performance' in analysis:
        report_lines.append("2. 算法性能排名")
        report_lines.append("-"*30)
        solver_scores = analysis['solver_performance'][('total_reward_score', 'mean')].sort_values(ascending=False)
        for i, (solver, score) in enumerate(solver_scores.items(), 1):
            report_lines.append(f"{i}. {solver}: {score:.2f}")
        report_lines.append("")
    
    # 3. 单项指标分析
    report_lines.append("3. 单项指标分析")
    report_lines.append("-"*30)
    
    # 完成率分析
    completion_stats = df['completion_rate'].describe()
    report_lines.append("完成率分析:")
    report_lines.append(f"  平均完成率: {completion_stats['mean']:.4f}")
    report_lines.append(f"  最高完成率: {completion_stats['max']:.4f}")
    report_lines.append(f"  最低完成率: {completion_stats['min']:.4f}")
    report_lines.append("")
    
    # 目标满足率分析
    satisfied_stats = df['satisfied_targets_rate'].describe()
    report_lines.append("目标满足率分析:")
    report_lines.append(f"  平均满足率: {satisfied_stats['mean']:.4f}")
    report_lines.append(f"  最高满足率: {satisfied_stats['max']:.4f}")
    report_lines.append(f"  最低满足率: {satisfied_stats['min']:.4f}")
    report_lines.append("")
    
    # 资源利用率分析
    utilization_stats = df['resource_utilization_rate'].describe()
    report_lines.append("资源利用率分析:")
    report_lines.append(f"  平均利用率: {utilization_stats['mean']:.4f}")
    report_lines.append(f"  最高利用率: {utilization_stats['max']:.4f}")
    report_lines.append(f"  最低利用率: {utilization_stats['min']:.4f}")
    report_lines.append("")
    
    # 任务分配均衡性分析
    balance_stats = df['load_balance_score'].describe()
    report_lines.append("任务分配均衡性分析:")
    report_lines.append(f"  平均均衡性: {balance_stats['mean']:.4f}")
    report_lines.append(f"  最高均衡性: {balance_stats['max']:.4f}")
    report_lines.append(f"  最低均衡性: {balance_stats['min']:.4f}")
    report_lines.append("")
    
    # 路径长度分析
    distance_stats = df['total_distance'].describe()
    report_lines.append("路径长度分析:")
    report_lines.append(f"  平均路径长度: {distance_stats['mean']:.2f}")
    report_lines.append(f"  最短路径长度: {distance_stats['min']:.2f}")
    report_lines.append(f"  最长路径长度: {distance_stats['max']:.2f}")
    report_lines.append("")
    
    # 4. 场景难度分析
    if 'scenario_difficulty' in analysis:
        report_lines.append("4. 场景难度分析")
        report_lines.append("-"*30)
        for scenario, stats in analysis['scenario_difficulty'].iterrows():
            avg_score = stats[('total_reward_score', 'mean')]
            avg_completion = stats[('completion_rate', 'mean')]
            deadlock_count = stats[('is_deadlocked', 'sum')]
            report_lines.append(f"{scenario}:")
            report_lines.append(f"  平均评分: {avg_score:.2f}")
            report_lines.append(f"  平均完成率: {avg_completion:.4f}")
            report_lines.append(f"  死锁次数: {deadlock_count}")
            report_lines.append("")
    
    # 5. 障碍物影响分析
    if 'obstacle_impact' in analysis:
        report_lines.append("5. 障碍物影响分析")
        report_lines.append("-"*30)
        for mode, stats in analysis['obstacle_impact'].iterrows():
            avg_score = stats[('total_reward_score', 'mean')]
            avg_completion = stats[('completion_rate', 'mean')]
            avg_distance = stats[('total_distance', 'mean')]
            report_lines.append(f"{mode}:")
            report_lines.append(f"  平均评分: {avg_score:.2f}")
            report_lines.append(f"  平均完成率: {avg_completion:.4f}")
            report_lines.append(f"  平均路径长度: {avg_distance:.2f}")
            report_lines.append("")
    
    # 保存报告
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"评估报告已保存到: {report_path}")
    return report_path

def create_evaluation_visualizations(df, output_dir):
    """
    创建评估结果的可视化图表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图表样式
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('UAV任务规划算法评估结果可视化', fontsize=16, fontweight='bold')
    
    # 1. 算法性能对比（箱线图）
    if 'solver' in df.columns:
        df.boxplot(column='total_reward_score', by='solver', ax=axes[0,0])
        axes[0,0].set_title('算法性能对比')
        axes[0,0].set_xlabel('算法')
        axes[0,0].set_ylabel('总评分')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. 完成率分布
    df['completion_rate'].hist(bins=20, ax=axes[0,1], alpha=0.7, color='skyblue')
    axes[0,1].set_title('完成率分布')
    axes[0,1].set_xlabel('完成率')
    axes[0,1].set_ylabel('频次')
    
    # 3. 目标满足率分布
    df['satisfied_targets_rate'].hist(bins=20, ax=axes[0,2], alpha=0.7, color='lightgreen')
    axes[0,2].set_title('目标满足率分布')
    axes[0,2].set_xlabel('目标满足率')
    axes[0,2].set_ylabel('频次')
    
    # 4. 资源利用率分布
    df['resource_utilization_rate'].hist(bins=20, ax=axes[1,0], alpha=0.7, color='orange')
    axes[1,0].set_title('资源利用率分布')
    axes[1,0].set_xlabel('资源利用率')
    axes[1,0].set_ylabel('频次')
    
    # 5. 路径长度分布
    df['total_distance'].hist(bins=20, ax=axes[1,1], alpha=0.7, color='red')
    axes[1,1].set_title('路径长度分布')
    axes[1,1].set_xlabel('路径长度')
    axes[1,1].set_ylabel('频次')
    
    # 6. 负载均衡得分分布
    df['load_balance_score'].hist(bins=20, ax=axes[1,2], alpha=0.7, color='purple')
    axes[1,2].set_title('负载均衡得分分布')
    axes[1,2].set_xlabel('负载均衡得分')
    axes[1,2].set_ylabel('频次')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'evaluation_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"评估可视化图表已保存到: {plot_path}")
    return plot_path

def calculate_new_scores_for_csv_v3(input_csv_path, output_csv_path, output_dir="output/evaluation_analysis"):
    """
    为CSV文件中的结果计算新的评估分数并生成分析报告
    
    Args:
        input_csv_path: 输入CSV文件路径
        output_csv_path: 输出CSV文件路径
        output_dir: 输出目录
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_csv_path)
        print(f"成功读取CSV文件: {input_csv_path}")
        print(f"数据行数: {len(df)}")
        print(f"数据列: {list(df.columns)}")
        
        # 计算新的评估分数
        df['evaluation_score_v3'] = df.apply(evaluate_score_v3, axis=1)
        
        # 保存更新后的CSV文件
        df.to_csv(output_csv_path, index=False)
        print(f"更新后的CSV文件已保存到: {output_csv_path}")
        
        # 生成评估报告
        report_path = generate_evaluation_report(df, output_dir)
        
        # 创建可视化图表
        plot_path = create_evaluation_visualizations(df, output_dir)
        
        print("="*60)
        print("评估分析完成!")
        print(f"更新后的CSV文件: {output_csv_path}")
        print(f"评估报告: {report_path}")
        print(f"可视化图表: {plot_path}")
        print("="*60)
        
        return df
        
    except Exception as e:
        print(f"处理CSV文件时出错: {e}")
        return None