# -*- coding: utf-8 -*-
# 文件名: solution_evaluator.py
# 描述: 方案评估系统，使用已有评估代码对算法结果进行评价

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict
import time

from entities import UAV, Target
from evaluate import evaluate_plan
from scenarios import get_small_scenario, get_complex_scenario
from config import Config

class SolutionEvaluator:
    """方案评估系统"""
    
    def __init__(self):
        self.evaluation_results = []
        self.algorithm_metrics = defaultdict(list)
    
    def evaluate_single_solution(self, plan: Dict, uavs: List[UAV], targets: List[Target], 
                                algorithm_name: str, execution_time: float) -> Dict:
        """评估单个解决方案"""
        
        # 使用已有的评估函数
        quality_metrics = evaluate_plan(plan, uavs, targets, {})
        
        # 计算额外指标
        total_assignments = sum(len(plan.get(uav.id, {}).get('targets', [])) for uav in uavs)
        completion_rate = total_assignments / len(targets) if len(targets) > 0 else 0
        
        # 计算总距离
        total_distance = 0
        for uav in uavs:
            if uav.id in plan:
                uav_plan = plan[uav.id]
                for target_id, phi_idx in uav_plan.get('targets', []):
                    target = next(t for t in targets if t.id == target_id)
                    distance = np.linalg.norm(np.array(uav.position) - np.array(target.position))
                    total_distance += distance
        
        # 计算负载均衡性
        assignment_counts = [len(plan.get(uav.id, {}).get('targets', [])) for uav in uavs]
        if len(assignment_counts) > 1:
            load_balance = 1.0 - np.std(assignment_counts) / (np.mean(assignment_counts) + 1e-6)
        else:
            load_balance = 1.0
        
        # 综合评分
        comprehensive_score = (
            quality_metrics['completion_rate'] * 0.3 +
            quality_metrics['satisfied_targets_rate'] * 0.2 +
            quality_metrics['resource_utilization_rate'] * 0.2 +
            load_balance * 0.15 +
            (1.0 / (total_distance + 1e-6)) * 0.15
        )
        
        evaluation_result = {
            'algorithm': algorithm_name,
            'execution_time': execution_time,
            'completion_rate': completion_rate,
            'total_assignments': total_assignments,
            'total_distance': total_distance,
            'load_balance': load_balance,
            'comprehensive_score': comprehensive_score,
            **quality_metrics
        }
        
        return evaluation_result
    
    def evaluate_multiple_algorithms(self, algorithms_results: List[Tuple], 
                                   uavs: List[UAV], targets: List[Target]) -> pd.DataFrame:
        """评估多个算法的结果"""
        
        results = []
        
        for algorithm_name, plan, execution_time in algorithms_results:
            try:
                evaluation = self.evaluate_single_solution(plan, uavs, targets, algorithm_name, execution_time)
                results.append(evaluation)
                print(f"✓ {algorithm_name} 评估完成")
            except Exception as e:
                print(f"✗ {algorithm_name} 评估失败: {e}")
                results.append({
                    'algorithm': algorithm_name,
                    'execution_time': execution_time,
                    'completion_rate': 0,
                    'total_assignments': 0,
                    'total_distance': 0,
                    'load_balance': 0,
                    'comprehensive_score': 0,
                    'total_reward_score': -1000,
                    'completion_rate': 0,
                    'satisfied_targets_rate': 0,
                    'resource_utilization_rate': 0,
                    'load_balance_score': 0,
                    'total_distance': 0,
                    'is_deadlocked': 1,
                    'deadlocked_uav_count': len(uavs),
                    'sync_feasibility_rate': 0,
                    'resource_penalty': 1
                })
        
        return pd.DataFrame(results)
    
    def generate_evaluation_report(self, df: pd.DataFrame, output_dir: str = "output/evaluation"):
        """生成评估报告"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 算法性能排名
        df_sorted = df.sort_values('comprehensive_score', ascending=False)
        
        report = f"""
============================================================
算法方案评估报告
============================================================

1. 算法性能排名
------------------------------
"""
        
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            report += f"{i}. {row['algorithm']}: {row['comprehensive_score']:.3f}\n"
        
        report += f"""
2. 详细指标对比
------------------------------
"""
        
        # 计算统计信息
        for metric in ['completion_rate', 'total_distance', 'execution_time', 'load_balance']:
            if metric in df.columns:
                report += f"{metric}:\n"
                report += f"  平均值: {df[metric].mean():.3f}\n"
                report += f"  最高值: {df[metric].max():.3f}\n"
                report += f"  最低值: {df[metric].min():.3f}\n"
                report += f"  标准差: {df[metric].std():.3f}\n\n"
        
        # 3. 算法特点分析
        report += f"""
3. 算法特点分析
------------------------------
"""
        
        for _, row in df.iterrows():
            report += f"{row['algorithm']}:\n"
            report += f"  完成率: {row['completion_rate']:.2%}\n"
            report += f"  总距离: {row['total_distance']:.2f}\n"
            report += f"  执行时间: {row['execution_time']:.2f}s\n"
            report += f"  负载均衡: {row['load_balance']:.3f}\n"
            report += f"  综合评分: {row['comprehensive_score']:.3f}\n\n"
        
        # 保存报告
        with open(f"{output_dir}/evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        # 生成可视化
        self._generate_evaluation_plots(df, output_dir)
        
        print(f"评估报告已保存到: {output_dir}")
        return report
    
    def _generate_evaluation_plots(self, df: pd.DataFrame, output_dir: str):
        """生成评估图表"""
        plt.style.use('default')
        
        # 1. 综合评分对比
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['algorithm'], df['comprehensive_score'])
        plt.title('算法综合评分对比', fontsize=14)
        plt.ylabel('综合评分', fontsize=12)
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, df['comprehensive_score']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comprehensive_score_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 多指标雷达图
        metrics = ['completion_rate', 'load_balance', 'resource_utilization_rate', 'satisfied_targets_rate']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for _, row in df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['algorithm'])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('算法多指标对比', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/multi_metric_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 执行时间vs完成率散点图
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:len(df)]
        
        for i, (_, row) in enumerate(df.iterrows()):
            plt.scatter(row['execution_time'], row['completion_rate'], 
                       s=100, c=[colors[i]], label=row['algorithm'], alpha=0.7)
        
        plt.xlabel('执行时间 (s)', fontsize=12)
        plt.ylabel('完成率', fontsize=12)
        plt.title('执行时间 vs 完成率', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_vs_completion.png", dpi=300, bbox_inches='tight')
        plt.close()

def test_evaluation_system():
    """测试评估系统"""
    print("=" * 60)
    print("测试方案评估系统")
    print("=" * 60)
    
    # 获取测试场景
    uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
    config = Config()
    
    # 导入算法
    from PSOSolver import PSOSolver
    from ACOSolver import ACOSolver
    from CBBASolver import CBBASolver
    
    # 测试算法列表
    algorithms = [
        ("PSO", PSOSolver),
        ("ACO", ACOSolver),
        ("CBBA", CBBASolver)
    ]
    
    # 运行算法并收集结果
    algorithms_results = []
    
    for alg_name, alg_class in algorithms:
        print(f"\n运行 {alg_name} 算法...")
        try:
            solver = alg_class(uavs, targets, obstacles, config)
            start_time = time.time()
            plan, training_time, planning_time = solver.solve()
            execution_time = time.time() - start_time
            
            algorithms_results.append((alg_name, plan, execution_time))
            print(f"✓ {alg_name} 完成，耗时: {execution_time:.2f}s")
            
        except Exception as e:
            print(f"✗ {alg_name} 失败: {e}")
            algorithms_results.append((alg_name, {}, 0.0))
    
    # 评估结果
    evaluator = SolutionEvaluator()
    df = evaluator.evaluate_multiple_algorithms(algorithms_results, uavs, targets)
    
    # 生成报告
    report = evaluator.generate_evaluation_report(df)
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    
    return df, report

if __name__ == "__main__":
    df, report = test_evaluation_system()
    print(report) 