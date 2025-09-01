# -*- coding: utf-8 -*-
# 文件名: scenario_viewer.py
# 描述: 场景记录查看工具 - 用于查看和分析动态训练生成的场景数据

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse

class ScenarioViewer:
    """场景记录查看器"""
    
    def __init__(self, scenario_log_dir: str = "output/scenario_logs"):
        self.scenario_log_dir = scenario_log_dir
        self.scenario_files = []
        self.scenario_data = []
        
        if os.path.exists(scenario_log_dir):
            self.load_scenario_files()
        else:
            print(f"场景记录目录不存在: {scenario_log_dir}")
    
    def load_scenario_files(self):
        """加载场景文件列表"""
        self.scenario_files = []
        for filename in os.listdir(self.scenario_log_dir):
            if filename.endswith('.pkl'):
                self.scenario_files.append(filename)
        
        self.scenario_files.sort()  # 按文件名排序
        print(f"发现 {len(self.scenario_files)} 个场景文件")
    
    def load_scenario_data(self, filename: str) -> Optional[Dict]:
        """加载单个场景文件数据"""
        filepath = os.path.join(self.scenario_log_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"加载场景文件失败 {filename}: {e}")
            return None
    
    def load_all_scenario_data(self):
        """加载所有场景数据"""
        self.scenario_data = []
        for filename in self.scenario_files:
            data = self.load_scenario_data(filename)
            if data:
                self.scenario_data.append(data)
        
        print(f"成功加载 {len(self.scenario_data)} 个场景数据")
    
    def get_scenario_summary(self) -> Dict:
        """获取场景数据摘要"""
        if not self.scenario_data:
            self.load_all_scenario_data()
        
        if not self.scenario_data:
            return {}
        
        # 统计信息
        episodes = [data['episode'] for data in self.scenario_data]
        uav_counts = [data['uav_count'] for data in self.scenario_data]
        target_counts = [data['target_count'] for data in self.scenario_data]
        obstacle_counts = [data['obstacle_count'] for data in self.scenario_data]
        scenario_names = [data['scenario_name'] for data in self.scenario_data]
        
        # 场景类型统计
        scenario_type_counts = {}
        for name in scenario_names:
            scenario_type_counts[name] = scenario_type_counts.get(name, 0) + 1
        
        return {
            'total_scenarios': len(self.scenario_data),
            'episode_range': (min(episodes), max(episodes)),
            'scenario_types': scenario_type_counts,
            'uav_stats': {
                'min': min(uav_counts),
                'max': max(uav_counts),
                'avg': np.mean(uav_counts),
                'std': np.std(uav_counts)
            },
            'target_stats': {
                'min': min(target_counts),
                'max': max(target_counts),
                'avg': np.mean(target_counts),
                'std': np.std(target_counts)
            },
            'obstacle_stats': {
                'min': min(obstacle_counts),
                'max': max(obstacle_counts),
                'avg': np.mean(obstacle_counts),
                'std': np.std(obstacle_counts)
            }
        }
    
    def print_scenario_summary(self):
        """打印场景摘要"""
        summary = self.get_scenario_summary()
        if not summary:
            print("暂无场景数据")
            return
        
        print("\n" + "="*60)
        print("动态训练场景数据摘要")
        print("="*60)
        print(f"总场景数: {summary['total_scenarios']}")
        print(f"轮次范围: {summary['episode_range'][0]} - {summary['episode_range'][1]}")
        
        print(f"\n场景类型分布:")
        for scenario_type, count in summary['scenario_types'].items():
            print(f"  {scenario_type}: {count}个")
        
        print(f"\nUAV数量统计:")
        uav_stats = summary['uav_stats']
        print(f"  最小: {uav_stats['min']}, 最大: {uav_stats['max']}")
        print(f"  平均: {uav_stats['avg']:.1f} ± {uav_stats['std']:.1f}")
        
        print(f"\n目标数量统计:")
        target_stats = summary['target_stats']
        print(f"  最小: {target_stats['min']}, 最大: {target_stats['max']}")
        print(f"  平均: {target_stats['avg']:.1f} ± {target_stats['std']:.1f}")
        
        print(f"\n障碍物数量统计:")
        obstacle_stats = summary['obstacle_stats']
        print(f"  最小: {obstacle_stats['min']}, 最大: {obstacle_stats['max']}")
        print(f"  平均: {obstacle_stats['avg']:.1f} ± {obstacle_stats['std']:.1f}")
        
        print("="*60)
    
    def print_scenario_details(self, episode: int = None, filename: str = None):
        """打印特定场景的详细信息"""
        if not self.scenario_data:
            self.load_all_scenario_data()
        
        target_data = None
        
        if episode is not None:
            # 按轮次查找
            for data in self.scenario_data:
                if data['episode'] == episode:
                    target_data = data
                    break
        elif filename is not None:
            # 按文件名查找
            target_data = self.load_scenario_data(filename)
        
        if not target_data:
            print(f"未找到指定的场景数据")
            return
        
        print(f"\n场景详细信息:")
        print(f"轮次: {target_data['episode']}")
        print(f"场景名称: {target_data['scenario_name']}")
        print(f"时间戳: {target_data['timestamp']}")
        print(f"UAV数量: {target_data['uav_count']}")
        print(f"目标数量: {target_data['target_count']}")
        print(f"障碍物数量: {target_data['obstacle_count']}")
        
        print(f"\n配置信息:")
        config_info = target_data['config_info']
        for key, value in config_info.items():
            print(f"  {key}: {value}")
        
        # 打印UAV详细信息
        print(f"\nUAV详细信息:")
        for i, uav in enumerate(target_data['uavs']):
            print(f"  UAV {i+1}: 位置{uav.position}, 资源{uav.resources}, 最大距离{uav.max_distance}")
        
        # 打印目标详细信息
        print(f"\n目标详细信息:")
        for i, target in enumerate(target_data['targets']):
            print(f"  目标 {i+1}: 位置{target.position}, 需求{target.resources}, 价值{target.value}")
    
    def plot_scenario_evolution(self, save_path: str = None):
        """绘制场景演化图"""
        if not self.scenario_data:
            self.load_all_scenario_data()
        
        if not self.scenario_data:
            print("暂无场景数据可绘制")
            return
        
        # 准备数据
        episodes = [data['episode'] for data in self.scenario_data]
        uav_counts = [data['uav_count'] for data in self.scenario_data]
        target_counts = [data['target_count'] for data in self.scenario_data]
        obstacle_counts = [data['obstacle_count'] for data in self.scenario_data]
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # UAV数量演化
        ax1.plot(episodes, uav_counts, 'b-o', markersize=4)
        ax1.set_ylabel('UAV数量')
        ax1.set_title('UAV数量演化')
        ax1.grid(True, alpha=0.3)
        
        # 目标数量演化
        ax2.plot(episodes, target_counts, 'r-s', markersize=4)
        ax2.set_ylabel('目标数量')
        ax2.set_title('目标数量演化')
        ax2.grid(True, alpha=0.3)
        
        # 障碍物数量演化
        ax3.plot(episodes, obstacle_counts, 'g-^', markersize=4)
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('障碍物数量')
        ax3.set_title('障碍物数量演化')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_scenario_distribution(self, save_path: str = None):
        """绘制场景分布图"""
        if not self.scenario_data:
            self.load_all_scenario_data()
        
        if not self.scenario_data:
            print("暂无场景数据可绘制")
            return
        
        # 准备数据
        uav_counts = [data['uav_count'] for data in self.scenario_data]
        target_counts = [data['target_count'] for data in self.scenario_data]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # UAV数量分布
        ax1.hist(uav_counts, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('UAV数量')
        ax1.set_ylabel('频次')
        ax1.set_title('UAV数量分布')
        ax1.grid(True, alpha=0.3)
        
        # 目标数量分布
        ax2.hist(target_counts, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('目标数量')
        ax2.set_ylabel('频次')
        ax2.set_title('目标数量分布')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
    
    def export_scenario_report(self, output_file: str = "scenario_report.txt"):
        """导出场景报告"""
        if not self.scenario_data:
            self.load_all_scenario_data()
        
        if not self.scenario_data:
            print("暂无场景数据可导出")
            return
        
        summary = self.get_scenario_summary()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("动态训练场景数据报告\n")
            f.write("="*50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总场景数: {summary['total_scenarios']}\n")
            f.write(f"轮次范围: {summary['episode_range'][0]} - {summary['episode_range'][1]}\n\n")
            
            f.write("场景类型分布:\n")
            for scenario_type, count in summary['scenario_types'].items():
                f.write(f"  {scenario_type}: {count}个\n")
            f.write("\n")
            
            f.write("UAV数量统计:\n")
            uav_stats = summary['uav_stats']
            f.write(f"  最小: {uav_stats['min']}, 最大: {uav_stats['max']}\n")
            f.write(f"  平均: {uav_stats['avg']:.1f} ± {uav_stats['std']:.1f}\n\n")
            
            f.write("目标数量统计:\n")
            target_stats = summary['target_stats']
            f.write(f"  最小: {target_stats['min']}, 最大: {target_stats['max']}\n")
            f.write(f"  平均: {target_stats['avg']:.1f} ± {target_stats['std']:.1f}\n\n")
            
            f.write("障碍物数量统计:\n")
            obstacle_stats = summary['obstacle_stats']
            f.write(f"  最小: {obstacle_stats['min']}, 最大: {obstacle_stats['max']}\n")
            f.write(f"  平均: {obstacle_stats['avg']:.1f} ± {obstacle_stats['std']:.1f}\n\n")
            
            f.write("详细场景列表:\n")
            f.write("-"*50 + "\n")
            for data in self.scenario_data:
                f.write(f"Episode {data['episode']:4d}: {data['scenario_name']} "
                       f"({data['uav_count']}UAV, {data['target_count']}目标, {data['obstacle_count']}障碍物)\n")
        
        print(f"场景报告已导出到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='场景记录查看工具')
    parser.add_argument('--dir', default='output/scenario_logs', help='场景记录目录')
    parser.add_argument('--summary', action='store_true', help='显示场景摘要')
    parser.add_argument('--episode', type=int, help='查看特定轮次的场景详情')
    parser.add_argument('--file', help='查看特定文件的场景详情')
    parser.add_argument('--evolution', action='store_true', help='绘制场景演化图')
    parser.add_argument('--distribution', action='store_true', help='绘制场景分布图')
    parser.add_argument('--export', help='导出场景报告到指定文件')
    parser.add_argument('--save-plot', help='保存图表到指定文件')
    
    args = parser.parse_args()
    
    viewer = ScenarioViewer(args.dir)
    
    if args.summary:
        viewer.print_scenario_summary()
    
    if args.episode is not None:
        viewer.print_scenario_details(episode=args.episode)
    
    if args.file is not None:
        viewer.print_scenario_details(filename=args.file)
    
    if args.evolution:
        viewer.plot_scenario_evolution(save_path=args.save_plot)
    
    if args.distribution:
        viewer.plot_scenario_distribution(save_path=args.save_plot)
    
    if args.export:
        viewer.export_scenario_report(args.export)
    
    # 如果没有指定任何操作，默认显示摘要
    if not any([args.summary, args.episode, args.file, args.evolution, args.distribution, args.export]):
        viewer.print_scenario_summary()


if __name__ == "__main__":
    main()
