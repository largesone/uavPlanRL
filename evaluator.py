# -*- coding: utf-8 -*-
# 文件名: evaluator.py
# 描述: 模型评估和推理模块，支持单模型推理和集成推理

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Union, Optional

# 本地模块导入
from entities import UAV, Target
from scenarios import get_balanced_scenario, get_small_scenario, get_complex_scenario
from environment import UAVTaskEnv, DirectedGraph
from networks import create_network
from config import Config
from evaluate import evaluate_plan
from collections import defaultdict
from matplotlib.font_manager import FontProperties, findfont
from datetime import datetime

# =============================================================================
# 从main-old.py迁移的核心处理类
# =============================================================================

def set_chinese_font():
    """查找并设置一个可用的中文字体，以解决matplotlib中文乱码问题。"""
    # 常见的Windows/Linux/MacOS中文字体列表
    font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
    
    for font_name in font_names:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            # print(f"✅ 中文字体 '{font_name}' 设置成功。")
            return
    
    print("⚠️ 警告: 未找到任何可用的中文字体 (SimHei, Microsoft YaHei等)，图表中的中文可能显示为乱码。")


class PlanVisualizer:
    """方案可视化器 - 从main-old.py迁移"""
    
    def __init__(self, config):
        self.config = config
        # 设置中文字体
        set_chinese_font()
    
    def save(self, final_plan, uavs, targets, obstacles, scenario_name, training_time, 
             plan_generation_time, evaluation_metrics=None, deadlocked_tasks=None, suffix="", inference_mode="单模型推理"):
        """保存可视化方案 - 与main-old.py格式完全一致"""
        
        # 资源消耗精确模拟
        temp_uav_resources = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
        temp_target_resources = {t.id: t.resources.copy().astype(float) for t in targets}

        # 按事件分组处理协同任务
        events = defaultdict(list)
        for uav_id, tasks in final_plan.items():
            for task in tasks:
                event_key = (task.get('arrival_time', 0), task['target_id'])
                events[event_key].append({'uav_id': uav_id, 'task_ref': task})
        
        sorted_event_keys = sorted(events.keys())

        # 协同事件日志
        collaboration_log = "\n\n协同事件日志 (揭示资源竞争):\n" + "-"*36 + "\n"

        # 按事件顺序处理协作
        for event_key in sorted_event_keys:
            arrival_time, target_id = event_key
            collaborating_steps = events[event_key]
            
            target_remaining_need_before = temp_target_resources[target_id].copy()
            collaboration_log += f" * 事件: 在 t={arrival_time:.2f}s, 无人机(UAVs) {', '.join([str(s['uav_id']) for s in collaborating_steps])} 到达 目标 {target_id}\n"
            collaboration_log += f"   - 目标初始需求: {target_remaining_need_before}\n"

            for step in collaborating_steps:
                uav_id = step['uav_id']
                task = step['task_ref']

                uav_available_resources = temp_uav_resources[uav_id]
                actual_contribution = np.minimum(target_remaining_need_before, uav_available_resources)
                
                if np.all(actual_contribution < 1e-6):
                    task['resource_cost'] = np.zeros_like(uav_available_resources)
                    collaboration_log += f"     - UAV {uav_id} 尝试贡献，但目标需求已满足。贡献: [0. 0.]\n"
                    continue

                temp_uav_resources[uav_id] -= actual_contribution
                target_remaining_need_before -= actual_contribution
                task['resource_cost'] = actual_contribution
                collaboration_log += f"     - UAV {uav_id} 贡献 {actual_contribution}, 剩余资源 {temp_uav_resources[uav_id]}\n"
                
            temp_target_resources[target_id] = target_remaining_need_before
            collaboration_log += f"   - 事件结束，目标剩余需求: {target_remaining_need_before}\n\n"

        # 创建可视化图表
        fig, ax = plt.subplots(figsize=(22, 14))
        ax.set_facecolor("#f0f0f0")
        
        # 绘制障碍物
        for obs in obstacles:
            obs.draw(ax)

        # 计算目标协作详情
        target_collaborators_details = defaultdict(list)
        for uav_id, tasks in final_plan.items():
            for task in sorted(tasks, key=lambda x: x.get('step', 0)):
                target_id = task['target_id']
                resource_cost = task.get('resource_cost', np.zeros_like(uavs[0].resources))
                target_collaborators_details[target_id].append({
                    'uav_id': uav_id, 
                    'arrival_time': task.get('arrival_time', 0), 
                    'resource_cost': resource_cost
                })

        # 计算总体资源满足情况
        summary_text = ""
        if targets:
            satisfied_targets_count = 0
            resource_types = len(targets[0].resources) if targets else 2
            total_demand_all = np.sum([t.resources for t in targets], axis=0)

            all_resource_costs = [d['resource_cost'] for details in target_collaborators_details.values() for d in details]
            if not all_resource_costs:
                total_contribution_all_for_summary = np.zeros(resource_types)
            else:
                total_contribution_all_for_summary = np.sum(all_resource_costs, axis=0)

            for t in targets:
                current_target_contribution_sum = np.sum([d['resource_cost'] for d in target_collaborators_details.get(t.id, [])], axis=0)
                if np.all(current_target_contribution_sum >= t.resources - 1e-5):
                    satisfied_targets_count += 1
            
            num_targets = len(targets)
            satisfaction_rate_percent = (satisfied_targets_count / num_targets * 100) if num_targets > 0 else 100
            total_demand_safe = total_demand_all.copy()
            total_demand_safe[total_demand_safe == 0] = 1e-6
            overall_completion_rate_percent = np.mean(np.minimum(total_contribution_all_for_summary, total_demand_all) / total_demand_safe) * 100
            
            # 计算资源富裕度
            total_supply_all = np.sum([u.initial_resources for u in uavs], axis=0)
            resource_surplus = total_supply_all - total_demand_all
            resource_abundance_rate = (resource_surplus / total_demand_safe) * 100
            
            summary_text = (f"总体资源满足情况:\n--------------------------\n"
                          f"- 总需求/总贡献: {np.array2string(total_demand_all, formatter={'float_kind':lambda x: '%.0f' % x})} / {np.array2string(total_contribution_all_for_summary, formatter={'float_kind':lambda x: '%.1f' % x})}\n"
                          f"- 总供给/资源富裕度: {np.array2string(total_supply_all, formatter={'float_kind':lambda x: '%.0f' % x})} / {np.array2string(resource_abundance_rate, formatter={'float_kind':lambda x: '%.1f%%' % x})}\n"
                          f"- 已满足目标: {satisfied_targets_count} / {num_targets} ({satisfaction_rate_percent:.1f}%)\n"
                          f"- 满足率: {overall_completion_rate_percent:.1f}% (显示用，文件名使用标准评估指标)")

        # 绘制无人机起点
        ax.scatter([u.position[0] for u in uavs], [u.position[1] for u in uavs], 
                  c='blue', marker='s', s=150, label='无人机起点', zorder=5, edgecolors='black')
        
        for u in uavs:
            ax.annotate(f"UAV{u.id}", xy=(u.position[0], u.position[1]), fontsize=12, fontweight='bold', 
                       xytext=(0, -25), textcoords='offset points', ha='center', va='top')
            ax.annotate(f"初始: {np.array2string(u.initial_resources, formatter={'float_kind': lambda x: f'{x:.0f}'})}", 
                       xy=(u.position[0], u.position[1]), fontsize=8, xytext=(15, 10), 
                       textcoords='offset points', ha='left', color='navy')

        # 绘制目标
        ax.scatter([t.position[0] for t in targets], [t.position[1] for t in targets], 
                  c='red', marker='o', s=150, label='目标', zorder=5, edgecolors='black')
        
        for t in targets:
            demand_str = np.array2string(t.resources, formatter={'float_kind': lambda x: "%.0f" % x})
            annotation_text = f"目标 {t.id}\n总需求: {demand_str}\n------------------"
            
            total_contribution = np.sum([d['resource_cost'] for d in target_collaborators_details.get(t.id, [])], axis=0)
            details_text = sorted(target_collaborators_details.get(t.id, []), key=lambda x: x['arrival_time'])
            
            if not details_text:
                annotation_text += "\n未分配无人机"
            else:
                for detail in details_text:
                    annotation_text += f"\nUAV {detail['uav_id']} (T:{detail['arrival_time']:.1f}s) 贡献:{np.array2string(detail['resource_cost'], formatter={'float_kind': lambda x: '%.1f' % x})}"
            
            if np.all(total_contribution >= t.resources - 1e-5):
                satisfaction_str, bbox_color = "[OK] 需求满足", 'lightgreen'
            else:
                satisfaction_str, bbox_color = "[NG] 资源不足", 'mistyrose'
            
            annotation_text += f"\n------------------\n状态: {satisfaction_str}"
            
            ax.annotate(f"T{t.id}", xy=(t.position[0], t.position[1]), fontsize=12, fontweight='bold', 
                       xytext=(0, 18), textcoords='offset points', ha='center', va='bottom')
            ax.annotate(annotation_text, xy=(t.position[0], t.position[1]), fontsize=7, 
                       xytext=(15, -15), textcoords='offset points', ha='left', va='top', 
                       bbox=dict(boxstyle='round,pad=0.4', fc=bbox_color, ec='black', alpha=0.9, lw=0.5), zorder=8)

        # 绘制路径 - 修复：绘制连续的任务路径
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(uavs))) if uavs else []
        uav_color_map = {u.id: colors[i] for i, u in enumerate(uavs)}
        
        for uav_id, tasks in final_plan.items():
            uav_color = uav_color_map.get(uav_id, 'gray')
            temp_resources = next(u for u in uavs if u.id == uav_id).initial_resources.copy().astype(float)
            
            # 获取无人机起始位置
            uav = next(u for u in uavs if u.id == uav_id)
            current_pos = uav.position
            
            # 按步骤顺序排序任务
            sorted_tasks = sorted(tasks, key=lambda x: x.get('step', 0))
            
            # 绘制连续路径
            for i, task in enumerate(sorted_tasks):
                # 获取目标位置
                target_id = task['target_id']
                target = next(t for t in targets if t.id == target_id)
                target_pos = target.position
                
                # 使用PH-RRT算法生成曲线路径
                try:
                    from path_planning import PHCurveRRTPlanner
                    
                    # 创建PH-RRT规划器
                    planner = PHCurveRRTPlanner(
                        start=current_pos,
                        goal=target_pos,
                        start_heading=0.0,  # 可以从UAV获取实际朝向
                        goal_heading=0.0,   # 目标朝向
                        obstacles=obstacles,
                        config=self.config
                    )
                    
                    # 执行路径规划获取PH曲线
                    result = planner.plan()
                    
                    if result is not None:
                        path_points, distance = result
                        path_points = np.array(path_points)
                    else:
                        # 规划失败时生成平滑曲线
                        path_points = self._generate_smooth_curve(current_pos, target_pos)
                        
                except Exception as e:
                    print(f"PH-RRT规划失败: {e}，使用平滑曲线")
                    path_points = self._generate_smooth_curve(current_pos, target_pos)
                
                # 绘制路径
                ax.plot(path_points[:, 0], path_points[:, 1], 
                       color=uav_color, 
                       linestyle='-' if task.get('is_sync_feasible', True) else '--', 
                       linewidth=2, alpha=0.9, zorder=3)
                
                # 添加步骤标记 - 优化：改进序列顺序的显示清晰度
                mid_pos = path_points[len(path_points) // 2]
                step_number = task.get('step', i+1)
                
                # 主步骤标记（大圆圈）
                ax.text(mid_pos[0], mid_pos[1], str(step_number), 
                       color='white', backgroundcolor=uav_color, ha='center', va='center', 
                       fontsize=11, fontweight='bold', 
                       bbox=dict(boxstyle='circle,pad=0.3', fc=uav_color, ec='white', linewidth=2), zorder=6)
                
                # 步骤箭头指示（显示方向）
                if len(path_points) > 1:
                    # 在路径的方向上添加箭头
                    arrow_start_idx = int(len(path_points) * 0.7)
                    arrow_end_idx = min(arrow_start_idx + 5, len(path_points) - 1)
                    
                    if arrow_start_idx < arrow_end_idx:
                        start_point = path_points[arrow_start_idx]
                        end_point = path_points[arrow_end_idx]
                        
                        ax.annotate('', xy=end_point, xytext=start_point,
                                   arrowprops=dict(arrowstyle='->', color=uav_color, lw=2, alpha=0.8),
                                   zorder=5)
                
                # 在路径起始点添加小型步骤标记
                if i == 0:  # 第一个任务，显示“起”
                    start_pos = path_points[0]
                    ax.text(start_pos[0], start_pos[1], '起', 
                           color=uav_color, ha='center', va='center', 
                           fontsize=8, fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=uav_color, alpha=0.9), zorder=5)
                
                # 在路径结束点添加小型步骤标记
                end_pos = path_points[-1]
                if i == len(sorted_tasks) - 1:  # 最后一个任务，显示“终”
                    ax.text(end_pos[0], end_pos[1], '终', 
                           color=uav_color, ha='center', va='center', 
                           fontsize=8, fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=uav_color, alpha=0.9), zorder=5)
                else:
                    # 中间任务，显示下一步骤的方向
                    next_step = step_number + 1
                    ax.text(end_pos[0], end_pos[1], f'→{next_step}', 
                           color=uav_color, ha='center', va='center', 
                           fontsize=7, fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.15', fc='lightyellow', ec=uav_color, alpha=0.8), zorder=5)
                
                # 添加资源信息
                resource_cost = task.get('resource_cost', np.zeros_like(temp_resources))
                temp_resources -= resource_cost
                end_pos = path_points[-1]
                remaining_res_str = f"R: {np.array2string(temp_resources.clip(0), formatter={'float_kind': lambda x: f'{x:.0f}'})}"
                ax.annotate(remaining_res_str, xy=(end_pos[0], end_pos[1]),
                           color=uav_color, ha='center', va='center', 
                           fontsize=7, fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=uav_color, alpha=0.8, lw=0.5), 
                           xytext=(10, -10), textcoords='offset points', zorder=7)
                
                # 更新当前位置为目标位置，为下一个任务做准备
                current_pos = target_pos

        # 死锁检测信息
        deadlock_summary_text = ""
        if deadlocked_tasks and any(deadlocked_tasks.values()):
            deadlock_summary_text += "!!! 死锁检测 !!!\n--------------------------\n以下无人机未能完成其任务序列，可能陷入死锁：\n"
            for uav_id, tasks in deadlocked_tasks.items():
                if tasks:
                    deadlock_summary_text += f"- UAV {uav_id}: 等待执行 -> {' -> '.join([f'T{t[0]}' for t in tasks])}\n"
            deadlock_summary_text += ("-"*30) + "\n\n"

        # 报告头部
        report_header = f"---------- {scenario_name} 执行报告 ----------\n\n" + deadlock_summary_text
        if summary_text:
            report_header += summary_text + "\n" + ("-"*30) + "\n\n"
        
        # 添加评估指标到报告中
        if evaluation_metrics:
            report_header += "评估指标:\n--------------------------\n"
            # 定义指标的中文名称映射
            metric_names = {
                'resource_utilization_rate': '资源利用率',
                'completion_rate': '资源满足率',
                'sync_feasibility_rate': '同步可行率',
                'load_balance_score': '负载均衡度'
            }
            
            # 优先显示资源利用率，去掉归一化信息
            priority_metrics = ['resource_utilization_rate']
            other_metrics = ['sync_feasibility_rate', 'load_balance_score']
            
            # 优先显示资源利用率
            for key in priority_metrics:
                if key in evaluation_metrics:
                    value = evaluation_metrics[key]
                    report_header += f"  - {metric_names.get(key, key)}: {value:.4f}\n"
            
            # 显示其他指标
            for key in other_metrics:
                if key in evaluation_metrics:
                    value = evaluation_metrics[key]
                    report_header += f"  - {metric_names.get(key, key)}: {value:.4f}\n"
            
            report_header += "-" * 20 + "\n\n"

        # 生成详细报告
        report_body_image = ""
        report_body_file = ""
        
        for uav in uavs:
            uav_header = f"* 无人机 {uav.id} (初始资源: {np.array2string(uav.initial_resources, formatter={'float_kind': lambda x: f'{x:.0f}'})})\n"
            report_body_image += uav_header
            report_body_file += uav_header
            
            details = sorted(final_plan.get(uav.id, []), key=lambda x: x.get('step', 0))
            if not details:
                no_task_str = "  - 未分配任何任务\n"
                report_body_image += no_task_str
                report_body_file += no_task_str
            else:
                temp_resources_report = uav.initial_resources.copy().astype(float)
                for detail in details:
                    resource_cost = detail.get('resource_cost', np.zeros_like(temp_resources_report))
                    temp_resources_report -= resource_cost
                    sync_status = "" if detail.get('is_sync_feasible', True) else " (警告: 无法同步)"
                    
                    common_report_part = f"  {detail.get('step', 0)}. 飞向目标 {detail['target_id']}{sync_status}:\n"
                    common_report_part += f"     - 飞行距离: {detail.get('distance', 0):.2f} m, 速度: {detail.get('speed', 15):.2f} m/s, 到达时间点: {detail.get('arrival_time', 0):.2f} s\n"
                    common_report_part += f"     - 消耗资源: {np.array2string(resource_cost, formatter={'float_kind': lambda x: '%.1f' % x})}\n"
                    common_report_part += f"     - 剩余资源: {np.array2string(temp_resources_report.clip(0), formatter={'float_kind': lambda x: f'{x:.1f}'})}\n"
                    
                    report_body_image += common_report_part
                    report_body_file += common_report_part
            
            report_body_image += "\n"
            report_body_file += "\n"

        final_report_for_image = report_header + report_body_image
        final_report_for_file = report_header + report_body_file + collaboration_log

        # 添加报告到图片
        plt.subplots_adjust(right=0.75)
        fig.text(0.77, 0.95, final_report_for_image, transform=plt.gcf().transFigure, 
                ha="left", va="top", fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', alpha=0.9))

        # 处理training_time格式化
        if isinstance(training_time, (tuple, list)):
            actual_episodes = len(training_time[0]) if training_time and len(training_time) > 0 else 0
            estimated_time = actual_episodes * 0.13
            training_time_str = f"{estimated_time:.2f}s ({actual_episodes}轮)"
        else:
            training_time_str = f"{training_time:.2f}s"

        train_mode_str = '高精度PH-RRT' if getattr(self.config, 'USE_PHRRT_DURING_TRAINING', False) else '快速近似'
        plan_mode_str = '高精度PH-RRT' if getattr(self.config, 'USE_PHRRT_DURING_PLANNING', False) else '快速近似'
        
        title_text = (
            f"多无人机任务分配与路径规划 - {scenario_name} ({inference_mode})\n"
            f"UAV: {len(uavs)}, 目标: {len(targets)}, 障碍: {len(obstacles)} | 训练: {train_mode_str} | 规划: {plan_mode_str}\n"
            f"模型训练耗时: {training_time_str} | 方案生成耗时: {plan_generation_time:.2f}s"
        )
        ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)

        ax.set_xlabel("X坐标 (m)", fontsize=14)
        ax.set_ylabel("Y坐标 (m)", fontsize=14)
        ax.legend(loc="lower left")
        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_aspect('equal', adjustable='box')

        # 保存图片 - 添加完成率信息到文件名
        output_dir = "output/images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        clean_scenario_name = scenario_name.replace(' ', '_').replace(':', '')
        
        # 获取完成率信息用于文件名 - 修复：确保使用标准评估指标中的完成率
        completion_rate_for_filename = 0.0
        if evaluation_metrics:
            completion_rate_for_filename = evaluation_metrics.get('completion_rate', 0.0)
        else:
            # 如果没有evaluation_metrics，使用PlanVisualizer计算的完成率作为备选
            completion_rate_for_filename = overall_completion_rate_percent / 100.0 if 'overall_completion_rate_percent' in locals() else 0.0
        
        # 构建包含完成率信息的文件名
        completion_str = f"comp{completion_rate_for_filename:.3f}"
        base_filename = f"{clean_scenario_name}_{timestamp}_{completion_str}{suffix}"
        img_filepath = os.path.join(output_dir, f"{base_filename}.jpg")
        
        try:
            plt.savefig(img_filepath, dpi=300, format='jpg')
            # 移除重复的输出，只在ResultSaver中输出
        except Exception as e:
            print(f"❌ 错误：无法保存结果图至 {img_filepath}")
            print(f"📄 文件写入错误详情：")
            print(f"   - 文件名: {base_filename}.jpg")
            print(f"   - 存储位置: {output_dir}")
            print(f"   - 完整路径: {img_filepath}")
            print(f"   - 图表尺寸: {fig.get_size_inches()}")
            print(f"   - 错误原因: {e}")
            print(f"   - 错误类型: {type(e).__name__}")
            
            # 尝试输出图表内容信息
            try:
                print(f"   - 图表轴数量: {len(fig.axes)}")
                print(f"   - 图表DPI: {fig.dpi}")
                print(f"   - 输出目录是否存在: {os.path.exists(output_dir)}")
                print(f"   - 输出目录权限: {os.access(output_dir, os.W_OK) if os.path.exists(output_dir) else 'N/A'}")
            except Exception as inner_e:
                print(f"   - 无法获取详细信息: {inner_e}")
        
        plt.close(fig)
        
        return final_report_for_file, img_filepath

    def _generate_smooth_curve(self, start_pos, end_pos):
        """生成平滑曲线路径作为PH-RRT的备选方案"""
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        
        distance = np.linalg.norm(end_pos - start_pos)
        num_points = max(10, int(distance / 20))  # 每20米一个点
        
        path_points = []
        for i in range(num_points + 1):
            t = i / num_points
            
            # 基础直线插值
            base_point = start_pos + t * (end_pos - start_pos)
            
            # 添加贝塞尔曲线样式的偏移
            if i > 0 and i < num_points:
                # 计算垂直于直线方向的向量
                direction = end_pos - start_pos
                perpendicular = np.array([-direction[1], direction[0]])
                if np.linalg.norm(perpendicular) > 0:
                    perpendicular = perpendicular / np.linalg.norm(perpendicular)
                
                # 使用贝塞尔曲线的控制点逻辑
                curve_factor = 4 * t * (1 - t)  # 贝塞尔曲线权重
                curve_offset = curve_factor * min(30, distance * 0.15)  # 动态偏移量
                
                base_point += perpendicular * curve_offset
            
            path_points.append(base_point)
        
        return np.array(path_points)


class ResultSaver:
    """结果保存器 - 从main-old.py迁移"""
    
    def __init__(self, config):
        self.config = config
    
    def save_plan_details(self, report_content, scenario_name, timestamp=None, evaluation_metrics=None, suffix=""):
        """保存方案详情文件 - 修改为保存到output/images/目录，并合并评估指标"""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 修改保存路径到images目录
        report_dir = "output/images"
        os.makedirs(report_dir, exist_ok=True)
        
        clean_scenario_name = scenario_name.replace(' ', '_').replace(':', '')
        
        # 获取完成率信息用于文件名
        completion_rate_for_filename = 0.0
        if evaluation_metrics:
            completion_rate_for_filename = evaluation_metrics.get('completion_rate', 0.0)
        
        # 构建包含完成率信息的文件名
        completion_str = f"comp{completion_rate_for_filename:.3f}"
        base_filename = f"{clean_scenario_name}_{timestamp}_{completion_str}{suffix}"
        report_filepath = os.path.join(report_dir, f"{base_filename}.txt")
        
        try:
            # 合并评估指标到报告内容末尾
            combined_content = report_content
            if evaluation_metrics:
                combined_content += "\n\n" + "=" * 80 + "\n"
                combined_content += "评估指标详情\n"
                combined_content += "=" * 80 + "\n"
                for key, value in evaluation_metrics.items():
                    if isinstance(value, float):
                        combined_content += f"{key}: {value:.4f}\n"
                    else:
                        combined_content += f"{key}: {value}\n"
                combined_content += "=" * 80 + "\n"
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(combined_content)
            print(f"详细方案报告已保存至: {report_filepath}")
            return report_filepath
        except Exception as e:
            print(f"❌ 错误：无法保存任务报告至 {report_filepath}")
            print(f"📄 文件写入错误详情：")
            print(f"   - 文件名: {base_filename}.txt")
            print(f"   - 存储位置: {report_dir}")
            print(f"   - 完整路径: {report_filepath}")
            print(f"   - 内容长度: {len(combined_content)} 字符")
            print(f"   - 内容前100字符: {combined_content[:100]}...")
            print(f"   - 错误原因: {e}")
            print(f"   - 错误类型: {type(e).__name__}")
            
            # 尝试输出目录和权限信息
            try:
                print(f"   - 目录是否存在: {os.path.exists(report_dir)}")
                print(f"   - 目录权限: {os.access(report_dir, os.W_OK) if os.path.exists(report_dir) else 'N/A'}")
                if hasattr(os, 'statvfs'):
                    stat = os.statvfs(report_dir)
                    free_space = stat.f_bavail * stat.f_frsize / (1024**3)
                    print(f"   - 磁盘空间: {free_space:.2f} GB")
                else:
                    print(f"   - 磁盘空间: 无法检测")
            except Exception as inner_e:
                print(f"   - 无法获取详细信息: {inner_e}")
            
            return None

class ModelEvaluator:
    """模型评估器 - 支持单模型和集成推理"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"评估设备: {self.device}")
        
        # 评估统计
        self.evaluation_stats = {
            'scenario_results': [],
            'average_completion_rate': 0.0,
            'average_efficiency': 0.0,
            'evaluation_time': 0.0
        }

    def _generate_complete_visualization(self, scenario_name: str, inference_mode: str, 
                                    uavs, targets, obstacles, results, suffix: str = ""):
        """生成完整的可视化结果 - 集成PlanVisualizer和ResultSaver"""
        try:
            # 构建final_plan格式（从推理结果转换）
            final_plan = self._convert_results_to_plan(results, uavs, targets)
            
            # 创建可视化器和保存器
            visualizer = PlanVisualizer(self.config)
            saver = ResultSaver(self.config)
            
            # 计算评估指标
            final_uav_states = results.get('final_uav_states', None)
            evaluation_metrics = evaluate_plan(final_plan, uavs, targets, final_uav_states=final_uav_states)
            
            # 【重要修改】以推理结果为准，推理结果就是最终的分配方案
            if results and 'completion_rate' in results:
                # 使用推理结果中的完成率作为最终结果
                evaluation_metrics['completion_rate'] = results['completion_rate']
                print(f"[DEBUG] 使用推理结果中的完成率: {results['completion_rate']:.4f}")
                
                # 如果有推理任务分配方案，使用推理结果覆盖evaluate_plan的结果
                if 'inference_task_assignments' in results and 'inference_target_status' in results:
                    print(f"[DEBUG] 使用推理任务分配方案作为最终结果")
                    # 可以在这里添加逻辑来使用推理结果覆盖evaluate_plan的某些指标
            else:
                print(f"[DEBUG] 使用evaluate_plan计算的完成率: {evaluation_metrics.get('completion_rate', 0):.4f}")
            
            # 生成可视化和报告
            training_time = 0.0  # 推理阶段无训练时间
            plan_generation_time = self.evaluation_stats['evaluation_time']
            
            # 传递suffix和推理方式参数
            report_content, img_filepath = visualizer.save(
                final_plan, uavs, targets, obstacles, scenario_name,
                training_time, plan_generation_time, evaluation_metrics, None, suffix, inference_mode
            )
            
            # 保存详细报告，包含评估指标
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            report_filepath = saver.save_plan_details(report_content, scenario_name, timestamp, evaluation_metrics, suffix)
            
            print(f"完整可视化结果已生成: {scenario_name}")
            
        except Exception as e:
            print(f"生成完整可视化时出错: {e}")

    def _build_plan_from_inference_results(self, results, uavs, targets):
        """
        从推理结果构建执行计划
        
        Args:
            results: 推理结果字典
            uavs: UAV列表
            targets: 目标列表
            
        Returns:
            dict: 执行计划
        """
        final_plan = {uav.id: [] for uav in uavs}
        
        inference_assignments = results.get('inference_task_assignments', {})
        inference_targets = results.get('inference_target_status', {})
        
        print(f"[DEBUG] 从推理结果构建执行计划:")
        print(f"  - UAV分配方案数量: {len(inference_assignments)}")
        print(f"  - 目标状态数量: {len(inference_targets)}")
        
        # 为每个UAV构建任务序列
        for uav_id, uav_info in inference_assignments.items():
            consumed_resources = uav_info['consumed_resources']
            initial_resources = uav_info['initial_resources']
            
            # 找到该UAV贡献资源的目标
            uav_tasks = []
            for target_id, target_info in inference_targets.items():
                contributed_resources = target_info['contributed_resources']
                
                # 检查该UAV是否向此目标贡献了资源
                if np.any(contributed_resources > 0):
                    # 计算该UAV对此目标的贡献比例
                    # 这里简化处理，假设UAV按比例贡献资源
                    contribution_ratio = np.mean(contributed_resources / (initial_resources + 1e-6))
                    
                    if contribution_ratio > 0.1:  # 如果贡献比例超过10%，认为该UAV参与了此目标
                        # 找到对应的目标对象
                        target_obj = next((t for t in targets if t.id == target_id), None)
                        uav_obj = next((u for u in uavs if u.id == uav_id), None)
                        
                        if target_obj and uav_obj:
                            distance = np.linalg.norm(uav_obj.position - target_obj.position)
                            speed = 15.0  # 默认速度
                            
                            task = {
                                'target_id': target_id,
                                'step': len(uav_tasks) + 1,
                                'distance': distance,
                                'speed': speed,
                                'arrival_time': distance / speed,
                                'resource_cost': contributed_resources,
                                'is_sync_feasible': True  # 推理结果认为可行
                            }
                            uav_tasks.append(task)
            
            final_plan[uav_id] = uav_tasks
            print(f"  - UAV {uav_id}: 分配了 {len(uav_tasks)} 个任务")
        
        return final_plan

    def _convert_results_to_plan(self, results, uavs, targets):
        """将推理结果转换为final_plan格式"""
        final_plan = {uav.id: [] for uav in uavs}        
        
        # 优先使用推理结束时直接记录的、最准确的任务分配方案
        if 'inference_task_assignments' in results and 'inference_target_status' in results:
            print(f"[DEBUG] 使用推理结束时记录的最终分配方案构建执行计划")
            return self._build_plan_from_inference_results(results, uavs, targets)
        
        # 后备方案：通过模拟action_sequence来重建任务分配
        final_plan = {uav.id: [] for uav in uavs}
        action_sequence = results.get('action_sequence', [])
        print(f"[DEBUG] 通过模拟action_sequence重建执行计划，动作数量: {len(action_sequence)}")

        # --- 创建环境状态的深拷贝用于模拟 ---
        import copy
        temp_uavs = {u.id: copy.deepcopy(u) for u in uavs}
        temp_targets = {t.id: copy.deepcopy(t) for t in targets}
        
        step_counter = 0
        for action_idx in action_sequence:
            try:
                # 解码动作
                n_uavs = len(uavs)
                n_targets = len(targets)
                n_phi = getattr(self.config, 'GRAPH_N_PHI', 1)
                
                if n_uavs == 0 or n_targets == 0 or n_phi == 0: continue
                
                target_idx = action_idx // (n_uavs * n_phi)
                uav_idx = (action_idx % (n_uavs * n_phi)) // n_phi
                
                if not (target_idx < n_targets and uav_idx < n_uavs): continue

                target = targets[target_idx]
                uav = uavs[uav_idx]

                # 获取模拟中的当前状态
                sim_uav = temp_uavs[uav.id]
                sim_target = temp_targets[target.id]

                # --- [新增] 检查动作在当前模拟状态下是否有效 ---
                actual_contribution = np.minimum(sim_uav.resources, sim_target.remaining_resources)
                
                if np.sum(actual_contribution) > 1e-6:
                    # 只有当动作能产生实际贡献时，才记录到最终方案中
                    step_counter += 1
                    distance = np.linalg.norm(sim_uav.current_position - sim_target.position)
                    
                    task = {
                        'target_id': target.id,
                        'step': step_counter,
                        'distance': distance,
                        'speed': 15.0, # 默认速度
                        'arrival_time': step_counter * (distance / 15.0), # 简化到达时间
                        'is_sync_feasible': True,
                        'resource_cost': actual_contribution,
                    }
                    final_plan[uav.id].append(task)
                    
                    # --- [新增] 更新模拟状态 ---
                    sim_uav.resources -= actual_contribution
                    sim_target.remaining_resources -= actual_contribution
                    sim_uav.current_position = sim_target.position # 更新UAV位置

            except Exception as e:
                print(f"在模拟动作序列重建方案时出错: {e}")
                continue
        
        return final_plan

    def start_evaluation(self, model_paths: Union[str, List[str]], scenario_name: str = "small"):
        """
        启动评估过程
        
        Args:
            model_paths (Union[str, List[str]]): 模型路径，单个路径或路径列表
            scenario_name (str): 场景名称
        """
        print("=" * 60)
        print("开始模型评估")
        print("=" * 60)
        
        # 处理模型路径并定义文件名后缀
        if isinstance(model_paths, str):
            model_paths = [model_paths]
            inference_mode = "单模型推理"
            suffix = '_single_inference'  # 新增：单模型推理后缀
        elif len(model_paths) == 1:
            # 只有一个模型的情况，也认为是单模型推理
            inference_mode = "单模型推理"
            suffix = '_single_inference'  # 单模型推理后缀
        else:
            inference_mode = "集成推理"
            suffix = '_ensemble_inference'  # 新增：集成推理后缀
        
        print(f"推理模式: {inference_mode}")
        print(f"模型数量: {len(model_paths)}")
        print(f"评估场景: {scenario_name}")
        
        # 显示路径规划算法信息
        train_algo = "高精度PH-RRT" if getattr(self.config, 'USE_PHRRT_DURING_TRAINING', False) else "快速近似"
        plan_algo = "高精度PH-RRT" if getattr(self.config, 'USE_PHRRT_DURING_PLANNING', False) else "快速近似"
        print(f"训练算法: {train_algo}")
        print(f"规划算法: {plan_algo}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 加载场景
        uavs, targets, obstacles = self._load_scenario(scenario_name)
        
        if len(model_paths) == 1:
            results = self._single_model_inference(model_paths[0], uavs, targets, obstacles, scenario_name)
        else:
            results = self._ensemble_inference(model_paths, uavs, targets, obstacles, scenario_name)
        
        end_time = time.time()
        self.evaluation_stats['evaluation_time'] = end_time - start_time
        
        # --- 单一数据源处理流程 ---
        if results:
            # 1. 重建带有正确资源消耗的规划方案
            final_plan = self._convert_results_to_plan(results, uavs, targets)
            
            # 2. 调用权威评估函数，生成唯一的评估指标
            evaluation_metrics = evaluate_plan(
                final_plan, uavs, targets, final_uav_states=results.get('final_uav_states')
            )
            
            # 3. 将权威评估结果合并到results中，作为唯一数据源
            results.update(evaluation_metrics)

            # 4. 处理评估结果（用于控制台输出）
            self._process_evaluation_results(results)

            # 5. 生成完整的可视化结果（用于报告和图片）
            self._generate_complete_visualization(scenario_name, inference_mode, uavs, targets, obstacles, results, suffix)

        print(f"\n评估完成! 总耗时: {self.evaluation_stats['evaluation_time']:.2f}秒")
    
    def _load_scenario(self, scenario_name: str):
        """
        加载指定场景 - 支持静态和动态场景
        
        Args:
            scenario_name (str): 场景名称
            
        Returns:
            tuple: (uavs, targets, obstacles)
        """
        obstacle_tolerance = getattr(self.config, 'OBSTACLE_TOLERANCE', 50.0)
        
        # 静态预定义场景
        if scenario_name == "balanced":
            return get_balanced_scenario(obstacle_tolerance)
        elif scenario_name == "small":
            print(f"使用静态small场景进行推理")
            return get_small_scenario(obstacle_tolerance)
        elif scenario_name == "complex":
            return get_complex_scenario(obstacle_tolerance)
        # 动态场景
        elif scenario_name in ["easy", "medium", "hard"]:
            print(f"使用动态{scenario_name}场景进行推理")
            # 对于动态场景，返回空列表。
            # 真正的场景将在 env.reset() 调用中生成。
            return [], [], []
            # # 创建临时环境来生成动态场景
            # from environment import UAVTaskEnv
            # # 直接创建临时环境，让环境内部处理图创建
            # temp_env = UAVTaskEnv([], [], None, [], self.config, obs_mode="graph")
            # temp_env._initialize_entities(scenario_name)
            # return temp_env.uavs, temp_env.targets, temp_env.obstacles
        else:
            print(f"未知场景名称: {scenario_name}，使用默认small场景")
            return get_small_scenario(obstacle_tolerance)
    
    def _single_model_inference(self, model_path: str, uavs, targets, obstacles, scenario_name='easy'):
        """
        单模型推理（使用Softmax采样）
        
        Args:
            model_path (str): 模型路径
            uavs: UAV列表
            targets: 目标列表
            obstacles: 障碍物列表
            scenario_name (str): 场景名称，用于环境重置
            
        Returns:
            dict: 推理结果
        """
        print(f"执行单模型推理: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在 - {model_path}")
            return None
        
        # 创建图和环境
        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
        
        # 计算输入输出维度
        if self.config.NETWORK_TYPE == "ZeroShotGNN":
            obs_mode = "graph"
            i_dim = 64  # 占位值
            o_dim = len(targets) * len(uavs) * graph.n_phi

            if o_dim <= 0 and scenario_name in ["easy", "medium", "hard"]:
                # 如果是动态场景且初始为空，则使用配置中的最大实体数来创建网络
                max_o_dim = self.config.MAX_TARGETS * self.config.MAX_UAVS * self.config.GRAPH_N_PHI
                o_dim = max_o_dim
                # 为了区分，可以在日志中添加不同的提示
                if "_ensemble" in self.start_evaluation.__name__: # 这是一个简化的判断，实际应看调用栈
                     print(f"动态场景集成推理：使用最大动作空间占位符 o_dim={o_dim}")
                else:
                     print(f"动态场景单模型推理：使用最大动作空间占位符 o_dim={o_dim}")

        else:
            obs_mode = "flat"
            target_dim = 7 * len(targets)
            uav_dim = 8 * len(uavs)
            collaboration_dim = len(targets) * len(uavs)
            global_dim = 10
            i_dim = target_dim + uav_dim + collaboration_dim + global_dim
            o_dim = len(targets) * len(uavs) * graph.n_phi
        
        # 创建网络并加载模型
        network = create_network(
            self.config.NETWORK_TYPE, 
            i_dim, 
            self.config.hyperparameters.hidden_dim, 
            o_dim, 
            self.config
        ).to(self.device)
        
        try:
            # 加载模型文件
            try:
                # 首先尝试使用weights_only=False加载
                try:
                    model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                except TypeError as type_error:
                    # 处理PyTorch 2.6版本中weights_only参数的变化
                    if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                        print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                        model_data = torch.load(model_path, map_location=self.device)
                    else:
                        raise
                except Exception as check_error:
                    print(f"格式检查时使用weights_only=False加载失败，尝试使用weights_only=True: {check_error}")
                    try:
                        model_data = torch.load(model_path, map_location=self.device, weights_only=True)
                    except TypeError as type_error:
                        # 处理PyTorch 2.6版本中weights_only参数的变化
                        if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                            print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                            model_data = torch.load(model_path, map_location=self.device)
                        else:
                            raise
            except Exception as load_error:
                # 如果失败，尝试使用weights_only=True加载
                print(f"使用weights_only=False加载失败，尝试使用weights_only=True: {load_error}")
                try:
                    model_data = torch.load(model_path, map_location=self.device, weights_only=True)
                except TypeError as type_error:
                    # 处理PyTorch 2.6版本中weights_only参数的变化
                    if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                        print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                        model_data = torch.load(model_path, map_location=self.device)
                    else:
                        raise
            
            # 检查模型数据格式
            if isinstance(model_data, dict) and 'model_state_dict' in model_data:
                # 新格式：包含模型状态和额外信息
                network.load_state_dict(model_data['model_state_dict'])
                print(f"模型加载成功(新格式): {os.path.basename(model_path)}")
            else:
                # 旧格式：直接是状态字典
                network.load_state_dict(model_data)
                print(f"模型加载成功(旧格式): {os.path.basename(model_path)}")
            
            network.eval()
        except Exception as e:
            print(f"模型加载失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            print(f"尝试检查模型格式...")
            try:
                model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                print(f"模型数据类型: {type(model_data)}")
                if isinstance(model_data, dict):
                    print(f"模型数据键: {list(model_data.keys())}")
            except Exception as inner_e:
                print(f"模型格式检查失败: {inner_e}")
            return None
        
        # 创建环境
        env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode=obs_mode)
        
        # 执行推理，传递scenario_name参数
        results = self._run_inference(network, env, use_softmax_sampling=True, scenario_name=scenario_name)
        
        return results
    
    def _ensemble_inference(self, model_paths: List[str], uavs, targets, obstacles, scenario_name='easy'):
        """
        集成推理（集成Softmax）
        
        Args:
            model_paths (List[str]): 模型路径列表
            uavs: UAV列表
            targets: 目标列表
            obstacles: 障碍物列表
            scenario_name (str): 场景名称，用于环境重置
            
        Returns:
            dict: 推理结果
        """
        print(f"执行集成推理，模型数量: {len(model_paths)}")
        
        # 检查所有模型文件
        valid_models = []
        for path in model_paths:
            if os.path.exists(path):
                valid_models.append(path)
            else:
                print(f"警告: 模型文件不存在 - {path}")
        
        if not valid_models:
            print("错误: 没有有效的模型文件")
            return None
        
        print(f"有效模型数量: {len(valid_models)}")
        
        # 创建图和环境
        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
        
        # 计算输入输出维度
        if self.config.NETWORK_TYPE == "ZeroShotGNN":
            obs_mode = "graph"
            i_dim = 64
            o_dim = len(targets) * len(uavs) * graph.n_phi
            if o_dim <= 0 and scenario_name in ["easy", "medium", "hard"]:
                # 如果是动态场景且初始为空，则使用配置中的最大实体数来创建网络
                max_o_dim = self.config.MAX_TARGETS * self.config.MAX_UAVS * self.config.GRAPH_N_PHI
                o_dim = max_o_dim
                # 为了区分，可以在日志中添加不同的提示
                if "_ensemble" in self.start_evaluation.__name__: # 这是一个简化的判断，实际应看调用栈
                     print(f"动态场景集成推理：使用最大动作空间占位符 o_dim={o_dim}")
                else:
                     print(f"动态场景单模型推理：使用最大动作空间占位符 o_dim={o_dim}")

        else:
            obs_mode = "flat"
            target_dim = 7 * len(targets)
            uav_dim = 8 * len(uavs)
            collaboration_dim = len(targets) * len(uavs)
            global_dim = 10
            i_dim = target_dim + uav_dim + collaboration_dim + global_dim
            o_dim = len(targets) * len(uavs) * graph.n_phi
        
        # 加载所有模型
        networks = []
        for model_path in valid_models:
            network = create_network(
                self.config.NETWORK_TYPE,
                i_dim,
                self.config.hyperparameters.hidden_dim,
                o_dim,
                self.config
            ).to(self.device)
            
            try:
                # 加载模型文件
                try:
                    # 首先尝试使用weights_only=False加载
                    model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                except TypeError as type_error:
                    # 处理PyTorch 2.6版本中weights_only参数的变化
                    if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                        print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                        model_data = torch.load(model_path, map_location=self.device)
                    else:
                        raise
                except Exception as load_error:
                    # 如果失败，尝试使用weights_only=True加载
                    print(f"使用weights_only=False加载失败，尝试使用weights_only=True: {load_error}")
                    try:
                        model_data = torch.load(model_path, map_location=self.device, weights_only=True)
                    except TypeError as type_error:
                        # 处理PyTorch 2.6版本中weights_only参数的变化
                        if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                            print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                            model_data = torch.load(model_path, map_location=self.device)
                        else:
                            raise
                
                # 检查模型数据格式
                if isinstance(model_data, dict) and 'model_state_dict' in model_data:
                    # 新格式：包含模型状态和额外信息
                    network.load_state_dict(model_data['model_state_dict'])
                    print(f"模型加载成功(新格式): {os.path.basename(model_path)}")
                else:
                    # 旧格式：直接是状态字典
                    network.load_state_dict(model_data)
                    print(f"模型加载成功(旧格式): {os.path.basename(model_path)}")
                
                network.eval()
                networks.append(network)
            except Exception as e:
                print(f"模型加载失败 {model_path}: {e}")
                print(f"错误类型: {type(e).__name__}")
                print(f"尝试检查模型格式...")
                try:
                    model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                    print(f"模型数据类型: {type(model_data)}")
                    if isinstance(model_data, dict):
                        print(f"模型数据键: {list(model_data.keys())}")
                except Exception as inner_e:
                    print(f"模型格式检查失败: {inner_e}")
        
        if not networks:
            print("错误: 没有成功加载的模型")
            return None
        
        # 创建环境
        env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode=obs_mode)
        
        # 执行集成推理，传递scenario_name参数
        results = self._run_ensemble_inference(networks, env, scenario_name)
        
        return results
    
    def _run_inference(self, network, env, use_softmax_sampling=True, scenario_name='easy'):
        """
        运行单模型推理
        
        Args:
            network: 神经网络模型
            env: 环境
            use_softmax_sampling (bool): 是否使用Softmax采样
            scenario_name (str): 场景名称，用于环境重置
            
        Returns:
            dict: 推理结果
        """
        # 【修复】在环境重置时传递正确的scenario_name参数，推理时静默重置
        reset_options = {'scenario_name': scenario_name, 'silent_reset': True}
        reset_result = env.reset(options=reset_options)
        # 处理reset返回的tuple格式 (state, info)
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        total_reward = 0.0
        step_count = 0
        max_steps = 100
        action_sequence = []
        
        with torch.no_grad():
            while step_count < max_steps:
                # 准备状态张量
                if env.obs_mode == "flat":
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:  # graph mode
                    state_tensor = {}
                    for key, value in state.items():
                        if key == "masks":
                            mask_tensor = {}
                            for mask_key, mask_value in value.items():
                                mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0).to(self.device)
                            state_tensor[key] = mask_tensor
                        else:
                            state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                
                # 获取Q值
                q_values = network(state_tensor)
                
                # 获取动作掩码
                action_mask = env.get_action_mask()
                valid_actions = np.where(action_mask)[0]
                
                if len(valid_actions) == 0:
                    break
                
                # 添加调试信息：显示动作候选列表
                if self.config.ENABLE_DEBUG:
                    print(f"\n[DEBUG] 步骤 {step_count + 1} 动作候选列表: "
                          f"  有效动作数量: {len(valid_actions)}"
                          f"  有效动作索引: {valid_actions.tolist()}")
                    
                    # 显示所有动作的详细信息
                    for i, action_idx in enumerate(valid_actions):
                        try:
                            target_idx, uav_idx, phi_idx = env._action_to_assignment(action_idx)
                            target = env.targets[target_idx]
                            uav = env.uavs[uav_idx]
                            q_value = q_values[0][action_idx].item()
                            
                            # 计算距离和资源信息
                            uav_pos = uav.position
                            target_pos = target.position
                            distance = np.linalg.norm(uav_pos - target_pos)
                            
                            # 获取UAV当前资源和目标需求
                            uav_resources = uav.resources
                            target_needs = target.resources
                            
                            # 计算可能的资源贡献
                            possible_contribution = np.minimum(uav_resources, target_needs)
                            total_possible = np.sum(possible_contribution)
                        

                            print(f"    动作{i+1} (索引{action_idx}): UAV{uav.id}->Target{target.id}"
                                  f"      - Q值: {q_value:.3f}"
                                  f"      - 距离: {distance:.2f}m"
                                  f"      - UAV资源: {uav_resources}"
                                  f"      - 目标需求: {target_needs}"
                                  f"      - 可能贡献: {possible_contribution} (总计: {total_possible:.1f})")

                        except Exception as e:
                            print(f"    动作{i+1} (索引{action_idx}): 解析失败 - {e}"
                                  f"      - Q值: {q_values[0][action_idx].item():.3f}")
                
                # 选择动作
                if use_softmax_sampling:
                    # Softmax采样
                    valid_q_values = q_values[0][valid_actions]
                    probs = torch.softmax(valid_q_values / 0.1, dim=0)  # 温度参数0.1
                    action_idx = torch.multinomial(probs, 1).item()
                    action = valid_actions[action_idx]
                    
                    # 添加调试信息：显示选择过程
                    if self.config.ENABLE_DEBUG:
                        selected_prob = probs[action_idx].item()
                        print(f"  [DEBUG] Softmax采样选择过程:" 
                            f"    - 有效Q值: {valid_q_values.tolist()}"
                            f"    - 选择概率: {probs.tolist()}"
                            f"    - 选择索引: {action_idx} \n"
                            f"    - 最终动作: {action}, 概率: {selected_prob:.4f}")
                else:
                    # 贪婪选择
                    masked_q_values = q_values[0].clone()
                    masked_q_values[~torch.tensor(action_mask, dtype=torch.bool)] = float('-inf')
                    action = masked_q_values.argmax().item()
                    
                    # 添加调试信息：显示贪婪选择
                    if self.config.ENABLE_DEBUG:
                        max_q_value = masked_q_values[action].item()
                        print(f"  [DEBUG] 贪婪选择过程:")
                        print(f"    - 最大Q值: {max_q_value:.3f}")
                        print(f"    - 选择动作: {action}")
                
                # 执行动作前检查是否有实际贡献（双重验证）
                target_idx, uav_idx, phi_idx = env._action_to_assignment(action)
                target = env.targets[target_idx]
                uav = env.uavs[uav_idx]
                
                # 双重检查：确保动作有效且有实际贡献
                if not env._is_valid_action(target, uav, phi_idx) or not env._has_actual_contribution(target, uav):
                    # 跳过无效或无贡献的动作
                    print(f"⚠️ 跳过无效动作: UAV{uav.id} -> Target{target.id} (无资源贡献)")
                    continue
                
                # 执行动作
                next_state, reward, done, truncated, info = env.step(action)
                # 从info中提取reward_breakdown
                reward_breakdown = info.get('reward_breakdown', {})

                # 添加调试信息
                if self.config.ENABLE_DEBUG:
                    print(f"[DEBUG] 步骤 {step_count + 1}: UAV{uav.id} -> Target{target.id}, 动作={action}, 奖励={reward:.2f}")

                # 如果奖励异常（小于-100），打印详细的奖励分解
                if reward < -100:
                    print(f"\n[DEBUG] 异常奖励检测! 奖励值: {reward:.2f}")
                    print("详细奖励分解:")
                    if reward_breakdown:
                        # 打印第一层奖励分解
                        if 'layer1_breakdown' in reward_breakdown:
                            print("  第一层奖励 (资源匹配):")
                            for key, value in reward_breakdown['layer1_breakdown'].items():
                                print(f"    {key}: {value:.4f}")
                            print(f"  第一层总计: {reward_breakdown.get('layer1_total', 0):.4f}")
                        
                        # 打印第二层奖励分解
                        if 'layer2_breakdown' in reward_breakdown:
                            print("  第二层奖励 (优化选择):")
                            for key, value in reward_breakdown['layer2_breakdown'].items():
                                print(f"    {key}: {value:.4f}")
                            print(f"  第二层总计: {reward_breakdown.get('layer2_total', 0):.4f}")
                        
                        # 打印其他奖励信息
                        other_keys = ['base_reward', 'shaping_reward', 'final_total_reward', 'was_clipped']
                        for key in other_keys:
                            if key in reward_breakdown:
                                print(f"  {key}: {reward_breakdown[key]}")
                    else:
                        print("  未找到详细奖励分解信息")
                    print()

                total_reward += reward
                action_sequence.append(action)
                state = next_state
                step_count += 1
                
                if done or truncated:
                    break
        
        # 【重要修复】以推理结果为准，记录推理结束时的任务分配方案
        # 推理过程已经完成了任务分配决策，实际执行只是进行路径规划等后续工作
        
        # 计算推理结束时的完成率（基于推理结果）
        total_demand = np.sum([t.resources for t in env.targets], axis=0)
        total_contribution = np.zeros_like(total_demand, dtype=np.float64)
        for target in env.targets:
            target_contribution = target.resources - target.remaining_resources
            total_contribution += target_contribution.astype(np.float64)
        
        total_demand_sum = np.sum(total_demand)
        total_contribution_sum = np.sum(total_contribution)
        completion_rate = total_contribution_sum / total_demand_sum if total_demand_sum > 0 else 1.0
        
        # 【调试】显示推理结束时的任务分配结果
        print(f"[DEBUG] 推理任务分配结果:")
        print(f"  - 总需求: {total_demand} (总和: {total_demand_sum})")
        print(f"  - 推理分配总贡献: {total_contribution} (总和: {total_contribution_sum})")
        print(f"  - 推理完成率: {completion_rate:.4f}")
        
        # 计算目标完成率（完全满足的目标数量比例）
        satisfied_targets = sum(1 for t in env.targets if np.all(t.remaining_resources <= 1e-6))
        total_targets = len(env.targets)
        target_completion_rate = satisfied_targets / total_targets if total_targets > 0 else 1.0
        print(f"  - 推理时完全满足目标数: {satisfied_targets}/{total_targets}")
        print(f"  - 推理时目标完成率: {target_completion_rate:.4f}")
        
        # 显示每个目标的详细状态
        for i, target in enumerate(env.targets):
            remaining = target.remaining_resources
            is_satisfied = np.all(remaining <= 1e-6)
            print(f"  - 目标{i+1}: 剩余需求{remaining}, 完全满足: {is_satisfied}")
        
        # 记录推理结束时的任务分配方案
        inference_task_assignments = {}
        for uav in env.uavs:
            inference_task_assignments[uav.id] = {
                'initial_resources': uav.initial_resources.copy(),
                'final_resources': uav.resources.copy(),
                'consumed_resources': uav.initial_resources - uav.resources
            }
        
        inference_target_status = {}
        for target in env.targets:
            inference_target_status[target.id] = {
                'required_resources': target.resources.copy(),
                'remaining_resources': target.remaining_resources.copy(),
                'contributed_resources': target.resources - target.remaining_resources
            }
        
        print(f"[DEBUG] 推理任务分配方案已记录，包含{len(inference_task_assignments)}个UAV和{len(inference_target_status)}个目标")
    
        # 添加调试信息：显示动作序列
        if self.config.ENABLE_DEBUG:
            print(f"\n[DEBUG] 推理完成，动作序列: {action_sequence}")
            print(f"[DEBUG] 总步数: {step_count}, 总奖励: {total_reward:.2f}, 完成率: {completion_rate:.4f}")
        
        # 保存推理完成后的UAV状态，用于资源利用率计算
        final_uav_states = []
        for uav in env.uavs:
            final_uav_states.append({
                'id': uav.id,
                'initial_resources': uav.initial_resources.copy(),
                'final_resources': uav.resources.copy()
            })
        
        return {
            'total_reward': total_reward,
            'completion_rate': completion_rate,  # 基于推理结果计算的完成率
            'step_count': step_count,
            'action_sequence': action_sequence,
            'final_state': state,
            'final_uav_states': final_uav_states,
            'inference_task_assignments': inference_task_assignments,  # 推理任务分配方案
            'inference_target_status': inference_target_status  # 推理目标状态
        }
    
    def _run_ensemble_inference(self, networks, env, scenario_name='easy'):
        """
        运行集成推理
        
        Args:
            networks: 网络模型列表
            env: 环境
            scenario_name (str): 场景名称，用于环境重置
            
        Returns:
            dict: 推理结果
        """
        # 【修复】在环境重置时传递正确的scenario_name参数，推理时静默重置
        reset_options = {'scenario_name': scenario_name, 'silent_reset': True}
        reset_result = env.reset(options=reset_options)
        # 处理reset返回的tuple格式 (state, info)
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        total_reward = 0.0
        step_count = 0
        max_steps = 100
        action_sequence = []
        
        with torch.no_grad():
            while step_count < max_steps:
                # 准备状态张量
                if env.obs_mode == "flat":
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:  # graph mode
                    state_tensor = {}
                    for key, value in state.items():
                        if key == "masks":
                            mask_tensor = {}
                            for mask_key, mask_value in value.items():
                                mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0).to(self.device)
                            state_tensor[key] = mask_tensor
                        else:
                            state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                
                # 获取所有模型的Q值并平均
                ensemble_q_values = None
                for network in networks:
                    q_values = network(state_tensor)
                    if ensemble_q_values is None:
                        ensemble_q_values = q_values
                    else:
                        ensemble_q_values += q_values
                
                ensemble_q_values /= len(networks)
                
                # 获取动作掩码
                action_mask = env.get_action_mask()
                valid_actions = np.where(action_mask)[0]
                
                if len(valid_actions) == 0:
                    break
                
                # 添加调试信息：显示动作候选列表
                if self.config.ENABLE_DEBUG:
                    print(f"\n[DEBUG] 集成推理步骤 {step_count + 1} 动作候选列表:")
                    print(f"  有效动作数量: {len(valid_actions)}")
                    print(f"  有效动作索引: {valid_actions.tolist()}")
                    
                    # 显示所有动作的详细信息
                    for i, action_idx in enumerate(valid_actions):
                        try:
                            target_idx, uav_idx, phi_idx = env._action_to_assignment(action_idx)
                            target = env.targets[target_idx]
                            uav = env.uavs[uav_idx]
                            q_value = ensemble_q_values[0][action_idx].item()
                            
                            # 计算距离和资源信息
                            uav_pos = uav.position
                            target_pos = target.position
                            distance = np.linalg.norm(uav_pos - target_pos)
                            
                            # 获取UAV当前资源和目标需求
                            uav_resources = uav.resources
                            target_needs = target.resources
                            
                            # 计算可能的资源贡献
                            possible_contribution = np.minimum(uav_resources, target_needs)
                            total_possible = np.sum(possible_contribution)
                            
                            print(f"    动作{i+1} (索引{action_idx}): UAV{uav.id}->Target{target.id}")
                            print(f"      - Q值: {q_value:.3f}")
                            print(f"      - 距离: {distance:.2f}m")
                            print(f"      - UAV资源: {uav_resources}")
                            print(f"      - 目标需求: {target_needs}")
                            print(f"      - 可能贡献: {possible_contribution} (总计: {total_possible:.1f})")
                        except Exception as e:
                            print(f"    动作{i+1} (索引{action_idx}): 解析失败 - {e}")
                            print(f"      - Q值: {ensemble_q_values[0][action_idx].item():.3f}")
                
                # 集成Softmax采样
                valid_q_values = ensemble_q_values[0][valid_actions]
                probs = torch.softmax(valid_q_values / 0.1, dim=0)
                action_idx = torch.multinomial(probs, 1).item()
                action = valid_actions[action_idx]
                
                # 添加调试信息：显示选择过程
                if self.config.ENABLE_DEBUG:
                    selected_prob = probs[action_idx].item()
                    print(f"  [DEBUG] 集成Softmax采样选择过程:")
                    print(f"    - 有效Q值: {valid_q_values.tolist()}")
                    print(f"    - 选择概率: {probs.tolist()}")
                    print(f"    - 选择索引: {action_idx}")
                    print(f"    - 最终动作: {action}, 概率: {selected_prob:.4f}")
                
                # 执行动作前检查是否有实际贡献（双重验证）
                target_idx, uav_idx, phi_idx = env._action_to_assignment(action)
                target = env.targets[target_idx]
                uav = env.uavs[uav_idx]
                
                # 双重检查：确保动作有效且有实际贡献
                if not env._is_valid_action(target, uav, phi_idx) or not env._has_actual_contribution(target, uav):
                    # 跳过无效或无贡献的动作
                    print(f"⚠️ 跳过无效动作: UAV{uav.id} -> Target{target.id} (无资源贡献)")
                    continue
                
                # 执行动作
                next_state, reward, done, truncated, info = env.step(action)
                
                # 添加调试信息
                if self.config.ENABLE_DEBUG:
                    print(f"[DEBUG] 步骤 {step_count + 1}: UAV{uav.id} -> Target{target.id}, 动作={action}, 奖励={reward:.2f}")
                
                total_reward += reward
                action_sequence.append(action)
                state = next_state
                step_count += 1
                
                if done or truncated:
                    break
        
        # 计算完成率 - 修复：使用标准评估指标的计算方式
        # 基于资源贡献与需求的比率，而不是简单的目标数量比率
        total_demand = np.sum([t.resources for t in env.targets], axis=0)
        total_demand_safe = np.maximum(total_demand, 1e-6)
        
        # 计算实际资源贡献 - 修复：确保使用float类型避免数据类型不匹配
        total_contribution = np.zeros_like(total_demand, dtype=np.float64)
        for target in env.targets:
            target_contribution = target.resources - target.remaining_resources
            total_contribution += target_contribution.astype(np.float64)
        
            # 【修改】使用“总贡献/总需求”的标准方法计算完成率
            total_demand_sum = np.sum(total_demand)
            total_contribution_sum = np.sum(np.minimum(total_contribution, total_demand))
            completion_rate = total_contribution_sum / total_demand_sum if total_demand_sum > 0 else 1.0
       
        # 添加调试信息：显示动作序列
        if self.config.ENABLE_DEBUG:
            print(f"\n[DEBUG] 集成推理完成，动作序列: {action_sequence}")
            print(f"[DEBUG] 总步数: {step_count}, 总奖励: {total_reward:.2f}, 完成率: {completion_rate:.4f}")
        
        # 保存推理完成后的UAV状态，用于资源利用率计算
        final_uav_states = []
        for uav in env.uavs:
            final_uav_states.append({
                'id': uav.id,
                'initial_resources': uav.initial_resources.copy(),
                'final_resources': uav.resources.copy()
            })
        
        return {
            'total_reward': total_reward,
            'completion_rate': completion_rate,
            'step_count': step_count,
            'action_sequence': action_sequence,
            'final_state': state,
            'ensemble_size': len(networks),
            'final_uav_states': final_uav_states
        }
    
    def _process_evaluation_results(self, results):
        """处理评估结果"""
        if results is None:
            print("评估失败，无有效结果")
            return
        
        self.evaluation_stats['scenario_results'].append(results)
        self.evaluation_stats['average_completion_rate'] = results['completion_rate']
        
        # 计算效率（奖励/步数）
        efficiency = results['total_reward'] / max(results['step_count'], 1)
        self.evaluation_stats['average_efficiency'] = efficiency
        
        print(f"\n评估结果:"
            f"  总奖励: {results['total_reward']:.2f}"
            f"  满足率: {results['completion_rate']:.3f}"
            f"  步数: {results['step_count']}"
            f"  效率: {efficiency:.2f}")
        
        if 'ensemble_size' in results:
            print(f"  集成规模: {results['ensemble_size']}")
    



def start_evaluation(config: Config, model_paths: Union[str, List[str]], scenario_name: str = "small"):
    """
    评估入口函数
    
    Args:
        config (Config): 配置对象
        model_paths (Union[str, List[str]]): 模型路径
        scenario_name (str): 场景名称
    """
    evaluator = ModelEvaluator(config)
    evaluator.start_evaluation(model_paths, scenario_name)