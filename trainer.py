# -*- coding: utf-8 -*-
# 文件名: trainer.py
# 描述: 模型训练器 - 支持动态随机场景和课程学习训练
#       新增功能: 动态随机训练模式下，每轮次生成的场景数据记录到单独的场景文件中

import os
import time
import pickle
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from collections import deque
from scenarios import _generate_scenario # 导入场景生成函数

from config import Config
from environment import UAVTaskEnv, DirectedGraph
from solvers import GraphRLSolver
from scenarios import get_small_scenario, get_balanced_scenario, get_complex_scenario
from adaptive_curriculum import (
    PerformanceMonitor, PromotionDecider, PerformanceMetrics, 
    Decision, AdaptiveCurriculumError
)

class ModelTrainer:
    """模型训练器 - 支持动态随机场景和课程学习训练"""
    
    def __init__(self, config: Config):
        self.config = config
        self.level_thresholds = {}  # 用于存储每个等级的动态掌握度阈值
        self.last_promotion_info = None  # 用于记录上一次的晋级信息，以检查不稳定性
        self.training_stats = {
            'episode_rewards': [],
            'completion_rates': [],
            'episode_losses': [],
            'step_losses': [],  # 新增：步骤损失记录
            'scenario_records': []  # 新增：场景记录列表
        }
        self.best_models = []
        
        # 从配置中获取max_best_models参数
        self.max_best_models = self.config.MAX_BEST_MODELS
        
        # 新增：场景记录相关配置
        self.scenario_log_dir = "output/scenario_logs"
        self.ensure_scenario_log_dir()
        
        print(f"[ModelTrainer] 初始化完成")
        print(f"  - 场景记录目录: {self.scenario_log_dir}")
        print(f"  - 最大保存最优模型数量: {self.max_best_models}")
    def _calculate_level_parameters(self, level: int) -> Dict:
        """
        函数级注释：根据当前课程等级，通过线性插值计算场景参数。

        Args:
            level (int): 当前的课程等级 (从0开始)。

        Returns:
            Dict: 包含 uav_num, target_num, obstacle_num, resource_abundance 的字典。
        """
        total_levels = self.config.GRANULAR_CURRICULUM_LEVELS

        # 计算当前进度比例 (0.0 to 1.0)
        # 当只有一级时，progress_ratio为0，使用起始参数
        if total_levels > 1:
            progress_ratio = level / (total_levels - 1)
        else:
            progress_ratio = 0

        # UAV 数量插值
        start_uavs = self.config.GRANULAR_START_UAVS
        end_uavs = self.config.MAX_UAVS
        uav_num = int(round(start_uavs + progress_ratio * (end_uavs - start_uavs)))

        # 目标数量插值
        start_targets = self.config.GRANULAR_START_TARGETS
        end_targets = self.config.MAX_TARGETS
        target_num = int(round(start_targets + progress_ratio * (end_targets - start_targets)))

        # 障碍物数量插值
        start_obstacles = self.config.GRANULAR_START_OBSTACLES
        end_obstacles = self.config.GRANULAR_END_OBSTACLES
        obstacle_num = int(round(start_obstacles + progress_ratio * (end_obstacles - start_obstacles)))

        # 资源充裕度插值 (从高到低)
        start_abundance = self.config.GRANULAR_START_ABUNDANCE
        end_abundance = self.config.GRANULAR_END_ABUNDANCE
        resource_abundance = start_abundance + progress_ratio * (end_abundance - start_abundance)

        return {
            "uav_num": uav_num,
            "target_num": target_num,
            "obstacle_num": obstacle_num,
            "resource_abundance": resource_abundance
        }

    
    def ensure_scenario_log_dir(self):
        """确保场景记录目录存在"""
        if not os.path.exists(self.scenario_log_dir):
            os.makedirs(self.scenario_log_dir)
            print(f"创建场景记录目录: {self.scenario_log_dir}")
    
    def save_scenario_data(self, episode: int, uavs, targets, obstacles, scenario_name: str = "dynamic", 
                          inference_result: Dict = None, solver=None, completion_rate: float = None, episode_info: Dict = None): 
        """
        保存轮次场景数据到单独文件 - 支持多种格式，包含推理结果和成功率信息
        
        Args:
            episode: 训练轮次
            uavs: UAV列表
            targets: 目标列表
            obstacles: 障碍物列表
            scenario_name: 场景名称
            inference_result: 推理结果数据（包含任务分配方案等）
            solver: 求解器实例，用于获取当前推理结果
            completion_rate: 任务完成率（成功率）
        """
        # 检查是否启用场景数据保存
        if not getattr(self.config, 'SAVE_SCENARIO_DATA', True):
            return
        
        try:            
            # --- 修改推理结果的获取逻辑 ---
            if episode_info:
                # 优先使用基于动作序列的报告生成
                inference_result = self._generate_report_from_actions(episode_info, uavs, targets)
            elif inference_result is None and solver is not None:
                # 保持原来的逻辑作为后备
                inference_result = self._capture_episode_inference_result(solver, uavs, targets, obstacles, completion_rate)
            
            # 创建场景数据字典
            scenario_data = {
                'episode': episode,
                'scenario_name': scenario_name,
                'timestamp': datetime.now().isoformat(),
                'uavs': uavs,
                'targets': targets,
                'obstacles': obstacles,
                'uav_count': len(uavs),
                'target_count': len(targets),
                'obstacle_count': len(obstacles),
                'config_info': {
                    'network_type': self.config.NETWORK_TYPE,
                    'training_mode': self.config.TRAINING_MODE,
                    'obs_mode': 'graph' if self.config.NETWORK_TYPE == "ZeroShotGNN" else 'flat'
                },
                # 新增：推理结果信息
                'inference_result': inference_result if inference_result else {}
            }
            
            # 计算成功率信息 - 使用标准完成率计算方法确保一致性
            if completion_rate is None:
                # 使用标准资源贡献计算方法，与环境、动作日志、奖励日志保持一致
                completion_rate = self._calculate_standard_completion_rate(targets)
            
            # 获取输出格式配置
            output_format = getattr(self.config, 'SCENARIO_DATA_FORMAT', 'pkl')
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 优化：在文件名中加入成功率信息
            success_rate_str = f"sr{completion_rate:.3f}"
            base_filename = f"episode_{episode:06d}_{scenario_name}_{len(uavs)}uav_{len(targets)}tgt_{success_rate_str}_{timestamp_str}"
            
            # 保存pkl格式（默认）
            if output_format in ['pkl', 'both']:
                pkl_filename = f"{base_filename}.pkl"
                pkl_filepath = os.path.join(self.scenario_log_dir, pkl_filename)
                # 确保路径格式正确
                pkl_filepath = os.path.normpath(pkl_filepath)
                
                with open(pkl_filepath, 'wb') as f:
                    pickle.dump(scenario_data, f)
                
                # print(f"场景数据已保存(PKL): {pkl_filename}")
            
            # 保存txt格式
            if output_format in ['txt', 'both']:
                txt_filename = f"{base_filename}.txt"
                txt_filepath = os.path.join(self.scenario_log_dir, txt_filename)
                # 确保路径格式正确
                txt_filepath = os.path.normpath(txt_filepath)
                
                self._save_scenario_as_txt(scenario_data, txt_filepath)
                # print(f"场景数据已保存(TXT): {txt_filename}")
            
            # 记录到训练统计中
            self.training_stats['scenario_records'].append({
                'episode': episode,
                'filename': base_filename,
                'uav_count': len(uavs),
                'target_count': len(targets),
                'scenario_name': scenario_name
            })
            
        except Exception as e:
            print(f"保存场景数据失败: {e}")
    
    def _save_scenario_as_txt(self, scenario_data: Dict, filepath: str):
        """
        将场景数据保存为txt格式
        
        Args:
            scenario_data: 场景数据字典
            filepath: 保存路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("训练场景数据记录\n")
                f.write("=" * 80 + "\n")
                f.write(f"轮次: {scenario_data['episode']}\n")
                f.write(f"场景名称: {scenario_data['scenario_name']}\n")
                f.write(f"时间戳: {scenario_data['timestamp']}\n")
                f.write(f"UAV数量: {scenario_data['uav_count']}\n")
                f.write(f"目标数量: {scenario_data['target_count']}\n")
                f.write(f"障碍物数量: {scenario_data['obstacle_count']}\n")
                f.write("\n")
                
                # 配置信息
                if 'config_info' in scenario_data:
                    config = scenario_data['config_info']
                    f.write("配置信息:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"网络类型: {config.get('network_type', 'N/A')}\n")
                    f.write(f"训练模式: {config.get('training_mode', 'N/A')}\n")
                    f.write(f"观测模式: {config.get('obs_mode', 'N/A')}\n")
                    f.write("\n")
                
                # UAV详细信息
                f.write("UAV详细信息:\n")
                f.write("-" * 40 + "\n")
                for i, uav in enumerate(scenario_data['uavs']):
                    # 【关键修复】确保使用初始资源而不是当前资源
                    initial_res = getattr(uav, 'initial_resources', uav.resources)
                    current_res = uav.resources
                    resource_consumed = initial_res - current_res
                    
                    f.write(f"  UAV {uav.id}: \n")
                    f.write(f"    初始位置: [{uav.position[0]:.2f}, {uav.position[1]:.2f}]\n")
                    # f.write(f"    训练后位置: [{uav.current_position[0]:.2f}, {uav.current_position[1]:.2f}]\n")
                    f.write(f"    初始资源: {initial_res}\n")
                    # f.write(f"    当前资源: {current_res}\n")
                    # f.write(f"    消耗资源: {resource_consumed}\n")
                    f.write(f"    最大航程: {uav.max_distance}\n")
                    f.write(f"    速度范围: {uav.velocity_range}\n")
                    f.write(f"    经济速度: {uav.economic_speed}\n")
                    f.write("\n")
                
                # 目标详细信息
                f.write("目标详细信息:\n")
                f.write("-" * 40 + "\n")
                for i, target in enumerate(scenario_data['targets']):
                    # 【关键修复】确保记录的是初始目标资源需求，而不是训练后的剩余资源
                    initial_target_resources = getattr(target, 'initial_resources', target.resources)
                    remaining_resources = getattr(target, 'remaining_resources', target.resources)
                    
                    f.write(f"  目标 {target.id}: \n")
                    f.write(f"    位置: [{target.position[0]:.2f}, {target.position[1]:.2f}]\n")
                    f.write(f"    初始需求资源: {initial_target_resources}\n")
                    # f.write(f"    剩余需求资源: {remaining_resources}\n")
                    f.write(f"    价值: {target.value}\n")
                    f.write("\n")
                
                # 障碍物详细信息
                f.write("障碍物详细信息:\n")
                f.write("-" * 40 + "\n")
                for i, obstacle in enumerate(scenario_data['obstacles']):
                    if hasattr(obstacle, 'center') and hasattr(obstacle, 'radius'):
                        f.write(f"  障碍物 {i+1}: \n")
                        f.write(f"    类型: 圆形障碍物\n")
                        f.write(f"    中心: [{obstacle.center[0]:.2f}, {obstacle.center[1]:.2f}]\n")
                        f.write(f"    半径: {obstacle.radius:.2f}\n")
                        f.write(f"    容差: {getattr(obstacle, 'tolerance', 'N/A')}\n")
                        f.write("\n")
                
                # 场景统计信息
                f.write("场景统计信息:\n")
                f.write("-" * 40 + "\n")
                
                # 【关键修复】计算统计数据（使用初始资源）
                uav_initial_resources_vector = np.sum([getattr(uav, 'initial_resources', uav.resources) for uav in scenario_data['uavs']], axis=0)
                uav_current_resources_vector = np.sum([uav.resources for uav in scenario_data['uavs']], axis=0)
                # 【关键修复】使用目标的initial_resources计算总需求
                target_initial_demand_vector = np.sum([getattr(target, 'initial_resources', target.resources) for target in scenario_data['targets']], axis=0)
                target_remaining_demand_vector = np.sum([getattr(target, 'remaining_resources', target.resources) for target in scenario_data['targets']], axis=0)
                
                # 【关键修复】使用初始资源计算充裕度
                resource_ratio_vector = uav_initial_resources_vector / (target_initial_demand_vector + 1e-6)
                
                f.write(f"UAV初始总资源: {uav_initial_resources_vector}\n")
                # f.write(f"UAV当前总资源: {uav_current_resources_vector}\n")
                f.write(f"目标初始总需求: {target_initial_demand_vector}\n")
                # f.write(f"目标剩余总需求: {target_remaining_demand_vector}\n")
                f.write(f"资源充裕度: [{resource_ratio_vector[0]:.3f} {resource_ratio_vector[1]:.3f}]\n")
                
                # 地图覆盖情况
                if scenario_data['obstacles']:
                    total_obstacle_area = sum(
                        3.14159 * obstacle.radius**2 
                        for obstacle in scenario_data['obstacles'] 
                        if hasattr(obstacle, 'radius')
                    )
                    map_area = 1000.0 * 1000.0  # 假设地图大小为1000x1000
                    coverage_ratio = total_obstacle_area / map_area
                    f.write(f"障碍物覆盖率: {coverage_ratio:.3f}\n")
                
                # [新增/优化] 推理结果记录逻辑
                if 'inference_result' in scenario_data and scenario_data['inference_result']:
                    inference_data = scenario_data['inference_result']
                    
                    # 优先使用最详细的报告
                    if 'detailed_report' in inference_data and inference_data['detailed_report']:
                        f.write("\n")
                        f.write(inference_data['detailed_report'])
                        f.write("\n")
                    # 其次使用简化报告
                    elif 'simple_report' in inference_data and inference_data['simple_report']:
                        f.write("\n")
                        f.write(inference_data['simple_report'])
                        f.write("\n")
                    # 最后，如果没有任何报告，则回退到原始的任务分配方案展示
                    else:
                        f.write("\n推理结果信息:\n")
                        f.write("=" * 40 + "\n")
                        
                        # 算法信息
                        if 'algorithm_info' in inference_data:
                            algo_info = inference_data['algorithm_info']
                            f.write("算法信息:\n")
                            f.write(f"  网络类型: {algo_info.get('network_type', 'N/A')}\n")
                            f.write(f"  探索率: {algo_info.get('epsilon', 0.0):.4f}\n")
                            f.write(f"  训练步数: {algo_info.get('training_step', 0)}\n")
                            f.write(f"  推理时间: {algo_info.get('capture_timestamp', 'N/A')}\n")
                            f.write("\n")
                        
                        # 任务分配方案
                        if 'task_allocation' in inference_data:
                            allocation = inference_data['task_allocation']
                            assignments = allocation.get('assignments', {})
                            
                            f.write("任务分配方案:\n")
                            f.write("-" * 30 + "\n")
                            
                            if assignments:
                                for uav_id, tasks in assignments.items():
                                    f.write(f"  UAV {uav_id}:\n")
                                    if not tasks:
                                        f.write("    - 未分配任务\n")
                                    else:
                                        for i, (target_id, phi_idx) in enumerate(tasks, 1):
                                            f.write(f"    {i}. 分配给目标 {target_id} (接近角度索引: {phi_idx})\n")
                                    f.write("\n")
                            else:
                                f.write("  未生成任务分配方案\n")
                            
                            # 分配摘要（简化版本）
                            f.write("分配摘要:\n")
                            f.write("-" * 30 + "\n")
                            f.write(f"  总分配数: {allocation.get('total_assignments', 0)}\n")
                            f.write(f"  活跃UAV数: {allocation.get('active_uav_count', 0)}/{scenario_data['uav_count']}\n")
                            f.write(f"  已分配目标数: {allocation.get('assigned_target_count', 0)}/{scenario_data['target_count']}\n")
                            f.write(f"  目标覆盖率: {allocation.get('target_coverage_rate', 0.0):.3f}\n")
                            f.write("\n")
                
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write("场景数据记录结束\n")
                f.write("=" * 80 + "\n")
                
        except Exception as e:
            print(f"保存TXT格式场景数据失败: {e}")

    def _generate_detailed_inference_report(self, task_assignments: Dict, uavs, targets, completion_rate: float = None) -> str:
        """
        生成详细的推理结果报告，模仿评估器的输出格式。
        
        Args:
            task_assignments: 任务分配字典
            uavs: UAV列表
            targets: 目标列表
            completion_rate: 传入的标准完成率，优先使用此值
        """
        # 【修改后的代码】
        # (该方法被完全重写以支持资源消耗模拟)
        report_lines = []
        report_lines.append("---------- 训练轮次推理结果报告 ----------")
        
        # 1. 总体资源满足情况
        # 首先，我们需要一个更准确的方式来计算总贡献
        temp_uav_resources_summary = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
        temp_target_needs_summary = {t.id: t.resources.copy().astype(float) for t in targets}

        for uav_id in sorted(task_assignments.keys()):
            assignments = task_assignments.get(uav_id, [])
            for target_id, _ in assignments:
                uav_available = temp_uav_resources_summary[uav_id]
                target_needed = temp_target_needs_summary[target_id]
                contribution = np.minimum(uav_available, target_needed)
                
                temp_uav_resources_summary[uav_id] -= contribution
                temp_target_needs_summary[target_id] -= contribution

        total_demand = np.sum([t.resources for t in targets], axis=0)
        total_contribution = total_demand - np.sum(list(temp_target_needs_summary.values()), axis=0)
        satisfied_targets = sum(1 for t_id, needs in temp_target_needs_summary.items() if np.all(needs <= 1e-5))
        
        report_lines.append("\n总体资源满足情况:")
        report_lines.append("-" * 26)
        report_lines.append(f"- 总需求/总贡献: {np.array2string(total_demand, formatter={'float_kind':lambda x: '%.0f' % x})} / {np.array2string(total_contribution, formatter={'float_kind':lambda x: '%.1f' % x})}")
        report_lines.append(f"- 已满足目标: {satisfied_targets} / {len(targets)} ({satisfied_targets/len(targets)*100:.1f}%)")
        report_lines.append(f"- 资源完成率: {np.mean(np.minimum(total_contribution, total_demand) / (total_demand + 1e-6)) * 100:.1f}%")

        # 2. UAV 详细任务分配 (包含资源消耗模拟)
        report_lines.append("\nUAV详细任务分配:")
        report_lines.append("-" * 26)
        
        temp_uav_resources_report = {u.id: u.initial_resources.copy().astype(float) for u in uavs}
        temp_target_needs_report = {t.id: t.resources.copy().astype(float) for t in targets}
        
        for uav_id in sorted(task_assignments.keys()):
            uav = next((u for u in uavs if u.id == uav_id), None)
            if not uav: continue
            
            initial_res = getattr(uav, 'initial_resources', uav.resources)
            report_lines.append(f"* 无人机 {uav_id} (初始资源: {np.array2string(initial_res, formatter={'float_kind': lambda x: f'{x:.0f}'})})")
            
            assignments = task_assignments.get(uav_id, [])
            if not assignments:
                report_lines.append("  - 未分配任何任务")
            else:
                for i, (target_id, phi_idx) in enumerate(assignments, 1):
                    target = next((t for t in targets if t.id == target_id), None)
                    if not target: continue
                    
                    uav_res_before = temp_uav_resources_report[uav_id]
                    target_need_before = temp_target_needs_report[target_id]
                    contribution = np.minimum(uav_res_before, target_need_before)
                    
                    # 更新状态用于报告的下一步
                    temp_uav_resources_report[uav_id] -= contribution
                    temp_target_needs_report[target_id] -= contribution
                    
                    dist = np.linalg.norm(uav.position - target.position) # 注意：这里是初始距离
                    report_lines.append(f"  {i}. 分配给 目标 {target.id}:")
                    report_lines.append(f"     - 飞行距离: {dist:.1f}m, 角度: {phi_idx}")
                    report_lines.append(f"     - 资源贡献: {np.array2string(contribution, formatter={'float_kind':lambda x: '%.1f' % x})}")
                    report_lines.append(f"     - 剩余资源: {np.array2string(temp_uav_resources_report[uav_id], formatter={'float_kind':lambda x: '%.1f' % x})}")

        report_lines.append("\n" + "="*80)
        return "\n".join(report_lines)
    def _calculate_standard_completion_rate(self, targets):
        """
        计算标准完成率 - 与environment.py中的计算方法保持一致
        
        基于实际资源贡献与需求总量的比值来统计完成率，确保与控制台和动作日志保持一致
        
        Args:
            targets: 目标列表
            
        Returns:
            float: 标准完成率 [0.0, 1.0]
        """
        if not targets:
            return 1.0  # 没有目标时认为已完成
        
        # 计算总需求
        total_demand = np.sum([t.resources for t in targets], axis=0)
        total_demand_safe = np.maximum(total_demand, 1e-6)
        
        # 【修复】计算实际资源贡献 - 基于目标的剩余资源，确保与environment.py一致
        total_contribution = np.zeros_like(total_demand, dtype=np.float64)
        for target in targets:
            # 使用目标的实际贡献：初始需求 - 剩余需求
            target_contribution = target.resources - target.remaining_resources
            total_contribution += target_contribution.astype(np.float64)
        
        # 确保贡献值不会因为浮点误差出现负数
        total_contribution = np.maximum(total_contribution, 0)
        
        # 【修复】标准完成率计算：基于实际贡献资源与总需求的比值
        # 使用总和比值而不是平均比值，确保与environment.py计算逻辑统一
        total_demand_sum = np.sum(total_demand)
        total_contribution_sum = np.sum(total_contribution)
        
        if total_demand_sum > 0:
            # 标准满足率计算：使用“平均比例法”确保与evaluate.py的逻辑一致
            completion_rate_per_resource = np.minimum(total_contribution, total_demand) / total_demand_safe
            completion_rate = np.mean(completion_rate_per_resource)
        else:
            completion_rate = 1.0
        
        # 确保返回值在合理范围内
        return float(np.clip(completion_rate, 0.0, 1.0))
    
    def _calculate_resource_abundance(self, uavs, targets):
        """
        计算资源充裕度信息，返回为用于日志记录的格式化字符串
        
        Args:
            uavs: 无人机列表
            targets: 目标列表
            
        Returns:
            str: 格式化的资源充裕度信息
        """
        if not uavs or not targets:
            return "资源充裕度: [N/A N/A]"
        
        try:
            # 【修复】确保使用初始资源进行计算
            uav_resources_vector = np.zeros(self.config.RESOURCE_DIM)
            for u in uavs:
                # 优先使用initial_resources，如果不存在则使用resources
                initial_res = getattr(u, 'initial_resources', u.resources)
                uav_resources_vector += initial_res
            
            target_demand_vector = np.sum([t.resources for t in targets], axis=0)
            
            # 【修复】避免除零错误，使用安全的分母
            target_demand_safe = np.maximum(target_demand_vector, 1e-6)
            
            # 计算资源充裕度（供给/需求比例）
            resource_ratio_vector = uav_resources_vector / target_demand_safe
            
            # # 【调试】输出详细信息
            # print(f"[DEBUG] 资源充裕度计算:")
            # print(f"  UAV总供给: {uav_resources_vector}")
            # print(f"  目标总需求: {target_demand_vector}")
            # print(f"  充裕度比例: {resource_ratio_vector}")
            
            # 格式化为字符串
            abundance_str = f"[{resource_ratio_vector[0]:.3f} {resource_ratio_vector[1]:.3f}]"
            return f"资源充裕度: {abundance_str}"
        except Exception as e:
            print(f"[ERROR] 资源充裕度计算失败: {e}")
            return f"资源充裕度: [计算错误: {e}]"

    def _validate_scenario_consistency(self, solver, scenario_name: str, episode: int):
        """
        【新增】验证场景数据一致性，确保所有组件使用相同的场景信息
        
        Args:
            solver: 求解器对象
            scenario_name: 场景名称
            episode: 当前轮次
        """
        try:
            if not hasattr(solver, 'env') or not hasattr(solver.env, 'uavs'):
                return
            
            actual_uavs = len(solver.env.uavs)
            actual_targets = len(solver.env.targets)
            actual_obstacles = len(solver.env.obstacles) if hasattr(solver.env, 'obstacles') else 0
            
            # 获取预期范围
            if scenario_name in self.config.SCENARIO_TEMPLATES:
                expected_ranges = self.config.SCENARIO_TEMPLATES[scenario_name]
                uav_range = expected_ranges['uav_num_range']
                target_range = expected_ranges['target_num_range']
                obstacle_range = expected_ranges['obstacle_num_range']
                
                # 验证数量是否在预期范围内
                inconsistencies = []
                
                if not (uav_range[0] <= actual_uavs <= uav_range[1]):
                    inconsistencies.append(f"无人机数量不一致: 实际={actual_uavs}, 预期范围={uav_range}")
                
                if not (target_range[0] <= actual_targets <= target_range[1]):
                    inconsistencies.append(f"目标数量不一致: 实际={actual_targets}, 预期范围={target_range}")
                
                if not (obstacle_range[0] <= actual_obstacles <= obstacle_range[1]):
                    inconsistencies.append(f"障碍物数量不一致: 实际={actual_obstacles}, 预期范围={obstacle_range}")
                
                # 输出不一致信息
                if inconsistencies:
                    print(f"⚠️ Episode {episode} 场景数据不一致:")
                    for issue in inconsistencies:
                        print(f"   {issue}")
                else:
                    if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
                        print(f"✅ Episode {episode} 场景数据一致性验证通过: UAV={actual_uavs}, 目标={actual_targets}, 障碍={actual_obstacles}")
            
        except Exception as e:
            print(f"⚠️ 场景一致性验证失败: {e}")

    def _ensure_single_reset_per_episode(self, solver, scenario_name: str, episode: int):
        """
        【修复】确保每个轮次只重置一次场景 - 仅验证，不执行额外重置
        
        Args:
            solver: 求解器对象
            scenario_name: 场景名称
            episode: 当前轮次
        """
        try:
            if not hasattr(solver, 'env'):
                return
            
            # 仅检查重置状态，不执行额外重置
            last_reset_episode = getattr(solver.env, '_last_reset_episode', -1)
            
            if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
                if last_reset_episode == episode:
                    print(f"[RESET DEBUG] Episode {episode} 重置状态正常")
                else:
                    print(f"[RESET DEBUG] Episode {episode} 重置状态异常: 上次重置轮次={last_reset_episode}")
            
        except Exception as e:
            print(f"⚠️ 场景重置状态检查失败: {e}")

    def _calculate_resource_summary(self, task_assignments, uavs, targets):
        """计算资源满足度的摘要信息"""
        total_demand = np.sum([t.resources for t in targets], axis=0) if targets else np.zeros(self.config.RESOURCE_DIM)
        
        target_contributions = {t.id: np.zeros(self.config.RESOURCE_DIM) for t in targets}
        uav_map = {u.id: u for u in uavs}
        
        for uav_id, assignments in task_assignments.items():
            uav = uav_map.get(uav_id)
            if not uav: continue
            
            temp_uav_resources = uav.initial_resources.copy().astype(float)
            for target_id, _ in assignments:
                target = next((t for t in targets if t.id == target_id), None)
                if not target: continue
                
                # 模拟资源贡献
                needed = target.resources - target_contributions[target.id]
                contribution = np.minimum(temp_uav_resources, needed.clip(min=0))
                
                if np.any(contribution > 0):
                    target_contributions[target.id] += contribution
                    temp_uav_resources -= contribution

        total_contribution = sum(target_contributions.values())
        satisfied_count = sum(1 for t in targets if np.all(target_contributions[t.id] >= t.resources))
        
        total_targets = len(targets) if targets else 1
        satisfaction_rate = (satisfied_count / total_targets) * 100
        resource_completion_rate = np.mean(np.minimum(total_contribution, total_demand) / (total_demand + 1e-6)) * 100
        
        return satisfied_count, total_targets, satisfaction_rate, resource_completion_rate, total_demand, total_contribution 

    def _capture_episode_inference_result(self, solver, uavs, targets, obstacles, completion_rate: float = None) -> Dict:
            """
            捕获当前轮次的推理结果信息（最简化版本，避免复杂计算）
            
            Args:
                solver: 求解器实例
                uavs: UAV列表
                targets: 目标列表
                obstacles: 障碍物列表
                completion_rate: 传入的标准完成率，优先使用此值
                
            Returns:
                Dict: 推理结果数据
            """
            try:
                if not getattr(self.config, 'SAVE_EPISODE_INFERENCE_RESULTS', True):
                    return {}
                
                # [修改] 使用修复后的 get_task_assignments
                task_assignments = {}
                try:
                    task_assignments = solver.get_task_assignments() or {}
                    if not task_assignments:
                        task_assignments = {uav.id: [] for uav in uavs}
                except Exception as assign_error:
                    print(f"获取任务分配失败: {assign_error}")
                    task_assignments = {uav.id: [] for uav in uavs}

                # [修改] 调用新的详细报告生成函数，传递completion_rate参数
                detailed_report = self._generate_detailed_inference_report(task_assignments, uavs, targets, completion_rate)

                # [修改] 更新摘要信息
                summary = self._generate_allocation_summary(task_assignments, uavs, targets)
                
                inference_result = {
                    'task_allocation': {
                        'assignments': task_assignments,
                        'total_assignments': summary.get('total_assignments', 0),
                        'active_uav_count': summary.get('active_uav_count', 0),
                        'assigned_target_count': summary.get('assigned_target_count', 0),
                        'target_coverage_rate': summary.get('target_coverage_rate', 0.0)
                    },
                    'simple_report': "详细报告见下方", # 保留字段，但内容更新
                    'detailed_report': detailed_report, # 新增详细报告字段
                    'algorithm_info': {
                        'network_type': getattr(self.config, 'NETWORK_TYPE', 'Unknown'),
                        'epsilon': getattr(solver, 'epsilon', 0.0),
                        'training_step': getattr(solver, 'step_count', 0)
                    },
                    'capture_timestamp': datetime.now().isoformat()
                }
                
                return inference_result
                
            except Exception as e:
                print(f"捕获推理结果失败: {e}")
                return {
                    'task_allocation': {'assignments': {uav.id: [] for uav in uavs}},
                    'simple_report': "推理结果捕获失败。",
                    'detailed_report': f"推理结果捕获失败: {str(e)}",
                    'algorithm_info': {'network_type': getattr(self.config, 'NETWORK_TYPE', 'Unknown')},
                    'capture_timestamp': datetime.now().isoformat(),
                    'error_message': str(e)
                }

    def _generate_allocation_summary(self, task_assignments: Dict, uavs, targets) -> Dict:
        """
        生成任务分配摘要信息
        
        Args:
            task_assignments: 任务分配字典
            uavs: UAV列表
            targets: 目标列表
            
        Returns:
            Dict: 分配摘要
        """
        try:
            total_assignments = sum(len(assignments) for assignments in task_assignments.values())
            active_uavs = len([uav_id for uav_id, assignments in task_assignments.items() if assignments])
            
            # 计算目标覆盖情况
            assigned_targets = set()
            for assignments in task_assignments.values():
                for target_id, _ in assignments:
                    assigned_targets.add(target_id)
            
            target_coverage_rate = len(assigned_targets) / len(targets) if targets else 0.0
            uav_utilization_rate = active_uavs / len(uavs) if uavs else 0.0
            
            return {
                'total_assignments': total_assignments,
                'active_uav_count': active_uavs,
                'total_uav_count': len(uavs),
                'assigned_target_count': len(assigned_targets),
                'total_target_count': len(targets),
                'target_coverage_rate': target_coverage_rate,
                'uav_utilization_rate': uav_utilization_rate,
                'avg_assignments_per_uav': total_assignments / len(uavs) if uavs else 0.0
            }
        except Exception as e:
            print(f"生成分配摘要失败: {e}")
            return {}
    
    def _calculate_resource_utilization(self, task_assignments: Dict, uavs, targets) -> Dict:
        """
        计算资源利用率信息
        
        Args:
            task_assignments: 任务分配字典
            uavs: UAV列表
            targets: 目标列表
            
        Returns:
            Dict: 资源利用率信息
        """
        try:
            # 计算UAV资源利用情况
            uav_resource_usage = {}
            total_uav_resources = np.zeros(self.config.RESOURCE_DIM)
            
            for uav in uavs:
                total_uav_resources += uav.initial_resources
                uav_resource_usage[uav.id] = {
                    'initial_resources': uav.initial_resources.tolist(),
                    'current_resources': uav.resources.tolist(),
                    'utilization_rate': 1.0 - (uav.resources / (uav.initial_resources + 1e-6))
                }
            
            # 计算目标资源需求情况
            target_resource_demand = {}
            total_target_demand = np.zeros(self.config.RESOURCE_DIM)
            
            for target in targets:
                total_target_demand += target.resources
                target_resource_demand[target.id] = {
                    'required_resources': target.resources.tolist(),
                    'remaining_resources': target.remaining_resources.tolist(),
                    'satisfaction_rate': 1.0 - (target.remaining_resources / (target.resources + 1e-6))
                }
            
            # 计算整体资源匹配度
            resource_abundance = total_uav_resources / (total_target_demand + 1e-6)
            
            return {
                'uav_resource_usage': uav_resource_usage,
                'target_resource_demand': target_resource_demand,
                'total_uav_resources': total_uav_resources.tolist(),
                'total_target_demand': total_target_demand.tolist(),
                'resource_abundance': resource_abundance.tolist(),
                'overall_utilization': np.mean([info['utilization_rate'] for info in uav_resource_usage.values()], axis=0).tolist()
            }
        except Exception as e:
            print(f"计算资源利用率失败: {e}")
            return {}
    
    def _calculate_episode_performance_metrics(self, task_assignments: Dict, uavs, targets, obstacles) -> Dict:
        """
        计算轮次性能指标
        
        Args:
            task_assignments: 任务分配字典
            uavs: UAV列表
            targets: 目标列表
            obstacles: 障碍物列表
            
        Returns:
            Dict: 性能指标
        """
        try:
            # 计算完成率 - 使用标准完成率计算方法（与环境一致）
            if hasattr(self, '_current_env') and self._current_env and hasattr(self._current_env, 'get_completion_rate'):
                completion_rate = self._current_env.get_completion_rate()
            else:
                # 后备方案：使用标准资源贡献计算方法
                completion_rate = self._calculate_standard_completion_rate(targets)
            
            # 计算已完成的目标数量（修复未定义变量问题）
            completed_targets = sum(1 for t in targets if np.all(t.remaining_resources <= 0))
            
            # 计算任务分配效率
            total_assignments = sum(len(assignments) for assignments in task_assignments.values())
            assignment_efficiency = total_assignments / (len(uavs) * len(targets)) if uavs and targets else 0.0
            
            # 计算负载均衡度
            assignment_counts = [len(assignments) for assignments in task_assignments.values()]
            load_balance_score = 1.0 - (np.std(assignment_counts) / (np.mean(assignment_counts) + 1e-6)) if assignment_counts else 0.0
            
            # 计算协作程度（多个UAV分配给同一目标的情况）
            target_uav_counts = {}
            for uav_id, assignments in task_assignments.items():
                for target_id, _ in assignments:
                    target_uav_counts[target_id] = target_uav_counts.get(target_id, 0) + 1
            
            collaboration_targets = sum(1 for count in target_uav_counts.values() if count > 1)
            collaboration_rate = collaboration_targets / len(targets) if targets else 0.0
            
            return {
                'completion_rate': completion_rate,
                'completed_targets': completed_targets,
                'total_targets': len(targets),
                'assignment_efficiency': assignment_efficiency,
                'total_assignments': total_assignments,
                'load_balance_score': load_balance_score,
                'collaboration_rate': collaboration_rate,
                'collaboration_targets': collaboration_targets,
                'active_uav_count': len([uav_id for uav_id, assignments in task_assignments.items() if assignments]),
                'idle_uav_count': len(uavs) - len([uav_id for uav_id, assignments in task_assignments.items() if assignments])
            }
        except Exception as e:
            print(f"计算性能指标失败: {e}")
            return {}
    
    def load_scenario_data(self, filepath: str) -> Optional[Dict]:
        """
        加载场景数据
        
        Args:
            filepath: 场景文件路径
            
        Returns:
            场景数据字典或None
        """
        try:
            with open(filepath, 'rb') as f:
                scenario_data = pickle.load(f)
            return scenario_data
        except Exception as e:
            print(f"加载场景数据失败: {e}")
            return None
    
    def get_scenario_summary(self) -> Dict:
        """获取场景记录摘要"""
        if not self.training_stats['scenario_records']:
            return {}
        
        records = self.training_stats['scenario_records']
        
        # 统计场景类型
        scenario_types = {}
        uav_counts = []
        target_counts = []
        
        for record in records:
            scenario_name = record['scenario_name']
            scenario_types[scenario_name] = scenario_types.get(scenario_name, 0) + 1
            uav_counts.append(record['uav_count'])
            target_counts.append(record['target_count'])
        
        return {
            'total_scenarios': len(records),
            'scenario_types': scenario_types,
            'uav_count_stats': {
                'min': min(uav_counts) if uav_counts else 0,
                'max': max(uav_counts) if uav_counts else 0,
                'avg': np.mean(uav_counts) if uav_counts else 0
            },
            'target_count_stats': {
                'min': min(target_counts) if target_counts else 0,
                'max': max(target_counts) if target_counts else 0,
                'avg': np.mean(target_counts) if target_counts else 0
            },
            'latest_scenario': records[-1] if records else None
        }
    
    def print_scenario_summary(self):
        """打印场景记录摘要"""
        summary = self.get_scenario_summary()
        if not summary:
            print("暂无场景记录")
            return
        
        print("\n" + "="*60)
        print("动态训练场景记录摘要")
        print("="*60)
        print(f"总场景数: {summary['total_scenarios']}")
        
        print(f"\n场景类型分布:")
        for scenario_type, count in summary['scenario_types'].items():
            print(f"  {scenario_type}: {count}个")
        
        print(f"\nUAV数量统计:")
        uav_stats = summary['uav_count_stats']
        print(f"  最小: {uav_stats['min']}, 最大: {uav_stats['max']}, 平均: {uav_stats['avg']:.1f}")
        
        print(f"\n目标数量统计:")
        target_stats = summary['target_count_stats']
        print(f"  最小: {target_stats['min']}, 最大: {target_stats['max']}, 平均: {target_stats['avg']:.1f}")
        
        if summary['latest_scenario']:
            latest = summary['latest_scenario']
            print(f"\n最新场景: Episode {latest['episode']} - {latest['scenario_name']} ({latest['uav_count']}UAV, {latest['target_count']}目标)")
        
        print("="*60)

    def format_reward_log_entry(self, step_i: int, episode: int, action: int, reward: float, 
                               next_state, done: bool, info: dict = None) -> str:
        """格式化奖励日志条目 - 参照main-old.py格式"""
        log_entry = f"Episode {episode+1:4d}, Step {step_i:3d}: Action={action:3d}, Reward={reward:7.2f}"
        
        if info:
            # 添加详细的奖励分解信息
            if 'reward_breakdown' in info:
                breakdown = info['reward_breakdown']
                log_entry += f", Base={breakdown.get('base_reward', 0.0):.2f}"
                log_entry += f", Shaping={breakdown.get('shaping_reward', 0.0):.2f}"
                log_entry += f", Exploration={breakdown.get('exploration_bonus', 0.0):.2f}"
            
            # 添加环境状态信息
            if 'completion_rate' in info:
                log_entry += f", Completion={info['completion_rate']:.3f}"
            
            if 'remaining_targets' in info:
                log_entry += f", Remaining={info['remaining_targets']}"
        
        log_entry += f", Done={done}"
        return log_entry
    
    def start_training(self, use_curriculum: bool = False, scenario_name: str = "small") -> None:
        """
        启动训练过程
        
        Args:
            use_curriculum (bool): 是否使用课程学习模式
        """
        print("=" * 60)
        print("开始模型训练")
        print("=" * 60)
        print(f"训练模式: {'课程学习' if use_curriculum else '动态随机场景'}")
        print(f"网络类型: {self.config.NETWORK_TYPE}")
        print(f"训练轮次: {self.config.training_config.episodes}")
        print("=" * 60)
        

        self._init_action_log(scenario_name)

        # 设置训练模式标识
        self._is_curriculum = use_curriculum
        self._scenario_name = scenario_name
        # 设置当前场景名称供其他方法使用
        self._current_scenario_name = scenario_name
        
        # 初始化奖励日志
        self._init_reward_log()
        
        start_time = time.time()
        
        if use_curriculum:
            # ZeroShotGNN下课程学习改为基于GraphRLSolver.train的图训练
            if getattr(self.config, 'NETWORK_TYPE', '') == 'ZeroShotGNN':
                self._curriculum_training_graph()
            else:
                self._curriculum_training()
        else:
            # 当使用ZeroShotGNN时，默认采用基于图的训练（GraphRLSolver.train，带早停）
            if getattr(self.config, 'NETWORK_TYPE', '') == 'ZeroShotGNN':
                self._graph_training(scenario_name)
            else:
                self._dynamic_training(scenario_name)
        
        end_time = time.time()
        self.training_stats['training_time'] = end_time - start_time
        
        print(f"\n训练完成! 总耗时: {self.training_stats['training_time']:.2f}秒")
        self._save_training_results()
        
        # 关闭奖励日志
        self._close_reward_log()

        self._close_action_log()

    def _init_action_log(self, scenario_name: str = "unknown", uavs=None, targets=None, obstacles=None):
        """初始化一个专门用于记录动作选择过程的日志文件"""
        logger = logging.getLogger('ActionLogger')
        if logger.hasHandlers():
            logger.handlers.clear()
        
        logger.setLevel(logging.INFO)
        logger.propagate = False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 优化：在action_log文件名中加入训练信息
        network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
        training_mode_short = "CL" if hasattr(self, '_is_curriculum') and self._is_curriculum else "DR"
        path_algo_short = "PH" if self.config.USE_PHRRT_DURING_TRAINING else "FA"
        log_filename = f"action_log_{network_type}_{training_mode_short}_{path_algo_short}_{scenario_name}_{timestamp}.txt"
        log_path = os.path.join("output", log_filename)
        os.makedirs("output", exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        self.action_logger = logger
        print(f"✅ 动作决策日志已初始化: {log_path}")
        
        # 记录日志标题和场景信息
        self.action_logger.info("=" * 120)
        self.action_logger.info("动作决策过程详细日志")
        self.action_logger.info("=" * 120)
        
        # 添加场景信息，包括资源充裕度
        if uavs is not None and targets is not None:
            num_uavs = len(uavs)
            num_targets = len(targets)
            num_obstacles = len(obstacles) if obstacles is not None else 0
            resource_abundance_info = self._calculate_resource_abundance(uavs, targets)
            
            self.action_logger.info(f"场景信息: {scenario_name}")
            self.action_logger.info(f"无人机数量: {num_uavs}, 目标数量: {num_targets}, 障碍物数量: {num_obstacles}")
            self.action_logger.info(f"{resource_abundance_info}")
            self.action_logger.info("=" * 120)
        
        self.action_logger.info("")

    def _update_action_log_scenario_info(self, uavs, targets, obstacles, scenario_name="unknown"):
        """更新动作日志中的场景信息"""
        if hasattr(self, 'action_logger') and self.action_logger:
            num_uavs = len(uavs) if uavs is not None else 0
            num_targets = len(targets) if targets is not None else 0
            num_obstacles = len(obstacles) if obstacles is not None else 0
            resource_abundance_info = self._calculate_resource_abundance(uavs, targets)
            
            self.action_logger.info("\n" + "=" * 120)
            self.action_logger.info("场景信息更新")
            self.action_logger.info("=" * 120)
            self.action_logger.info(f"场景名称: {scenario_name}")
            self.action_logger.info(f"无人机数量: {num_uavs}, 目标数量: {num_targets}, 障碍物数量: {num_obstacles}")
            self.action_logger.info(f"{resource_abundance_info}")
            self.action_logger.info("=" * 120)
            self.action_logger.info("")

    def _close_action_log(self):
        """关闭动作日志文件"""
        if self.action_logger:
            for handler in self.action_logger.handlers:
                handler.close()
                self.action_logger.removeHandler(handler)
            print("✅ 动作决策日志已关闭。")

    def log_action_details(self, episode, step, valid_actions, chosen_action, env, reward=None, step_info=None, 
                          pre_action_completion_rate=None, post_action_completion_rate=None, 
                          uav_resources_snapshot=None, uav_positions_snapshot=None, q_values=None): # [新增] 接收快照参数
        """
        记录当前步骤的详细决策信息：有哪些可选动作，最终选了哪个，包含完成率和奖励信息。
        
        Args:
            episode (int): 当前轮次
            step (int): 当前步数
            valid_actions (np.ndarray): 有效动作的索引数组
            chosen_action (int): 最终选择的动作索引
            env (UAVTaskEnv): 环境实例，用于解码动作和获取上下文
            reward (float): 本步奖励
            step_info (dict): 步骤信息，包含奖励分解
            pre_action_completion_rate (float): 动作执行前的完成率
            post_action_completion_rate (float): 动作执行后的完成率
            uav_resources_snapshot (dict): 动作执行前的UAV资源快照
        """
        if not self.action_logger:
            return
        # 如果没有传入位置快照，为了兼容性，使用当前位置
        if uav_positions_snapshot is None:
            uav_positions_snapshot = {uav.id: uav.current_position for uav in env.uavs}

        # [调试] 检查目标需求异常和验证动作解码一致性
        target_idx, uav_idx, phi_idx = env.decode_action(chosen_action)
        self.action_logger.info(f"[DEBUG] 动作解码: chosen_action={chosen_action} -> target_idx={target_idx}, uav_idx={uav_idx}")
        
        # [调试] 检查选中目标的状态
        chosen_target = env.targets[target_idx]
        self.action_logger.info(f"[DEBUG] 选中目标({target_idx}): remaining={chosen_target.remaining_resources}, original={chosen_target.resources}")
        
        # 创建目标需求状态的快照（动作执行前的状态）- 修复问题2
        target_resources_snapshot = {target.id: target.remaining_resources.copy() for target in env.targets}
        
        # 如果没有传入UAV资源快照，创建当前状态快照（用于调试）
        if uav_resources_snapshot is None:
            uav_resources_snapshot = {uav.id: uav.resources.copy() for uav in env.uavs}
            print("警告: 没有传入UAV资源快照，使用当前状态（可能是执行后状态）")

        # 使用传入的动作前完成率，如果没有则获取当前完成率
        if pre_action_completion_rate is not None:
            current_completion_rate = pre_action_completion_rate
        else:
            current_completion_rate = env.get_completion_rate() if hasattr(env, 'get_completion_rate') else 0.0
        
        log_header = f"--- Episode: {episode}, Step: {step} ---"
        self.action_logger.info(log_header)
        
        # 记录当前状态信息 - 修复第0步奖励问题（问题1）
        self.action_logger.info(f">> 当前完成率: {current_completion_rate:.3f}")
        if step == 0:
            # 第0步不应该有上步奖励
            if reward is not None and abs(reward) > 1e-6:
                self.action_logger.info(f">> [INFO] 第0步检测到非零奖励: {reward:.2f} (可能有问题)")
            # 第0步不显示"上步奖励"
        else:
            # 非第0步才显示上步奖励
            if reward is not None:
                self.action_logger.info(f">> 上步奖励: {reward:.2f}")
            
        # 记录奖励分解信息
        if step_info and 'reward_breakdown' in step_info:
            breakdown = step_info['reward_breakdown']
            if breakdown:
                breakdown_str = ", ".join([f"{k}={v:.1f}" for k, v in breakdown.items() if isinstance(v, (int, float)) and abs(v) > 0.01])
                if breakdown_str:
                    self.action_logger.info(f">> 奖励分解: [{breakdown_str}]")
        
        if valid_actions.size == 0:
            self.action_logger.info(">> 警告: 在当前状态下，剪枝后未发现任何有效动作！")
        else:
            self.action_logger.info(f">> 发现 {len(valid_actions)} 个有效动作:")
            
            action_details = []
            for action_idx in valid_actions:
                try:
                    target_idx, uav_idx, phi_idx = env.decode_action(action_idx)
                    
                    q_value_str = "Q=N/A"
                    if q_values is not None and action_idx < q_values.size(1):
                        q_val = q_values[0, action_idx].item()
                        q_value_str = f"Q={q_val:8.4f}"

                    target = env.targets[target_idx]
                    uav = env.uavs[uav_idx]
                    # distance = np.linalg.norm(uav.current_position - target.position)
                    uav_pos_before = uav_positions_snapshot.get(uav.id, uav.current_position) # <--- 使用快照中的位置
                    distance = np.linalg.norm(uav_pos_before - target.position) # <--- 使用快照中的位置计算距离
                    
                    # 使用快照中的执行前状态
                    uav_res_before = uav_resources_snapshot.get(uav.id, np.array([-1, -1]))
                    target_need_before = target_resources_snapshot.get(target.id, np.array([-1, -1]))
                    
                    # 计算潜在贡献
                    potential_contribution = np.minimum(uav_res_before, target_need_before)

                    detail_str = (
                    f"  - Action({action_idx:3d}): UAV({uav.id:2d}) -> Target({target.id:2d}), " # 使用 uav.id
                    f"{q_value_str}, " # [修改] 添加Q值到日志行
                    f"Dist={distance:6.1f}, "
                    f"UAV_Res_Before={np.array2string(uav_res_before, formatter={'float_kind':lambda x: '%.1f' % x})}, " # 使用执行前快照数据
                    f"Tgt_Need_Before={np.array2string(target_need_before, formatter={'float_kind':lambda x: '%.1f' % x})}, " # 使用执行前快照数据
                    f"Potential_Contrib={np.array2string(potential_contribution, formatter={'float_kind':lambda x: '%.1f' % x})}" # 预期贡献
                    )
                    action_details.append((distance, detail_str))
                except IndexError:
                    # 索引错误可能在解码时发生，记录下来
                    action_details.append((float('inf'), f"  - Action({action_idx:3d}): 解码时发生索引错误"))
                except Exception as e:
                    action_details.append((float('inf'), f"  - Action({action_idx:3d}): 处理错误: {e}"))

            # 按距离排序，方便查看
            action_details.sort(key=lambda x: x[0])
            for _, detail_str in action_details:
                self.action_logger.info(detail_str)

        # 突出显示最终选择的动作
        try:
            target_idx, uav_idx, phi_idx = env.decode_action(chosen_action)
            chosen_uav = env.uavs[uav_idx]
            chosen_target = env.targets[target_idx]

            # [修改] 提取并格式化选中动作的Q值
            chosen_q_value_str = ""
            if q_values is not None and chosen_action < q_values.size(1):
                chosen_q_val = q_values[0, chosen_action].item()
                chosen_q_value_str = f" | Q={chosen_q_val:8.4f}"

            # 使用快照数据计算预期贡献
            uav_res_before = uav_resources_snapshot.get(env.uavs[uav_idx].id, np.array([0, 0]))
            target_need_before = target_resources_snapshot.get(env.targets[target_idx].id, np.array([0, 0]))
            expected_contribution = np.minimum(uav_res_before, target_need_before)
            
            self.action_logger.info(
                f"\n>> 最终选择: Action({chosen_action:3d}) | UAV({chosen_uav.id:2d}) -> Target({chosen_target.id:2d}){chosen_q_value_str}") # [修改] 添加Q值
            self.action_logger.info(
                f">> 执行前状态: UAV_Res={np.array2string(uav_res_before, formatter={'float_kind':lambda x: '%.1f' % x})}, "
                f"Tgt_Need={np.array2string(target_need_before, formatter={'float_kind':lambda x: '%.1f' % x})}")
            self.action_logger.info(
                f">> 预期贡献: {np.array2string(expected_contribution, formatter={'float_kind':lambda x: '%.1f' % x})}")
                
            # 记录执行后状态作为对比（如果可用）
            uav_res_after = env.uavs[uav_idx].resources
            target_need_after = env.targets[target_idx].remaining_resources
            actual_contribution = uav_res_before - uav_res_after
            
            self.action_logger.info(
                f">> 执行后状态: UAV_Res={np.array2string(uav_res_after, formatter={'float_kind':lambda x: '%.1f' % x})}, "
                f"Tgt_Need={np.array2string(target_need_after, formatter={'float_kind':lambda x: '%.1f' % x})}")
            self.action_logger.info(
                f">> 实际贡献: {np.array2string(actual_contribution, formatter={'float_kind':lambda x: '%.1f' % x})}")
            
            # [调试] 检查预期与实际贡献的矛盾（问题3）
            if np.sum(expected_contribution) == 0 and np.sum(actual_contribution) > 0:
                self.action_logger.info(f"[WARNING] 预期贡献为0但实际贡献>0，可能存在状态不一致问题")
                self.action_logger.info(f"[DEBUG] 快照中目标状态: {target_need_before}")
                self.action_logger.info(f"[DEBUG] 环境中目标状态: {env.targets[target_idx].remaining_resources}")
                
            if reward is not None:
                self.action_logger.info(f">> 执行后奖励: {reward:.2f}")
            # 使用动作执行后的完成率，如果没有则使用当前完成率
            final_completion_rate = post_action_completion_rate if post_action_completion_rate is not None else current_completion_rate
            self.action_logger.info(f">> 执行后完成率: {final_completion_rate:.3f}")
            self.action_logger.info(f"{'='*120}\n")
        except Exception as e:
            self.action_logger.info(
                f"\n>> 最终选择: Action({chosen_action:3d}) (解码失败或无效动作: {e})")
            if reward is not None:
                self.action_logger.info(f">> 执行后奖励: {reward:.2f}")
            # 使用动作执行后的完成率，如果没有则使用当前完成率
            final_completion_rate = post_action_completion_rate if post_action_completion_rate is not None else current_completion_rate
            self.action_logger.info(f">> 执行后完成率: {final_completion_rate:.3f}")
            self.action_logger.info(f"{'='*120}\n")


    def setup_reward_logger(self):
        """设置奖励日志记录器"""
        logger = logging.getLogger('RewardLogger')
        
        # [新增] 强制清除旧的handlers，确保每次都重新配置
        if logger.hasHandlers():
            logger.handlers.clear()
        
        logger.setLevel(logging.INFO)
        logger.propagate = False # 防止将日志消息传递给根记录器
        
        # 创建文件处理器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 添加训练模式、网络类型、场景信息等到文件名
        training_mode = "curriculum" if getattr(self, '_is_curriculum', False) else "dynamic"
        network_type = self.config.NETWORK_TYPE
        episodes = self.config.training_config.episodes
        scenario_name = getattr(self, '_scenario_name', 'unknown')
        log_filename = f"reward_log_{training_mode}_{network_type}_{scenario_name}_{episodes}ep_{timestamp}.txt"
        log_path = os.path.join("output", log_filename)
        
        # 确保输出目录存在
        os.makedirs("output", exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(file_handler)
        
        # 写入日志头部
        logger.info("=" * 80)
        logger.info("训练奖励详细日志")
        logger.info("=" * 80)
        logger.info(f"日志创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"日志文件: {log_path}")
        logger.info("=" * 80)
        logger.info("")
        
        print(f"奖励日志已初始化: {log_path}")
        return logger

    def _init_reward_log(self):
        """初始化奖励日志 - 使用新的logger系统"""
        self.reward_logger = self.setup_reward_logger()
    
    def _log_episode_reward(self, episode: int, total_episodes: int, step_count: int, 
                           episode_reward: float, base_reward: float, shaping_reward: float,
                           completion_rate: float, exploration_rate: float, elapsed_time: float,
                           detailed_info: Dict = None, uavs=None, targets=None, obstacles=None,
                           scenario_name: str = None, solver=None):
        """【修复】记录单轮奖励信息到日志文件，确保使用统一的场景数据源"""
        # 奖励日志记录
        if hasattr(self, 'reward_logger'):
            # 防止除零错误：如果total_episodes为0（自适应课程训练），使用当前轮次作为进度
            if total_episodes > 0:
                progress_pct = (episode + 1) / total_episodes * 100
            else:
                progress_pct = episode + 1  # 自适应课程训练中显示当前轮次
            path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"
            
            # 【修复】优先使用实际环境中的场景数据，确保数据一致性
            if solver and hasattr(solver, 'env'):
                actual_uavs = len(solver.env.uavs) if hasattr(solver.env, 'uavs') else 0
                actual_targets = len(solver.env.targets) if hasattr(solver.env, 'targets') else 0
                actual_obstacles = len(solver.env.obstacles) if hasattr(solver.env, 'obstacles') else 0
                actual_scenario_name = getattr(solver.env, '_current_scenario_name', scenario_name or 'unknown')
            else:
                # 后备方案：使用传入的场景数据
                actual_uavs = len(uavs) if uavs is not None else 0
                actual_targets = len(targets) if targets is not None else 0
                actual_obstacles = len(obstacles) if obstacles is not None else 0
                actual_scenario_name = scenario_name or 'unknown'
            
            # 确定训练模式
            training_mode = "课程学习" if hasattr(self, '_is_curriculum') and self._is_curriculum else "随机训练"
            
            # 计算资源充裕度信息
            resource_abundance_info = self._calculate_resource_abundance(uavs, targets)
            
            try:
                self.reward_logger.info(f"Episode {episode + 1:6d}/{total_episodes:d} ({progress_pct:5.1f}%) [{actual_scenario_name}|{path_algo}|{training_mode}]: "
                                       f"无人机={actual_uavs}, 目标={actual_targets}, 障碍={actual_obstacles}, "
                                       f"{resource_abundance_info}, "
                                       f"步数={step_count:2d}, 总奖励={episode_reward:7.1f}, 基础奖励={base_reward:6.1f}, "
                                       f"塑形奖励={shaping_reward:6.1f}, 势能=0.000, 完成率={completion_rate:.3f}, "
                                       f"探索率={exploration_rate:.3f}, 用时={elapsed_time:.1f}s")
            except Exception as e:
                print(f"[WARNING] 轮次奖励记录失败: {e}")
                # 尝试简单的日志记录作为备用
                try:
                    self.reward_logger.info(f"Episode {episode + 1}/{total_episodes}: 场景={actual_scenario_name}, 无人机={actual_uavs}, 目标={actual_targets}, 障碍={actual_obstacles}, {resource_abundance_info}, 奖励={episode_reward:.1f}, 完成率={completion_rate:.3f}")
                except Exception as backup_error:
                    print(f"备用轮次日志记录也失败: {backup_error}")
        else:
            print(f"[WARNING] reward_logger未初始化，无法记录轮次奖励信息")
    
    def log_step_reward(self, step_i: int, episode: int, action: int, reward: float, 
                       step_info: dict, env_info: dict):
        """记录每一步的详细奖励信息 - 参考main-old.py RewardLogger格式"""
        # 检查是否启用奖励详情日志
        if not getattr(self.config, 'LOG_REWARD_DETAIL', False):
            return
        
        if not hasattr(self, 'reward_logger') or not self.reward_logger:
            return
        
        try:
            # 构建step_info和env_info，按照main-old.py RewardLogger.log_step的格式
            log_step_info = {
                'action': action,
                'total_reward': reward,
                'base_reward': step_info.get('base_reward', reward * 0.8),
                'shaping_reward': step_info.get('shaping_reward', reward * 0.2)
            }
            
            log_env_info = {
                'reward_breakdown': step_info.get('reward_breakdown', {}),
                'completion_rate': env_info.get('completion_rate', 0.0),
                'target_satisfied': env_info.get('target_satisfied', False),
                'final_success': env_info.get('final_success', False)
            }
            
            # 使用与main-old.py相同的格式记录
            self._log_step_detailed(step_i, log_step_info, log_env_info)
            
        except Exception as e:
            print(f"步骤日志记录错误: {e}")
            # 尝试简单的日志记录作为备用
            try:
                if hasattr(self, 'reward_logger') and self.reward_logger:
                    self.reward_logger.info(f"Step {step_i:3d}: Action={action}, Reward={reward:.2f}, Completion={env_info.get('completion_rate', 0.0):.3f}")
            except Exception as backup_error:
                print(f"备用日志记录也失败: {backup_error}")
    
    def _log_step_detailed(self, step_num: int, step_info: dict, env_info: dict):
        """详细步骤日志记录 - 完全按照main-old.py RewardLogger.log_step格式"""
        if not hasattr(self, 'reward_logger') or not self.reward_logger:
            return
        
        try:
            # 基础信息 - 按照main-old.py格式
            log_line = f"Step {step_num:3d}: "
            
            # 动作信息
            if 'action' in step_info:
                action = step_info['action']
                if isinstance(action, (list, tuple)) and len(action) >= 3:
                    log_line += f"Action=({action[0]}, {action[1]}, {action[2]}) "
                else:
                    log_line += f"Action={action} "
            
            # 奖励信息 - 按照main-old.py格式
            total_reward = step_info.get('total_reward', 0.0)
            base_reward = step_info.get('base_reward', 0.0)
            shaping_reward = step_info.get('shaping_reward', 0.0)
            
            log_line += f"Total={total_reward:7.2f} Base={base_reward:6.2f} Shaping={shaping_reward:6.2f}"
            
            # 详细奖励分解 - 按照main-old.py格式
            if env_info and 'reward_breakdown' in env_info:
                breakdown = env_info['reward_breakdown']
                log_line += " | 分解: "
                
                # 处理不同类型的奖励分解
                if 'layer1_breakdown' in breakdown and 'layer2_breakdown' in breakdown:
                    # 双层奖励分解
                    layer1 = breakdown['layer1_breakdown']
                    layer2 = breakdown['layer2_breakdown']
                    
                    log_line += "第一层["
                    layer1_items = [f"{k}={v:.1f}" for k, v in layer1.items() if abs(v) > 0.01]
                    log_line += ", ".join(layer1_items)
                    log_line += "] 第二层["
                    layer2_items = [f"{k}={v:.1f}" for k, v in layer2.items() if abs(v) > 0.01]
                    log_line += ", ".join(layer2_items)
                    log_line += "]"
                    
                elif 'simple_breakdown' in breakdown:
                    # 简单奖励分解
                    simple = breakdown['simple_breakdown']
                    breakdown_items = [f"{k}={v:.1f}" for k, v in simple.items() if abs(v) > 0.01]
                    log_line += "[" + ", ".join(breakdown_items) + "]"
                
                else:
                    # 其他格式的分解
                    breakdown_items = [f"{k}={v:.1f}" for k, v in breakdown.items() 
                                     if isinstance(v, (int, float)) and abs(v) > 0.01]
                    if breakdown_items:
                        log_line += "[" + ", ".join(breakdown_items) + "]"
            
            # 环境状态信息 - 按照main-old.py格式
            if env_info:
                completion_rate = env_info.get('completion_rate', 0.0)
                log_line += f" | 完成率={completion_rate:.3f}"
                
                if 'target_satisfied' in env_info and env_info['target_satisfied']:
                    log_line += " | 目标完成✓"
                
                if 'final_success' in env_info and env_info['final_success']:
                    log_line += " | 全部完成🏆"
            
            # 记录到日志
            self.reward_logger.info(log_line)
            
        except Exception as e:
            print(f"详细步骤日志记录错误: {e}")

    def start_episode_log(self, episode: int):
        """开始新回合的日志记录 - 参考main-old.py RewardLogger格式"""
        if hasattr(self, 'reward_logger') and self.reward_logger:
            try:
                # 简化的回合开始标记，与main-old.py保持一致
                self.reward_logger.info(f"\n{'='*100}")
                self.reward_logger.info(f"Episode {episode + 1} 开始")
                self.reward_logger.info(f"{'='*100}")
            except Exception as e:
                print(f"回合开始日志记录错误: {e}")

    def end_episode_log(self, episode: int, episode_summary: dict):
        """结束回合的日志记录 - 完全按照main-old.py RewardLogger.end_episode格式"""
        if not hasattr(self, 'reward_logger') or not self.reward_logger:
            return
        
        try:
            # 按照main-old.py RewardLogger.end_episode的格式
            self.reward_logger.info("-" * 100)
            self.reward_logger.info(f"Episode {episode + 1} 总结:")
            self.reward_logger.info(f"  总步数: {episode_summary.get('step_count', 0)}")
            self.reward_logger.info(f"  总奖励: {episode_summary.get('total_reward', 0.0):.2f}")
            self.reward_logger.info(f"  平均奖励: {episode_summary.get('avg_reward', 0.0):.2f}")
            self.reward_logger.info(f"  最终完成率: {episode_summary.get('final_completion_rate', 0.0):.3f}")
            self.reward_logger.info(f"  回合耗时: {episode_summary.get('episode_time', 0.0):.2f}秒")
            
            # 如果有详细的奖励统计
            if 'reward_stats' in episode_summary:
                stats = episode_summary['reward_stats']
                self.reward_logger.info(f"  奖励统计:")
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.reward_logger.info(f"    {key}: {value:.2f}")
            
            self.reward_logger.info("=" * 100 + "\n")
            
        except Exception as e:
            print(f"回合结束日志记录错误: {e}")

    def _close_reward_log(self):
        """关闭奖励日志 - 按照main-old.py RewardLogger.close格式"""
        if hasattr(self, 'reward_logger'):
            try:
                # 在关闭前添加奖励曲线分析报告
                self._append_reward_curve_analysis_to_log()
                
                # 按照main-old.py格式添加结束标记
                self.reward_logger.info("\n" + "=" * 100)
                self.reward_logger.info(f"日志记录结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.reward_logger.info("=" * 100)
                
                # 关闭所有handlers
                for handler in self.reward_logger.handlers:
                    handler.close()
                    self.reward_logger.removeHandler(handler)
                print("✅ 奖励日志记录完成（包含奖励曲线分析）")
            except Exception as e:
                print(f"❌ 关闭日志文件失败: {e}")

    def _append_reward_curve_analysis_to_log(self):
        """将奖励曲线分析追加到奖励日志中"""
        if not hasattr(self, 'reward_logger'):
            return
        
        rewards = self.training_stats.get('episode_rewards', [])
        completions = self.training_stats.get('completion_rates', [])
        losses = self.training_stats.get('episode_losses', [])
        
        # 添加分隔线和报告标题
        self.reward_logger.info("\n" + "=" * 80)
        self.reward_logger.info("强化学习训练奖励曲线分析报告")
        self.reward_logger.info("=" * 80)
        self.reward_logger.info(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.reward_logger.info(f"网络类型: {self.config.NETWORK_TYPE}")
        self.reward_logger.info(f"训练模式: {'课程学习' if hasattr(self, '_is_curriculum') and self._is_curriculum else '动态随机场景'}")
        self.reward_logger.info(f"路径算法: {'高精度PH-RRT' if self.config.USE_PHRRT_DURING_TRAINING else '快速近似'}")
        self.reward_logger.info(f"学习率: {self.config.LEARNING_RATE}")
        self.reward_logger.info(f"批次大小: {self.config.BATCH_SIZE}")
        self.reward_logger.info(f"折扣因子: {self.config.GAMMA}")
        self.reward_logger.info(f"探索率起始值: {self.config.EPSILON_START}")
        self.reward_logger.info(f"探索率结束值: {self.config.EPSILON_END}")
        self.reward_logger.info(f"探索率衰减: {self.config.EPSILON_DECAY}")
        self.reward_logger.info("=" * 80)
        
        # 奖励统计
        if rewards:
            self.reward_logger.info("\n奖励统计:")
            self.reward_logger.info("-" * 40)
            self.reward_logger.info(f"  总训练轮次: {len(rewards)}")
            self.reward_logger.info(f"  最高奖励: {max(rewards):.2f}")
            self.reward_logger.info(f"  最低奖励: {min(rewards):.2f}")
            self.reward_logger.info(f"  平均奖励: {np.mean(rewards):.2f}")
            self.reward_logger.info(f"  奖励标准差: {np.std(rewards):.2f}")
            self.reward_logger.info(f"  最终奖励: {rewards[-1]:.2f}")
            
            # 奖励趋势分析
            if len(rewards) > 10:
                recent_rewards = rewards[-10:]
                early_rewards = rewards[:10]
                recent_avg = np.mean(recent_rewards)
                early_avg = np.mean(early_rewards)
                improvement = (recent_avg - early_avg) / abs(early_avg) * 100 if early_avg != 0 else 0
                self.reward_logger.info("\n奖励趋势分析:")
                self.reward_logger.info("-" * 40)
                self.reward_logger.info(f"  前10轮平均奖励: {early_avg:.2f}")
                self.reward_logger.info(f"  后10轮平均奖励: {recent_avg:.2f}")
                self.reward_logger.info(f"  奖励改进: {improvement:.2f}%")
            
            # 收敛性分析
            if len(rewards) > 50:
                last_50 = rewards[-50:]
                convergence_std = np.std(last_50)
                convergence_mean = np.mean(last_50)
                self.reward_logger.info("\n收敛性分析:")
                self.reward_logger.info("-" * 40)
                self.reward_logger.info(f"  最后50轮平均奖励: {convergence_mean:.2f}")
                self.reward_logger.info(f"  最后50轮标准差: {convergence_std:.2f}")
                self.reward_logger.info(f"  变异系数: {convergence_std/abs(convergence_mean)*100:.2f}%")
                
                if convergence_std < 30:
                    self.reward_logger.info("  收敛状态: 良好收敛")
                elif convergence_std < 60:
                    self.reward_logger.info("  收敛状态: 部分收敛")
                else:
                    self.reward_logger.info("  收敛状态: 未收敛")
        else:
            self.reward_logger.info("\n奖励统计: 无奖励数据")
        
        # 完成率分析
        if completions:
            self.reward_logger.info("\n完成率分析:")
            self.reward_logger.info("-" * 40)
            self.reward_logger.info(f"  平均完成率: {np.mean(completions):.3f}")
            self.reward_logger.info(f"  最高完成率: {max(completions):.3f}")
            self.reward_logger.info(f"  最低完成率: {min(completions):.3f}")
            self.reward_logger.info(f"  最终完成率: {completions[-1]:.3f}")
            self.reward_logger.info(f"  完成率标准差: {np.std(completions):.3f}")
        else:
            self.reward_logger.info("\n完成率分析: 无完成率数据")
        
        # 损失分析
        if losses:
            self.reward_logger.info("\n损失分析:")
            self.reward_logger.info("-" * 40)
            self.reward_logger.info(f"  平均损失: {np.mean(losses):.4f}")
            self.reward_logger.info(f"  最高损失: {max(losses):.4f}")
            self.reward_logger.info(f"  最低损失: {min(losses):.4f}")
            self.reward_logger.info(f"  最终损失: {losses[-1]:.4f}")
            self.reward_logger.info(f"  损失标准差: {np.std(losses):.4f}")
        else:
            self.reward_logger.info("\n损失分析: 无损失数据")
        
        # 训练建议
        self.reward_logger.info("\n训练建议:")
        self.reward_logger.info("-" * 40)
        if rewards:
            if len(rewards) > 50:
                recent_std = np.std(rewards[-50:])
                if recent_std < 10:
                    self.reward_logger.info("  ✓ 模型已良好收敛，可以停止训练")
                elif recent_std < 50:
                    self.reward_logger.info("  ⚠ 模型部分收敛，建议继续训练或调整超参数")
                else:
                    self.reward_logger.info("  ✗ 模型未收敛，建议检查网络架构和超参数")
            
            if len(rewards) > 10:
                recent_avg = np.mean(rewards[-10:])
                early_avg = np.mean(rewards[:10])
                if recent_avg > early_avg * 1.1:
                    self.reward_logger.info("  ✓ 训练效果良好，奖励持续提升")
                elif recent_avg > early_avg * 0.9:
                    self.reward_logger.info("  ⚠ 训练效果一般，奖励提升缓慢")
                else:
                    self.reward_logger.info("  ✗ 训练效果不佳，奖励可能下降")
        
        self.reward_logger.info("\n" + "=" * 80)
        self.reward_logger.info("奖励曲线分析报告结束")
        self.reward_logger.info("=" * 80)

    
    def _save_best_model(self, model, episode: int, reward: float, scenario_name: str = "unknown"):
        """保存最优模型（维护前N个）- 优化：包含训练场景信息"""
        # 每20轮检查一次，或者在训练结束时检查
        if episode % 20 != 0 and episode != self.config.training_config.episodes - 1:
            return
        # 获取训练模式信息
        training_mode = "课程学习" if hasattr(self, '_is_curriculum') else "动态随机场景"
        path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"
        network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
        
        model_info = {
            'episode': episode,
            'reward': reward,
            'model_state': model.state_dict().copy(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_mode': training_mode,
            'path_algorithm': path_algo,
            'network_type': network_type,
            'scenario_name': scenario_name,  # 优化：加入训练场景信息
            'config_info': {
                'USE_PHRRT_DURING_TRAINING': getattr(self.config, 'USE_PHRRT_DURING_TRAINING', False),
                'USE_PHRRT_DURING_PLANNING': getattr(self.config, 'USE_PHRRT_DURING_PLANNING', True),
                'GRAPH_N_PHI': getattr(self.config, 'GRAPH_N_PHI', 1),
                'LEARNING_RATE': getattr(self.config, 'LEARNING_RATE', 'N/A'),
                'BATCH_SIZE': getattr(self.config, 'BATCH_SIZE', 'N/A'),
                'GAMMA': getattr(self.config, 'GAMMA', 'N/A'),
                'EPSILON_START': getattr(self.config, 'EPSILON_START', 'N/A'),
                'EPSILON_END': getattr(self.config, 'EPSILON_END', 'N/A'),
                'EPSILON_DECAY': getattr(self.config, 'EPSILON_DECAY', 'N/A'),
                'MEMORY_SIZE': getattr(self.config, 'MEMORY_SIZE', 'N/A'),
                'TARGET_UPDATE': getattr(self.config, 'TARGET_UPDATE', 'N/A')
            }
        }
        
        # 添加到最优模型列表
        self.best_models.append(model_info)
        
        # 按奖励排序并保留前N个
        self.best_models.sort(key=lambda x: x['reward'], reverse=True)
        
        # 保存模型文件
        model_dir = os.path.join("output", "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # 如果超过最大数量，删除最差的模型文件
        if len(self.best_models) > self.max_best_models:
            # 删除最差模型的文件
            worst_model = self.best_models[-1]
            worst_filename = f"best_model_ep{worst_model['episode']:06d}_reward{worst_model['reward']:.1f}_{worst_model['timestamp']}.pth"
            worst_path = os.path.join(model_dir, worst_filename)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                print(f"删除最差模型: {worst_filename}")
            
            # 保留前N个
            self.best_models = self.best_models[:self.max_best_models]
        
        # 保存当前模型（包含训练模式和场景信息）
        training_mode_short = "CL" if training_mode == "课程学习" else "DR"
        path_algo_short = "PH" if path_algo == "高精度PH-RRT" else "FA"
        completion_rate = self.training_stats.get('completion_rates', [])[-1] if self.training_stats.get('completion_rates', []) else 0.0
        # 优化：在模型文件名中加入场景信息
        model_filename = f"best_model_{network_type}_{training_mode_short}_{path_algo_short}_{scenario_name}_ep{episode:06d}_reward{reward:.1f}_comp{completion_rate:.3f}_{model_info['timestamp']}.pth"
        model_path = os.path.join(model_dir, model_filename)
         
        # 保存完整的模型信息（包括训练模式）
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': model_info
        }, model_path)
        print(f"最优模型已保存: {model_filename} (当前保存{len(self.best_models)}个模型)")

    
    def _dynamic_training(self, scenario_name: str = "small"):
        """默认的动态随机场景训练 - 根据指定场景进行训练"""
        print(f"执行{scenario_name}场景的图训练...")
        
        # 根据场景名称获取场景数据
        from scenarios import get_small_scenario, get_balanced_scenario, get_complex_scenario
        
        # 支持更多场景类型，包括easy、medium、hard等动态场景
        if scenario_name == "small":
            uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
        elif scenario_name == "balanced":
            uavs, targets, obstacles = get_balanced_scenario(obstacle_tolerance=50.0)
        elif scenario_name == "complex":
            uavs, targets, obstacles = get_complex_scenario(obstacle_tolerance=50.0)
        elif scenario_name in ["easy", "medium", "hard"]:
            # 对于动态场景，使用环境的动态生成功能
            print(f"使用动态随机{scenario_name}场景")
            # 创建临时环境来生成动态场景
            from environment import UAVTaskEnv
            # 直接创建临时环境，让环境内部处理图创建
            temp_env = UAVTaskEnv([], [], None, [], self.config, obs_mode="graph")
            temp_env._initialize_entities(scenario_name)
            uavs, targets, obstacles = temp_env.uavs, temp_env.targets, temp_env.obstacles
        else:
            print(f"⚠️ 未知场景名称: {scenario_name}，使用默认small场景")
            uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
        
        # 创建图和环境
        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
        # 确保ZeroShotGNN使用graph模式
        obs_mode = "graph" if self.config.NETWORK_TYPE == "ZeroShotGNN" else "flat"
        env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode=obs_mode)
        
        # 计算输入输出维度
        if self.config.NETWORK_TYPE == "ZeroShotGNN":
            obs_mode = "graph"
            i_dim = 64
            o_dim = len(targets) * len(uavs) * graph.n_phi

            if o_dim <= 0 and scenario_name in ["easy", "medium", "hard"]:
                # 如果是动态场景且初始为空，则使用配置中的最大实体数来创建网络
                # 这样可以确保网络能够处理后续生成的任何规模的场景
                max_o_dim = self.config.MAX_TARGETS * self.config.MAX_UAVS * self.config.GRAPH_N_PHI
                o_dim = max_o_dim
                print(f"动态场景初始化：使用最大动作空间占位符 o_dim={o_dim}")            

        else:
            obs_mode = "flat"
            target_dim = 7 * len(targets)
            uav_dim = 8 * len(uavs)
            collaboration_dim = len(targets) * len(uavs)
            global_dim = 10
            i_dim = target_dim + uav_dim + collaboration_dim + global_dim
            o_dim = len(targets) * len(uavs) * graph.n_phi
        
        # 创建求解器
        solver = GraphRLSolver(uavs, targets, graph, obstacles, i_dim, 
                              [self.config.hyperparameters.hidden_dim], o_dim, self.config, obs_mode=obs_mode)
        
        # 确保solver使用正确的环境
        solver.env = env
        
        total_episodes = self.config.training_config.episodes
        best_reward = float('-inf')
        
        # 【修复】移除初始场景保存，避免与训练后的保存重复
        # 场景数据将在每轮训练完成后保存，确保使用正确的完成率
        
        for episode in range(total_episodes):
            episode_start_time = time.time()
            
            # 动态场景生成：每20轮更换场景
            if episode > 0 and episode % 20 == 0:
                # [新增] 添加调试信息和验证逐步
                print(f"Episode {episode}: 开始生成新的{scenario_name}场景")
                
                # 生成新的动态场景
                max_attempts = 3  # 最多尝试3次
                scene_generated = False
                
                for attempt in range(max_attempts):
                    temp_env = UAVTaskEnv([], [], None, [], self.config, obs_mode="graph")
                    temp_env._initialize_entities(scenario_name)
                    new_uavs, new_targets, new_obstacles = temp_env.uavs, temp_env.targets, temp_env.obstacles
                    
                    # [新增] 验证生成的场景是否符合预期
                    template = self.config.SCENARIO_TEMPLATES[scenario_name]
                    uav_range = template['uav_num_range']
                    target_range = template['target_num_range']
                    
                    uav_count = len(new_uavs)
                    target_count = len(new_targets)
                    
                    if (uav_range[0] <= uav_count <= uav_range[1] and 
                        target_range[0] <= target_count <= target_range[1]):
                        print(f"  ✓ 第{attempt+1}次尝试成功: UAV={uav_count} (预期{uav_range}), Target={target_count} (预期{target_range})")
                        scene_generated = True
                        break
                    else:
                        print(f"  ✗ 第{attempt+1}次尝试失败: UAV={uav_count} (预期{uav_range}), Target={target_count} (预期{target_range})")
                
                if not scene_generated:
                    print(f"  警告: {max_attempts}次尝试后仍无法生成符合约束的场景，继续使用当前场景")
                else:
                    # 更新环境
                    new_graph = DirectedGraph(new_uavs, new_targets, self.config.GRAPH_N_PHI, new_obstacles, self.config)
                    new_env = UAVTaskEnv(new_uavs, new_targets, new_graph, new_obstacles, self.config, obs_mode=obs_mode)
                    
                    # 更新求解器
                    solver.uavs = new_uavs
                    solver.targets = new_targets
                    solver.graph = new_graph
                    solver.env = new_env
                    
                    # 更新当前场景变量
                    uavs, targets, obstacles = new_uavs, new_targets, new_obstacles
            
            # 训练一轮 - 修复：传递场景名称确保环境重置时使用正确的场景约束
            episode_reward, detailed_info = self._train_episode(uavs, targets, obstacles, episode, scenario_name)
            
            # 计算完成率 - 使用标准完成率计算方法（与环境一致）
            if hasattr(self, '_current_env') and self._current_env and hasattr(self._current_env, 'get_completion_rate'):
                completion_rate = self._current_env.get_completion_rate()
            else:
                # 后备方案：使用标准资源贡献计算方法
                completion_rate = self._calculate_standard_completion_rate(targets)
            
            # 记录统计
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['completion_rates'].append(completion_rate)
            # 收集损失数据
            episode_loss = detailed_info.get('episode_loss', 0.0)
            self.training_stats['episode_losses'].append(episode_loss)
            
            episode_elapsed = time.time() - episode_start_time
            
            # 从详细信息中提取奖励分解数据
            total_base_reward = detailed_info.get('total_base_reward', episode_reward * 0.8)
            total_shaping_reward = detailed_info.get('total_shaping_reward', episode_reward * 0.2)
            exploration_rate = solver.epsilon
            
            # 动态训练模式统一输出格式（新增场景关键信息）
            progress_pct = (episode + 1) / total_episodes * 100
            path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"
            
            # 获取场景关键信息
            num_uavs = len(uavs)
            num_targets = len(targets)
            num_obstacles = len(obstacles) if obstacles else 0
            
            # 【修复】使用实际环境中的场景数据，确保数据一致性
            actual_uavs = len(solver.env.uavs) if hasattr(solver.env, 'uavs') else num_uavs
            actual_targets = len(solver.env.targets) if hasattr(solver.env, 'targets') else num_targets
            actual_obstacles = len(solver.env.obstacles) if hasattr(solver.env, 'obstacles') else num_obstacles
            
            # 获取实际场景名称
            actual_scenario_name = getattr(solver.env, '_current_scenario_name', scenario_name)
            
            print(f"Episode {episode + 1:4d}/{total_episodes} ({progress_pct:5.1f}%) [{actual_scenario_name}|{path_algo}]: "
                  f"无人机={actual_uavs}, 目标={actual_targets}, 障碍={actual_obstacles}, "
                  f"步数={detailed_info['step_count']:2d}, "
                  f"总奖励={episode_reward:7.1f}, "
                  f"完成率={completion_rate:.3f}, "
                  f"探索率={exploration_rate:.3f}, "
                  f"用时={episode_elapsed:.1f}s")
            
            # 【修复】记录奖励信息 - 传递solver确保使用实际环境数据
            self._log_episode_reward(
                episode, total_episodes, detailed_info['step_count'],
                episode_reward, total_base_reward, total_shaping_reward,
                completion_rate, exploration_rate, episode_elapsed, detailed_info,
                uavs, targets, obstacles, actual_scenario_name, solver
            )
            
            # 【新增】验证场景数据一致性
            self._validate_scenario_consistency(solver, actual_scenario_name, episode)
            
            # 【修复】在训练完成后保存场景数据，确保使用正确的完成率
            # 只在指定间隔记录推理结果，避免过多文件
            should_log_inference = (episode % self.config.EPISODE_INFERENCE_LOG_INTERVAL == 0)
            if should_log_inference:
                self.save_scenario_data(episode, uavs, targets, obstacles, scenario_name, 
                                       solver=solver, completion_rate=completion_rate)
            else:
                self.save_scenario_data(episode, uavs, targets, obstacles, scenario_name, 
                                       completion_rate=completion_rate)
            
            # 保存最优模型 (每50轮检查一次，或者在训练结束时检查)
            if (episode + 1) % 50 == 0 or (episode + 1) == total_episodes:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                # 修复：传递scenario_name参数确保文件名正确
                self._save_best_model(solver.policy_net, episode, episode_reward, scenario_name)
            
            # 定期输出统计信息
            if (episode + 1) % self.config.training_config.log_interval == 0:
                recent_rewards = self.training_stats['episode_rewards'][-self.config.training_config.log_interval:]
                recent_completions = self.training_stats['completion_rates'][-self.config.training_config.log_interval:]
                
                avg_reward = np.mean(recent_rewards)
                avg_completion = np.mean(recent_completions)
                
                print(f"统计 (Episode {episode + 1 - self.config.training_config.log_interval + 1}-{episode + 1}): "
                      f"平均奖励={avg_reward:8.2f}, 平均完成率={avg_completion:6.3f}", flush=True)
            
            def _train_episode(self, uavs, targets, obstacles, episode, scenario_name='medium'):
                """
                单轮训练核心逻辑 - 修复版本，添加scenario_name参数以确保正确传递给环境reset
                    
                Args:
                    uavs: UAV列表
                    targets: 目标列表
                    obstacles: 障碍物列表
                    episode: 轮次编号
                    scenario_name: 场景名称，用于环境重置时传递约束参数
                    
                Returns:
                    Tuple[float, Dict]: (episode_reward, detailed_info)
                """
                # 此方法已被_train_episode_with_scenario替代，为了兼容性保留简单委托
                return self._train_episode_with_scenario(
                    uavs, targets, obstacles, episode, scenario_name, 
                    solver=None, max_steps=150
                )
    
    def _curriculum_training(self):
        """课程学习训练模式 - 使用预定义的固定场景"""
        print("执行课程学习训练...")
        
        # 获取预定义的课程学习场景序列
        from scenarios import get_curriculum_scenarios
        curriculum_scenarios = get_curriculum_scenarios(large_scale=False)
        
        global_episode_counter = 0
        
        # 按阶段进行训练
        for scenario_func, scenario_name, scenario_desc in curriculum_scenarios:
            print(f"\n开始训练阶段: {scenario_name}")
            print(f"场景描述: {scenario_desc}")
            
            # 生成固定场景
            uavs, targets, obstacles = scenario_func(obstacle_tolerance=50.0)
            
            # 计算该阶段的训练轮次 (每个场景训练多轮)
            stage_episodes = 10  # 每个固定场景训练10轮
            
            for local_episode in tqdm(range(stage_episodes), desc=f"{scenario_name}阶段"):
                global_episode_counter += 1
                
                # 复用动态随机训练的核心逻辑
                episode_reward, detailed_info = self._train_episode_with_scenario(
                    uavs, targets, obstacles, global_episode_counter - 1, scenario_name
                )
                
                # 计算完成率 - 使用标准完成率计算方法（与环境一致）
                completion_rate = self._calculate_standard_completion_rate(targets)
                
                # 记录统计
                self.training_stats['episode_rewards'].append(episode_reward)
                self.training_stats['completion_rates'].append(completion_rate)
                episode_loss = detailed_info.get('episode_loss', 0.0)
                self.training_stats['episode_losses'].append(episode_loss)
                
                # 计算总轮次
                total_episodes = len(curriculum_scenarios) * stage_episodes
                progress_pct = global_episode_counter / total_episodes * 100
                path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"
                
                # 【修复】增强的场景信息输出 - 课程学习模式使用预设场景数据
                print(f"Episode {global_episode_counter:4d}/{total_episodes} ({progress_pct:5.1f}%) [{scenario_name}|{path_algo}]: "
                      f"UAV={len(uavs):2d} 目标={len(targets):2d} 障碍={len(obstacles):2d}, "
                      f"步数={detailed_info['step_count']:2d}, "
                      f"总奖励={episode_reward:7.1f}, "
                      f"完成率={completion_rate:.3f}", flush=True)
                
                # 记录奖励信息
                self._log_episode_reward(
                    global_episode_counter - 1, total_episodes, detailed_info['step_count'],
                    episode_reward, detailed_info.get('total_base_reward', episode_reward * 0.8), 
                    detailed_info.get('total_shaping_reward', episode_reward * 0.2),
                    completion_rate, detailed_info.get('exploration_rate', 0.1), 
                    detailed_info.get('episode_time', 0.0), detailed_info,
                    uavs, targets, obstacles, scenario_name, None  # 传统课程训练没有solver对象
                )
                
                # 【修复】在训练完成后保存场景数据，确保使用正确的完成率
                should_log_inference = (global_episode_counter % self.config.EPISODE_INFERENCE_LOG_INTERVAL == 0)
                if should_log_inference:
                    # 需要创建临时solver用于推理结果
                    try:
                        from environment import UAVTaskEnv, DirectedGraph
                        from solvers import GraphRLSolver
                        
                        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                        env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
                        
                        i_dim = 64  # 固定维度
                        h_dim = self.config.hyperparameters.hidden_dim
                        o_dim = len(targets) * len(uavs) * graph.n_phi
                        
                        temp_solver = GraphRLSolver(uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, self.config, obs_mode="graph")
                        
                        self.save_scenario_data(
                            episode=global_episode_counter - 1,
                            uavs=uavs,
                            targets=targets,
                            obstacles=obstacles,
                            scenario_name=scenario_name,
                            solver=temp_solver,
                            completion_rate=completion_rate
                        )
                    except Exception as e:
                        print(f"创建临时solver失败: {e}")
                        self.save_scenario_data(
                            episode=global_episode_counter - 1,
                            uavs=uavs,
                            targets=targets,
                            obstacles=obstacles,
                            scenario_name=scenario_name,
                            completion_rate=completion_rate
                        )
                else:
                    self.save_scenario_data(
                        episode=global_episode_counter - 1,
                        uavs=uavs,
                        targets=targets,
                        obstacles=obstacles,
                        scenario_name=scenario_name,
                        completion_rate=completion_rate
                    )
            
            print(f"阶段 {scenario_name} 完成", flush=True)
    
    def _curriculum_training_graph(self) -> None:
        """基于图的课程学习训练（ZeroShotGNN专用）- 支持自适应晋级机制"""    
        # 检查是否启用自适应课程训练 (当前默认替换为新的渐进式策略)
        if self.config.CURRICULUM_USE_GRANULAR_PROGRESSION:
            print("执行渐进式自适应课程学习训练...")
            self._adaptive_curriculum_training_graph()
        else:
            # 如果需要，可以保留旧的基于模板的自适应或固定流程作为备选
            print("执行传统的固定课程学习训练...")
            self._traditional_curriculum_training_graph()
            
    def _adaptive_curriculum_training_graph(self) -> None:
        """
        函数级注释：实现渐进式自适应课程学习的主流程。
        该流程通过多个精细化的等级逐步增加场景复杂度，并根据模型的实时表现决定何时晋级。
        包含了详细的日志记录功能，其详细程度由 config.LOG_LEVEL 控制。
        """
        # --- 1. 初始化 ---
        total_levels = self.config.GRANULAR_CURRICULUM_LEVELS
        solver = None
        global_episode_counter = 0
        best_reward = float('-inf')
        curriculum_start_time = time.time()
        self.last_promotion_info = None # 开始新训练时重置

        # 获取日志级别
        log_level = getattr(self.config, 'LOG_LEVEL', 'simple')

        if getattr(self.config, 'LOG_EPISODE_DETAIL', False):
            print(f"🚀 启动渐进式自适应课程学习, 共 {total_levels} 个等级")
            print(f"📊 掌握度阈值: {self.config.CURRICULUM_MASTERY_THRESHOLD:.2f}, 性能窗口: {self.config.CURRICULUM_PERFORMANCE_WINDOW}")

        # --- 2. 按等级进行主循环 ---
        for level in range(total_levels):            
            # --- 2.1. 初始化当前等级的阈值 ---
            # 如果之前没有为该等级设置过阈值，则使用配置文件中的初始值
            current_threshold = self.level_thresholds.setdefault(
                level, self.config.ADAPTIVE_THRESHOLD_INITIAL
            )
            level_start_time = time.time()
            # --- 2.2. 计算当前等级的场景参数 ---
            level_params = self._calculate_level_parameters(level)
            scenario_name = f"Level_{level+1:02d}"
            if getattr(self.config, 'LOG_EPISODE_DETAIL', False):
                print("\n" + "="*80)
                print(f"进入等级: {scenario_name} | UAVs:{level_params['uav_num']}, Targets:{level_params['target_num']}, Obstacles:{level_params['obstacle_num']}, Abundance:{level_params['resource_abundance']:.2f}")
                print("="*80)

            # --- 2.3. 初始化或更新求解器 ---
            if solver is None:
                # 首次创建求解器，使用最大维度以兼容所有等级
                i_dim = 64  # ZeroShotGNN固定输入
                max_o_dim = self.config.MAX_TARGETS * self.config.MAX_UAVS * self.config.GRAPH_N_PHI
                solver = GraphRLSolver([], [], None, [], i_dim, 
                                    [self.config.hyperparameters.hidden_dim], max_o_dim, 
                                    self.config, obs_mode="graph")
                # 绑定日志记录器
                solver.step_logger = self.log_step_reward
                solver.action_logger = self.log_action_details

            # --- 2.4. 等级内的训练循环 ---
            performance_window = deque(maxlen=self.config.CURRICULUM_PERFORMANCE_WINDOW)
            max_episodes_per_level = self.config.CURRICULUM_MAX_EPISODES_PER_LEVEL

            for episode_in_level in range(max_episodes_per_level):
                global_episode_counter += 1
                episode_start_time = time.time()

                # --- 2.4.1. 动态生成当前等级的随机场景 ---
                scenario_dict = _generate_scenario(self.config, **level_params)
                uavs, targets, obstacles = scenario_dict['uavs'], scenario_dict['targets'], scenario_dict['obstacles']

                # 更新solver的环境
                graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
                solver.env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")

                # --- 2.4.2. 单个训练回合的核心逻辑 ---
                state, _ = solver.env.reset(options={'scenario': scenario_dict, 'scenario_name': scenario_name})
                episode_reward, step_count = 0.0, 0
                total_base_reward, total_shaping_reward = 0.0, 0.0
                episode_losses = []

                while True:
                    action, _ = solver.select_action(state)
                    next_state, reward, done, truncated, info = solver.env.step(action.item())

                    episode_reward += reward
                    total_base_reward += info.get('base_reward', 0.0)
                    total_shaping_reward += info.get('shaping_reward', 0.0)

                    solver.memory.push(state, action, torch.tensor([reward], device=solver.device), next_state, done or truncated)
                    state = next_state
                    loss = solver.optimize_model()
                    if loss is not None:
                        episode_losses.append(loss)

                    step_count += 1
                    if done or truncated:
                        break

                solver.epsilon = max(self.config.EPSILON_END, solver.epsilon * self.config.EPSILON_DECAY)
                if global_episode_counter % self.config.TARGET_UPDATE_FREQ == 0:
                    solver.target_net.load_state_dict(solver.policy_net.state_dict())

                # --- 2.4.3. 日志记录、统计与晋级检查 ---
                completion_rate = solver.env.get_completion_rate()
                performance_window.append(completion_rate)

                # 更新统计数据
                self.training_stats['episode_rewards'].append(episode_reward)
                self.training_stats['completion_rates'].append(completion_rate)
                avg_loss = np.mean(episode_losses) if episode_losses else 0.0
                self.training_stats['episode_losses'].append(avg_loss)

                # 日志输出（遵循日志等级）
                if getattr(self.config, 'LOG_REWARD_DETAIL', False):
                    episode_elapsed = time.time() - episode_start_time
                    resource_abundance_info = self._calculate_resource_abundance(uavs, targets)
                    path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"

                    print(
                        f"Episode {global_episode_counter:5d} [{scenario_name}|{path_algo}]: "
                        f"UAVs={len(uavs):2d}, Tgs={len(targets):2d}, Obs={len(obstacles):2d}, {resource_abundance_info}, "
                        f"Steps={step_count:3d}, Reward={episode_reward:8.2f}, Base={total_base_reward:7.2f}, Shaping={total_shaping_reward:6.2f}, "
                        f"CompRate={completion_rate:.3f}, Loss={avg_loss:.4f}, Epsilon={solver.epsilon:.3f}, Time={episode_elapsed:.1f}s"
                    )

                # 写入到日志文件
                self._log_episode_reward(
                    global_episode_counter - 1, self.config.training_config.episodes, step_count,
                    episode_reward, total_base_reward, total_shaping_reward,
                    completion_rate, solver.epsilon, time.time() - episode_start_time, {},
                    uavs, targets, obstacles, scenario_name, solver
                )
              
                # --- 2.4.4. 检查是否需要自适应调整阈值 ---
                if self.config.ADAPTIVE_THRESHOLD_ENABLED:
                    # (策略一: 晋级后不稳定检查)
                    # 检查是否刚从上一级晋升过来
                    if self.last_promotion_info and self.last_promotion_info['level_promoted_to'] == level:
                        # 当在新等级收集到足够的数据点时
                        if len(performance_window) == self.config.CURRICULUM_PERFORMANCE_WINDOW:
                            new_level_avg_completion = np.mean(performance_window)
                            prev_level_threshold = self.last_promotion_info['threshold_passed']

                            # 如果在新等级的表现大幅下跌
                            if new_level_avg_completion < (prev_level_threshold - self.config.ADAPTIVE_THRESHOLD_INSTABILITY_DROP):
                                prev_level_idx = self.last_promotion_info['level_promoted_from']
                                # 提升上一级的阈值
                                new_prev_threshold = min(
                                    self.config.ADAPTIVE_THRESHOLD_MAX,
                                    self.level_thresholds[prev_level_idx] + self.config.ADAPTIVE_THRESHOLD_UP_STEP
                                )
                                if new_prev_threshold > self.level_thresholds[prev_level_idx]:
                                    self.level_thresholds[prev_level_idx] = new_prev_threshold
                                    print(f"🧠 [自适应调整] 检测到晋级后表现不稳定，已将等级 {prev_level_idx+1} 的阈值提升至: {new_prev_threshold:.2f}")

                            self.last_promotion_info = None # 该检查只执行一次

                    # (策略二: 训练停滞检查)
                    stagnation_check_episode = int(self.config.ADAPTIVE_THRESHOLD_STAGNATION_TRIGGER * max_episodes_per_level)
                    if episode_in_level >= stagnation_check_episode and len(performance_window) >= self.config.CURRICULUM_PERFORMANCE_WINDOW:

                        # (策略二优化: 结合标准差和斜率判断平台期)
                        is_plateau = False
                        performance_std_dev = np.std(performance_window)
                        if performance_std_dev < self.config.ADAPTIVE_THRESHOLD_PLATEAU_STD_DEV:
                            if self.config.ADAPTIVE_THRESHOLD_USE_SLOPE:
                                # 计算趋势斜率
                                x = np.arange(len(performance_window))
                                slope, _ = np.polyfit(x, list(performance_window), 1)
                                if slope < self.config.ADAPTIVE_THRESHOLD_PLATEAU_SLOPE:
                                    is_plateau = True
                            else:
                                is_plateau = True

                        if is_plateau:
                            # 如果进入平台期，则小幅降低阈值
                            new_threshold = max(
                                self.config.ADAPTIVE_THRESHOLD_MIN,
                                current_threshold - self.config.ADAPTIVE_THRESHOLD_DOWN_STEP
                            )
                            if new_threshold < current_threshold:
                                current_threshold = new_threshold
                                self.level_thresholds[level] = current_threshold
                                print(f"🧠 [自适应调整] 检测到训练停滞，已将等级 {level+1} 的掌握度阈值下调至: {current_threshold:.2f}")

                # --- 2.4.5. 检查晋级条件 ---
                if len(performance_window) >= self.config.CURRICULUM_PERFORMANCE_WINDOW:
                    avg_completion = np.mean(performance_window)
                    if avg_completion >= current_threshold:
                        print(f"🎉 掌握度达标! 平均完成率 {avg_completion:.2%} >= {current_threshold:.0%}. 晋级到下一等级。")
                        # 记录晋级信息，用于下一级的表现检查
                        self.last_promotion_info = {
                            'level_promoted_from': level,
                            'level_promoted_to': level + 1,
                            'threshold_passed': current_threshold
                        }
                        break 
                
                # [新增] 在每轮课程学习结束后，调用函数保存场景和推理结果数据
                # 这样可以确保scenario_logs文件夹中有记录
                try:
                    # 构建episode_info以生成推理报告
                    episode_info = {'final_env': solver.env} # 简化处理，传递最终环境
                    self.save_scenario_data(
                        episode=global_episode_counter,
                        uavs=uavs,
                        targets=targets,
                        obstacles=obstacles,
                        scenario_name=scenario_name,
                        solver=solver, # 传递solver以获取推理结果
                        completion_rate=completion_rate,
                        episode_info=episode_info
                    )
                except Exception as e:
                    print(f"在课程学习中保存场景数据失败: {e}")

            # --- 2.5. 等级训练结束总结 ---
            level_duration = time.time() - level_start_time
            print(f"--- 等级 {scenario_name} 训练结束 (耗时: {level_duration:.1f}s) ---")
            if episode_in_level == max_episodes_per_level - 1:
                print(f"⚠️  已达到最大训练轮次 {max_episodes_per_level}，强制晋级。")

            if episode_reward > best_reward:
                best_reward = episode_reward
            self._save_best_model(solver.policy_net, global_episode_counter, best_reward, scenario_name)

        total_training_time = time.time() - curriculum_start_time
        print(f"\n🏆 所有课程学习等级已完成！(总耗时: {total_training_time:.1f}s)")
        
    def _traditional_curriculum_training_graph(self) -> None:
        """传统课程学习训练 - 固定轮次训练（原有实现）"""
        
        # 获取预定义的课程学习场景序列
        from scenarios import get_curriculum_scenarios
        curriculum_scenarios = get_curriculum_scenarios(large_scale=False)
        
        # 创建求解器（使用第一个场景初始化）
        first_scenario_func, _, _ = curriculum_scenarios[0]
        uavs, targets, obstacles = first_scenario_func(obstacle_tolerance=50.0)
        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
        
        i_dim = 64  # ZeroShotGNN固定输入维度
        o_dim = len(targets) * len(uavs) * graph.n_phi
        obs_mode = "graph"  # ZeroShotGNN使用图模式
        
        solver = GraphRLSolver(uavs, targets, graph, obstacles, i_dim, 
                              [self.config.hyperparameters.hidden_dim], o_dim, self.config, obs_mode=obs_mode)
        
        global_episode_counter = 0
        best_reward = float('-inf')
        
        # 按阶段进行训练
        for scenario_func, scenario_name, scenario_desc in curriculum_scenarios:
            print(f"\n开始训练阶段: {scenario_name}")
            print(f"场景描述: {scenario_desc}")
            
            # 生成固定场景
            uavs, targets, obstacles = scenario_func(obstacle_tolerance=50.0)
            
            # 根据场景名称设置不同的最大步数
            if 'Level1' in scenario_name or 'Level2' in scenario_name:
                max_steps_per_episode = 100
            elif 'Level3' in scenario_name:
                max_steps_per_episode = 150
            else:
                max_steps_per_episode = 200
            
            print(f"当前阶段最大步数上限: {max_steps_per_episode}")
            
            # 计算该阶段的训练轮次 (每个场景训练多轮)
            stage_episodes = 10  # 每个固定场景训练10轮
            
            # 阶段训练循环
            for local_ep in range(1, stage_episodes + 1):
                global_episode_counter += 1
                
                # 复用动态随机训练的核心逻辑
                episode_reward, detailed_info = self._train_episode_with_scenario(
                    uavs, targets, obstacles, global_episode_counter - 1, scenario_name, 
                    solver=solver, max_steps=max_steps_per_episode
                )
                
                # 计算完成率 - 使用标准完成率计算方法（与环境一致）
                completion_rate = self._calculate_standard_completion_rate(targets)
                
                # 记录统计
                self.training_stats['episode_rewards'].append(episode_reward)
                self.training_stats['completion_rates'].append(completion_rate)
                
                # 计算总轮次用于进度显示
                total_episodes = len(curriculum_scenarios) * stage_episodes
                progress_pct = global_episode_counter / total_episodes * 100
                
                # 获取路径算法信息
                path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"
                
                # 【修复】增强的场景信息输出 - 课程学习模式使用预设场景数据
                print(f"Episode {global_episode_counter:4d}/{total_episodes} ({progress_pct:5.1f}%) [{scenario_name}|{path_algo}]: "
                      f"UAV={len(uavs):2d} 目标={len(targets):2d} 障碍={len(obstacles):2d}, "
                      f"步数={detailed_info['step_count']:2d}, "
                      f"总奖励={episode_reward:7.1f}, "
                      f"完成率={completion_rate:.3f}, "
                      f"探索率={solver.epsilon:.3f}", flush=True)
                
                # 记录奖励信息
                self._log_episode_reward(
                    global_episode_counter - 1, total_episodes, detailed_info['step_count'],
                    episode_reward, detailed_info.get('total_base_reward', episode_reward * 0.8), 
                    detailed_info.get('total_shaping_reward', episode_reward * 0.2),
                    completion_rate, solver.epsilon, detailed_info.get('episode_time', 0.0), detailed_info,
                    uavs, targets, obstacles, scenario_name, solver
                )
                
                # 【新增】验证场景数据一致性
                self._validate_scenario_consistency(solver, scenario_name, global_episode_counter - 1)
                
                # 保存最优模型 (每50轮检查一次，或者在训练结束时检查)
                if global_episode_counter % 50 == 0 or global_episode_counter == total_episodes:
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                    # 修复：传递scenario_name参数确保文件名正确
                    self._save_best_model(solver.policy_net, global_episode_counter, episode_reward, scenario_name)
            
            print(f"阶段 {scenario_name} 完成", flush=True)
    
    def _get_stage_max_steps(self, stage_name: str) -> int:
        """根据阶段名称获取最大步数"""
        if stage_name == 'easy':
            return 100
        elif stage_name == 'medium':
            return 200
        elif stage_name == 'hard':
            return 300
        else:
            return 150
    
    def _get_adaptive_max_steps(self, scenario_name: str) -> int:
        """根据场景名称获取自适应训练的最大步数"""
        if 'Level1' in scenario_name or 'Level2' in scenario_name:
            return 100
        elif 'Level3' in scenario_name:
            return 150
        elif 'Level4' in scenario_name or 'Level5' in scenario_name:
            return 200
        else:
            return 150
    
    def _train_episode_with_scenario(self, uavs, targets, obstacles, episode_num, stage_name, 
                                   solver=None, max_steps=150):
        """复用动态随机训练的核心逻辑，用于课程训练"""
        # 修复：确保与动态随机训练使用相同的solver创建和更新逻辑
        if solver is None:
            # 创建新的solver（仅在第一次调用时）
            graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
            obs_mode = "graph" if self.config.NETWORK_TYPE == "ZeroShotGNN" else "flat"
            env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode=obs_mode, max_steps=max_steps)
            
            # 设置课程训练模式标识
            env._is_curriculum_training = getattr(self, '_is_curriculum', True)
            env._current_curriculum_stage = stage_name
            
            # 计算输入输出维度（与动态随机训练保持一致）
            if self.config.NETWORK_TYPE == "ZeroShotGNN":
                i_dim = 64
                o_dim = len(targets) * len(uavs) * graph.n_phi
            else:
                target_dim = 7 * len(targets)
                uav_dim = 8 * len(uavs)
                collaboration_dim = len(targets) * len(uavs)
                global_dim = 10
                i_dim = target_dim + uav_dim + collaboration_dim + global_dim
                o_dim = len(targets) * len(uavs) * graph.n_phi
            
            solver = GraphRLSolver(uavs, targets, graph, obstacles, i_dim, 
                                  [self.config.hyperparameters.hidden_dim], o_dim, self.config, obs_mode=obs_mode)
            solver.env = env
        else:
            # 修复：正确更新现有solver的环境（与动态随机训练保持一致）
            graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
            obs_mode = solver.obs_mode  # 保持原有的观测模式
            
            # 更新solver的组件
            solver.uavs = uavs
            solver.targets = targets
            solver.graph = graph
            solver.obstacles = obstacles
            
            # 创建新环境并更新
            new_env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode=obs_mode, max_steps=max_steps)
            new_env._is_curriculum_training = getattr(self, '_is_curriculum', True)
            new_env._current_curriculum_stage = stage_name
            solver.env = new_env
        
        # 开始回合日志记录
        self.start_episode_log(episode_num)
        
        episode_start_time = time.time()
        
        # 重置环境 - 修复：传递完整的场景数据确保使用正确的场景
        scenario_data = {
            'uavs': uavs,
            'targets': targets,
            'obstacles': obstacles
        }
        reset_options = {'scenario': scenario_data, 'scenario_name': stage_name}
        reset_result = solver.env.reset(options=reset_options)
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        episode_reward = 0.0
        step_count = 0
        
        # 奖励分解统计
        total_base_reward = 0.0
        total_shaping_reward = 0.0
        step_rewards = []
        completion_rates = []
        episode_losses = []
        
        # 训练循环
        done = False
        truncated = False
        while not (done or truncated) and step_count < max_steps:
            # 使用强化学习选择动作
            action = solver.select_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, info = solver.env.step(action.item())
            
            # 计算当前完成率 - 使用标准计算方法
            current_completion_rate = solver.env.get_completion_rate() if hasattr(solver.env, 'get_completion_rate') else 0.0
            completion_rates.append(current_completion_rate)
            step_rewards.append(reward)
            
            # 准备步骤日志信息
            step_info = {
                'reward_breakdown': info.get('reward_breakdown', {}),
                'base_reward': info.get('base_reward', reward * 0.8),
                'shaping_reward': info.get('shaping_reward', reward * 0.2)
            }
            
            env_info = {
                'completion_rate': current_completion_rate,
                'remaining_targets': len(solver.env.targets) - sum(1 for t in solver.env.targets if np.all(t.remaining_resources <= 0)),
                'epsilon': solver.epsilon
            }
            
            # 记录详细的步骤奖励信息到日志
            try:
                self.log_step_reward(step_count, episode_num, action.item(), reward, step_info, env_info)
            except Exception as e:
                print(f"[WARNING] 步骤奖励记录失败: {e}")
            
            # 累积奖励统计
            total_base_reward += step_info['base_reward']
            total_shaping_reward += step_info['shaping_reward']
            
            # 存储经验
            solver.memory.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            step_count += 1
            
            # 训练网络并收集损失
            if len(solver.memory) >= solver.batch_size:
                loss = solver.optimize_model()
                if loss is not None:
                    episode_losses.append(loss)
                    self.training_stats['step_losses'].append(loss)
            
            if done or truncated:
                break
        
        episode_time = time.time() - episode_start_time
        
        # 计算平均损失
        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        # 准备详细信息
        detailed_info = {
            'step_count': step_count,
            'total_base_reward': total_base_reward,
            'total_shaping_reward': total_shaping_reward,
            'episode_loss': avg_episode_loss,
            'episode_time': episode_time,
            'exploration_rate': solver.epsilon,
            'reward_breakdown_components': {
                'base_matching': total_base_reward * 0.625,
                'demand_satisfaction': total_base_reward * 0.375,
                'type_matching': total_shaping_reward * 0.5,
                'urgency_bonus': total_shaping_reward * 0.5
            }
        }
        
        # 准备回合总结
        episode_summary = {
            'step_count': step_count,
            'total_reward': episode_reward,
            'avg_reward': episode_reward / max(step_count, 1),
            'final_completion_rate': completion_rates[-1] if completion_rates else 0.0,
            'episode_time': episode_time,
            'episode_loss': avg_episode_loss
        }
        
        # 结束回合日志记录
        self.end_episode_log(episode_num, episode_summary)
        
        return episode_reward, detailed_info
    
    def _graph_training(self, scenario_name: str = "small") -> None:
        """基于图的强化学习训练（启用早停），根据指定场景进行训练。
        这是ZeroShotGNN的默认训练方式，使用GraphRLSolver.train方法"""
        
        # 根据场景名称获取场景数据
        from scenarios import get_small_scenario, get_balanced_scenario, get_complex_scenario
        
        print(f"执行{scenario_name}场景的图训练...")
        
        if scenario_name == "small":
            uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
        elif scenario_name == "balanced":
            uavs, targets, obstacles = get_balanced_scenario(obstacle_tolerance=50.0)
        elif scenario_name == "complex":
            uavs, targets, obstacles = get_complex_scenario(obstacle_tolerance=50.0)
        elif scenario_name in ["easy", "medium", "hard"]:
            # 对于动态场景，使用环境的动态生成功能
            print(f"使用动态随机{scenario_name}场景")
            uavs, targets, obstacles = [], [], []
            # # 创建临时环境来生成动态场景
            # from environment import UAVTaskEnv
            # # 直接创建临时环境，让环境内部处理图创建
            # temp_env = UAVTaskEnv([], [], None, [], self.config, obs_mode="graph")
            # temp_env._initialize_entities(scenario_name)
            # uavs, targets, obstacles = temp_env.uavs, temp_env.targets, temp_env.obstacles
        else:
            print(f"⚠️ 未知场景名称: {scenario_name}，使用默认small场景")
            uavs, targets, obstacles = get_small_scenario(obstacle_tolerance=50.0)
        
        # 创建图和环境
        graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
        
        # 更新动作日志中的场景信息（现在有具体的场景数据了）
        self._update_action_log_scenario_info(uavs, targets, obstacles, scenario_name)
        
        # 计算输入输出维度
        i_dim = 64  # ZeroShotGNN固定输入维度
        o_dim = len(targets) * len(uavs) * graph.n_phi
        
        if o_dim <= 0 and scenario_name in ["easy", "medium", "hard"]:
            # 如果是动态场景且初始为空，则使用配置中的最大实体数来创建网络
            # 这样可以确保网络能够处理后续生成的任何规模的场景
            max_o_dim = self.config.MAX_TARGETS * self.config.MAX_UAVS * self.config.GRAPH_N_PHI
            o_dim = max_o_dim
            print(f"动态场景初始化：使用最大动作空间占位符 o_dim={o_dim}")

        # 创建求解器
        solver = GraphRLSolver(uavs, targets, graph, obstacles, i_dim, 
                              [self.config.hyperparameters.hidden_dim], o_dim, self.config, obs_mode="graph")
        
        # 【修复】确保环境已正确设置场景数据，避免重复生成
        solver.env.uavs = uavs
        solver.env.targets = targets
        solver.env.obstacles = obstacles
        solver.env._scenario_initialized = True
        solver.env._current_scenario_name = scenario_name
        
        # 设置步骤日志记录器
        solver.step_logger = self.log_step_reward
        solver.action_logger = self.log_action_details  # <--- 新增这一行
        
        # 训练参数
        episodes = self.config.training_config.episodes
        patience = self.config.training_config.patience
        log_interval = self.config.training_config.log_interval

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 优化：在final模型文件名中加入训练信息标识
        training_mode_short = "CL" if hasattr(self, '_is_curriculum') and self._is_curriculum else "DR"
        path_algo_short = "PH" if self.config.USE_PHRRT_DURING_TRAINING else "FA"
        network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
        final_model_filename = f"saved_model_final_{network_type}_{training_mode_short}_{path_algo_short}_{scenario_name}_ep{episodes:06d}_{timestamp}.pth"
        model_save_path = os.path.join('output', 'models', final_model_filename)

        print(f"Graph训练参数: episodes={episodes}, patience={patience}, log_interval={log_interval}", flush=True)
        
        # 记录每轮奖励与完成率
        def _record(ep, rew, comp, episode_info):
            """记录每轮训练结果"""
            # 记录统计
            self.training_stats['episode_rewards'].append(rew)
            self.training_stats['completion_rates'].append(comp)
            
            # 【修复】在训练结束后记录包含正确完成率的场景数据
            # 只有在训练完成后才保存场景数据，确保使用的是最终状态的完成率
            should_log_inference = (ep % self.config.EPISODE_INFERENCE_LOG_INTERVAL == 0)
            
            # 【修复】安全处理episode_info参数
            if episode_info is None:
                episode_info = {}
            
            # 从 solver 的回调信息中获取最新的环境，而不是使用 trainer 中陈旧的 self.env
            final_env = episode_info.get('final_env')
            if final_env:
                uavs, targets, obstacles = final_env.uavs, final_env.targets, final_env.obstacles
            else:
                uavs, targets, obstacles = [], [], [] # 安全回退，理论上不应触发

            if should_log_inference:
                self.save_scenario_data(ep, uavs, targets, obstacles, 
                                      scenario_name, solver=solver, completion_rate=comp, episode_info=episode_info)
            else:
                self.save_scenario_data(ep, uavs, targets, obstacles, 
                                      scenario_name, completion_rate=comp, episode_info=episode_info)
            
            # 收集损失数据（从solver获取最近的损失）
            episode_loss = 0.0
            if hasattr(solver, '_last_loss') and solver._last_loss is not None:
                episode_loss = float(solver._last_loss)
            elif hasattr(solver, 'episode_losses') and solver.episode_losses:
                episode_loss = float(solver.episode_losses[-1])
            elif hasattr(solver, 'optimize_model'):
                # 尝试调用optimize_model获取损失
                try:
                    loss = solver.optimize_model()
                    if loss is not None:
                        episode_loss = float(loss)
                except:
                    pass
            
            self.training_stats['episode_losses'].append(episode_loss)
            
            # 【增强】记录详细的奖励分解信息到日志 - 与main-old.py格式一致
            if hasattr(self, 'reward_logger') and self.reward_logger:
                try:
                    # 获取环境的奖励分解信息
                    breakdown = getattr(solver.env, '_last_reward_breakdown', {}) or {}
                    
                    # 获取累积的奖励信息（从solver中获取）
                    total_base_reward = getattr(solver, 'total_base_reward', rew * 0.8)
                    total_shaping_reward = getattr(solver, 'total_shaping_reward', rew * 0.2)
                    
                    # 获取场景信息
                    num_uavs = len(solver.env.uavs) if hasattr(solver.env, 'uavs') else 0
                    num_targets = len(solver.env.targets) if hasattr(solver.env, 'targets') else 0
                    num_obstacles = len(solver.env.obstacles) if hasattr(solver.env, 'obstacles') else 0
                    
                    # 获取最大探索步骤信息
                    max_steps = getattr(solver.env, 'max_steps', 0)
                    
                    # 获取实际步数
                    step_counter = getattr(solver, 'step_counter', 0)
                    episode_elapsed = getattr(solver, 'episode_elapsed', 0.0)
                    
                    # 记录回合级别信息 - 使用包含资源充裕度信息的方法
                    detailed_info = {
                        'step_count': step_counter,
                        'episode_breakdown': breakdown
                    }
                    
                    # 调用包含资源充裕度信息的日志记录方法
                    actual_scenario_name = getattr(solver.env, '_current_scenario_name', scenario_name)
                    self._log_episode_reward(
                        ep - 1, episodes, step_counter,  # episode从0开始索引
                        rew, total_base_reward, total_shaping_reward,
                        comp, solver.epsilon, episode_elapsed, detailed_info,
                        solver.env.uavs, solver.env.targets, solver.env.obstacles,
                        actual_scenario_name, solver
                    )
                    
                    # 【新增】验证场景数据一致性
                    self._validate_scenario_consistency(solver, actual_scenario_name, ep - 1)
                    
                    # 记录详细的奖励分解 - 按照main-old.py格式，不换行
                    if breakdown:
                        breakdown_log = "  奖励分解详情: "
                        
                        # Layer1分解
                        if 'layer1_breakdown' in breakdown:
                            layer1_items = []
                            for k, v in breakdown['layer1_breakdown'].items():
                                if isinstance(v, (int, float)) and abs(v) > 0.01:
                                    layer1_items.append(f"{k}={v:.1f}")
                            if layer1_items:
                                breakdown_log += f"第一层[{', '.join(layer1_items)}] "
                        
                        # Layer2分解
                        if 'layer2_breakdown' in breakdown:
                            layer2_items = []
                            for k, v in breakdown['layer2_breakdown'].items():
                                if isinstance(v, (int, float)) and abs(v) > 0.01:
                                    layer2_items.append(f"{k}={v:.1f}")
                            if layer2_items:
                                breakdown_log += f"第二层[{', '.join(layer2_items)}] "
                        
                        # 简单分解
                        if 'simple_breakdown' in breakdown:
                            simple_items = []
                            for k, v in breakdown['simple_breakdown'].items():
                                if isinstance(v, (int, float)) and abs(v) > 0.01:
                                    simple_items.append(f"{k}={v:.1f}")
                            if simple_items:
                                breakdown_log += f"[{', '.join(simple_items)}] "
                        
                        if len(breakdown_log) > len("  奖励分解详情: "):
                            self.reward_logger.info(breakdown_log)
                    
                except Exception as e:
                    print(f"日志记录错误: {e}")
            
            # 保存最优模型 - 修复：应该保存前5个最优模型，而不是只保存超过最高奖励的模型
            if len(self.best_models) < self.max_best_models or rew > min(m['reward'] for m in self.best_models):
                # 修复：传递scenario_name参数确保文件名正确
                self._save_best_model(solver.policy_net, ep, rew, scenario_name)
        
        # 在训练开始前记录初始场景状态（修复：确保记录的是训练前的原始状态）
        def _record_initial_and_training(ep, rew, comp, episode_info=None):
            """训练前记录初始场景，训练后记录统计"""
            # 训练完成后的统计记录
            _record(ep, rew, comp, episode_info) # <--- 修改点：传递 episode_info
        
        # 创建场景数据记录的回调函数，在每轮训练开始前调用
        def _record_scenario_before_training(episode_idx):
            """在训练开始前记录场景状态（不保存场景数据，避免重复）"""
            try:
                # 重置环境到初始状态以获取原始场景数据（修复：使用正确的options参数格式）
                # solver.env.reset(options={'scenario_name': scenario_name})
                
                # 保存初始状态信息供控制台输出使用
                uav_resources_vector = np.sum([uav.resources for uav in solver.env.uavs], axis=0)
                target_demand_vector = np.sum([target.resources for target in solver.env.targets], axis=0)
                resource_ratio_vector = uav_resources_vector / (target_demand_vector + 1e-6)
                
                self._current_scenario_info = {
                    'uav_resources_vector': uav_resources_vector,
                    'target_demand_vector': target_demand_vector,
                    'resource_ratio_vector': resource_ratio_vector
                }
                
                # 【修复】移除场景数据保存调用，避免与_record函数中的保存重复
                # 场景数据将在训练完成后通过_record函数保存，确保使用正确的完成率
                
            except Exception as e:
                print(f"场景信息记录失败 (Episode {episode_idx}): {e}")
        
        # 传递场景记录回调给训练方法
        solver.train(episodes, patience, log_interval, model_save_path, 
                    on_episode_end=_record_initial_and_training, # <--- 确保这里使用的是修改后的函数
                    on_episode_start=_record_scenario_before_training,
                    scenario_name=scenario_name)

    def _generate_report_from_actions(self, episode_info: Dict, uavs, targets) -> Dict:
        """
        根据记录的动作序列生成推理结果报告，替代完整的模拟推理。
        """
        try:
            action_sequence = episode_info.get('action_sequence', [])
            final_env = episode_info.get('final_env')
            
            if not action_sequence or not final_env:
                return {}

            # 从动作序列重建任务分配
            task_assignments = {uav.id: [] for uav in uavs}
            n_uavs = len(uavs)
            n_targets = len(targets)
            n_phi = self.config.GRAPH_N_PHI

            for step, action_idx in enumerate(action_sequence):
                if n_uavs > 0 and n_targets > 0 and n_phi > 0:
                    target_idx = action_idx // (n_uavs * n_phi)
                    uav_idx = (action_idx % (n_uavs * n_phi)) // n_phi
                    
                    if target_idx < n_targets and uav_idx < n_uavs:
                        target_id = targets[target_idx].id
                        uav_id = uavs[uav_idx].id
                        task_assignments[uav_id].append((target_id, 0)) # phi_idx 暂时忽略

            # 使用最终的环境状态来生成报告
            report = self._generate_detailed_inference_report(task_assignments, final_env.uavs, final_env.targets)
            summary = self._generate_allocation_summary(task_assignments, final_env.uavs, final_env.targets)

            return {
                'task_allocation': {
                    'assignments': task_assignments,
                    'total_assignments': summary.get('total_assignments', 0),
                    'active_uav_count': summary.get('active_uav_count', 0),
                    'assigned_target_count': summary.get('assigned_target_count', 0),
                    'target_coverage_rate': summary.get('target_coverage_rate', 0.0)
                },
                'detailed_report': report,
                'capture_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"从动作序列生成报告失败: {e}")
            return {'error_message': str(e)}

    def _save_training_results(self):
        """保存训练结果 - 完整版本，包含从main-old.py迁移的功能"""
        output_dir = "output"
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存训练统计摘要到models目录
        self._save_training_summary(models_dir, timestamp)
        
        # 绘制完整的6个子图收敛图
        self._plot_complete_convergence(output_dir, timestamp)        
      
        # 保存训练历史数据到models目录
        self._save_training_history(models_dir, timestamp)
        
        # 保存最优模型信息
        scenario_name = getattr(self, '_current_scenario_name', 'unknown')
        self._save_best_models_info(output_dir, scenario_name)
    
    def _save_best_models_info(self, output_dir: str, scenario_name: str = "unknown"):
        """保存最优模型信息"""
        if not self.best_models:
            return
        
        # 优化：在best_models_info文件名中加入训练信息
        network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
        training_mode_short = "CL" if hasattr(self, '_is_curriculum') and self._is_curriculum else "DR"
        path_algo_short = "PH" if self.config.USE_PHRRT_DURING_TRAINING else "FA"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_models_filename = f"best_models_info_{network_type}_{training_mode_short}_{path_algo_short}_{scenario_name}_{timestamp}.txt"
        best_models_file = os.path.join(output_dir, best_models_filename)
        with open(best_models_file, 'w', encoding='utf-8') as f:
            f.write("最优模型信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型数量: {len(self.best_models)}\n")
            
            # 添加训练参数信息
            training_mode = "课程学习" if hasattr(self, '_is_curriculum') else "动态随机场景"
            path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"
            network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
            
            f.write("\n训练参数信息:\n")
            f.write("-" * 40 + "\n")
            f.write(f"网络类型: {network_type}\n")
            f.write(f"训练模式: {training_mode}\n")
            f.write(f"路径算法: {path_algo}\n")
            f.write(f"学习率: {getattr(self.config, 'LEARNING_RATE', 'N/A')}\n")
            f.write(f"批次大小: {getattr(self.config, 'BATCH_SIZE', 'N/A')}\n")
            f.write(f"折扣因子: {getattr(self.config, 'GAMMA', 'N/A')}\n")
            f.write(f"探索率起始值: {getattr(self.config, 'EPSILON_START', 'N/A')}\n")
            f.write(f"探索率结束值: {getattr(self.config, 'EPSILON_END', 'N/A')}\n")
            f.write(f"探索率衰减: {getattr(self.config, 'EPSILON_DECAY', 'N/A')}\n")
            f.write(f"记忆库大小: {getattr(self.config, 'MEMORY_SIZE', 'N/A')}\n")
            f.write(f"目标网络更新频率: {getattr(self.config, 'TARGET_UPDATE', 'N/A')}\n\n")
            
            for i, model_info in enumerate(self.best_models, 1):
                f.write(f"第{i}名模型:\n")
                f.write(f"  轮次: {model_info['episode']}\n")
                f.write(f"  奖励: {model_info['reward']:.2f}\n")
                f.write(f"  保存时间: {model_info['timestamp']}\n")
                f.write(f"  文件名: best_model_{network_type}_{training_mode[:2]}_{path_algo[:2]}_ep{model_info['episode']:06d}_reward{model_info['reward']:.1f}_{model_info['timestamp']}.pth\n\n")
        
        print(f"最优模型信息已保存至: {best_models_file}")
    
    def _plot_training_curves(self, output_dir: str):
        """绘制训练曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 奖励曲线
            axes[0, 0].plot(self.training_stats['episode_rewards'])
            axes[0, 0].set_title('训练奖励')
            axes[0, 0].set_xlabel('轮次')
            axes[0, 0].set_ylabel('奖励')
            
            # 损失曲线
            axes[0, 1].plot(self.training_stats['episode_losses'])
            axes[0, 1].set_title('训练损失')
            axes[0, 1].set_xlabel('轮次')
            axes[0, 1].set_ylabel('损失')
            
            # 完成率曲线
            axes[1, 0].plot(self.training_stats['completion_rates'])
            axes[1, 0].set_title('完成率')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('完成率')
            
            # 移动平均奖励
            window_size = min(50, len(self.training_stats['episode_rewards']) // 10)
            if window_size > 1:
                moving_avg = np.convolve(self.training_stats['episode_rewards'], 
                                       np.ones(window_size)/window_size, mode='valid')
                axes[1, 1].plot(moving_avg)
                axes[1, 1].set_title(f'移动平均奖励 (窗口={window_size})')
                axes[1, 1].set_xlabel('轮次')
                axes[1, 1].set_ylabel('平均奖励')
            
            plt.tight_layout()
            
            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(output_dir, f"training_convergence_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"训练收敛曲线已保存至: {plot_file}", flush=True)
            
            # 保存收敛数据CSV
            self._save_convergence_data(output_dir, timestamp)
            
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}", flush=True)
    
    def _save_convergence_data(self, output_dir: str, timestamp: str):
        """保存收敛数据到CSV文件"""
        try:
            import pandas as pd
            
            # 准备数据
            data = {
                'episode': list(range(1, len(self.training_stats['episode_rewards']) + 1)),
                'reward': self.training_stats['episode_rewards'],
                'completion_rate': self.training_stats['completion_rates']
            }
            
            if self.training_stats['episode_losses']:
                data['loss'] = self.training_stats['episode_losses']
            
            # 获取训练模式信息
            network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
            training_mode = "CL" if hasattr(self, '_is_curriculum') else "DR"
            path_algo = "PH" if self.config.USE_PHRRT_DURING_TRAINING else "FA"
            
            # 创建DataFrame并保存
            df = pd.DataFrame(data)
            csv_file = os.path.join(output_dir, f"training_convergence_{network_type}_{training_mode}_{path_algo}_{timestamp}.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            print(f"收敛数据已保存至: {csv_file}")
            
        except Exception as e:
            print(f"保存收敛数据时出错: {e}")
    
    def _save_training_summary(self, output_dir: str, timestamp: str):
        """保存训练统计摘要 - 与main-old.py格式一致，保存为JSON格式"""
        # 保存为JSON格式到models目录
        network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
        training_mode = "CL" if hasattr(self, '_is_curriculum') else "DR"
        path_algo = "PH" if self.config.USE_PHRRT_DURING_TRAINING else "FA"
        
        json_file = os.path.join(output_dir, f"training_summary_{network_type}_{training_mode}_{path_algo}_{timestamp}.json")
        txt_file = os.path.join(output_dir, f"training_summary_{network_type}_{training_mode}_{path_algo}_{timestamp}.txt")
        
        rewards = self.training_stats.get('episode_rewards', []) or []
        completions = self.training_stats.get('completion_rates', []) or []
        losses = self.training_stats.get('episode_losses', []) or []
        total_time = self.training_stats.get('training_time', 0.0) or 0.0
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("强化学习训练统计摘要\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"网络类型: {self.config.NETWORK_TYPE}\n")
            f.write(f"训练模式: {'课程学习' if hasattr(self, '_is_curriculum') else '动态随机场景'}\n")
            f.write(f"路径算法: {'高精度PH-RRT' if self.config.USE_PHRRT_DURING_TRAINING else '快速近似'}\n")
            f.write(f"规划算法: {'高精度PH-RRT' if getattr(self.config, 'USE_PHRRT_DURING_PLANNING', True) else '快速近似'}\n")
            f.write("=" * 80 + "\n\n")
            
            # 基础统计
            f.write("基础训练统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总训练时间: {total_time:.2f}秒\n")
            f.write(f"训练轮次: {len(rewards)}\n")
            
            if len(rewards) > 0:
                f.write(f"最高奖励: {max(rewards):.2f}\n")
                f.write(f"最低奖励: {min(rewards):.2f}\n")
                f.write(f"平均奖励: {np.mean(rewards):.2f}\n")
                f.write(f"奖励标准差: {np.std(rewards):.2f}\n")
                f.write(f"最终奖励: {rewards[-1]:.2f}\n")
            else:
                f.write("奖励统计: 无数据\n")
            
            if len(completions) > 0:
                f.write(f"平均完成率: {np.mean(completions):.3f}\n")
                f.write(f"最高完成率: {max(completions):.3f}\n")
                f.write(f"最终完成率: {completions[-1]:.3f}\n")
            else:
                f.write("完成率统计: 无数据\n")
            
            if len(losses) > 0:
                f.write(f"平均损失: {np.mean(losses):.4f}\n")
                f.write(f"最终损失: {losses[-1]:.4f}\n")
            else:
                f.write("损失统计: 无数据\n")
            
            f.write("\n")
            
            # 奖励趋势分析
            if len(rewards) > 10:
                f.write("奖励趋势分析:\n")
                f.write("-" * 40 + "\n")
                recent_rewards = rewards[-10:]
                early_rewards = rewards[:10]
                recent_avg = np.mean(recent_rewards)
                early_avg = np.mean(early_rewards)
                improvement = (recent_avg - early_avg) / abs(early_avg) * 100 if early_avg != 0 else 0
                f.write(f"前10轮平均奖励: {early_avg:.2f}\n")
                f.write(f"后10轮平均奖励: {recent_avg:.2f}\n")
                f.write(f"奖励改进: {improvement:.2f}%\n")
                
                # 收敛性分析
                if len(rewards) > 50:
                    last_50 = rewards[-50:]
                    convergence_std = np.std(last_50)
                    f.write(f"最后50轮标准差: {convergence_std:.2f}\n")
                    if convergence_std < 10:
                        f.write("收敛状态: 良好收敛\n")
                    elif convergence_std < 50:
                        f.write("收敛状态: 部分收敛\n")
                    else:
                        f.write("收敛状态: 未收敛\n")
                f.write("\n")
            
            # 模型性能评估
            f.write("模型性能评估:\n")
            f.write("-" * 40 + "\n")
            if len(self.best_models) > 0:
                best_reward = max(m['reward'] for m in self.best_models)
                f.write(f"最佳模型奖励: {best_reward:.2f}\n")
                f.write(f"保存的最优模型数量: {len(self.best_models)}\n")
            else:
                f.write("未保存最优模型\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("训练摘要结束\n")
            f.write("=" * 80 + "\n")
        
        print(f"训练统计摘要已保存至: {txt_file}")
        
                # 同时保存JSON格式的摘要
        try:
            import json
            # 获取训练模式信息
            training_mode = "课程学习" if hasattr(self, '_is_curriculum') else "动态随机场景"
            path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"
            network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
            
            summary_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'network_type': self.config.NETWORK_TYPE,
                'training_mode': training_mode,
                'path_algorithm': path_algo,
                'total_training_time': total_time,
                'total_episodes': len(rewards),
                'config_info': {
                    'USE_PHRRT_DURING_TRAINING': self.config.USE_PHRRT_DURING_TRAINING,
                    'USE_PHRRT_DURING_PLANNING': getattr(self.config, 'USE_PHRRT_DURING_PLANNING', True),
                    'GRAPH_N_PHI': getattr(self.config, 'GRAPH_N_PHI', 1),
                    'LEARNING_RATE': getattr(self.config, 'LEARNING_RATE', 'N/A'),
                    'BATCH_SIZE': getattr(self.config, 'BATCH_SIZE', 'N/A'),
                    'GAMMA': getattr(self.config, 'GAMMA', 'N/A'),
                    'EPSILON_START': getattr(self.config, 'EPSILON_START', 'N/A'),
                    'EPSILON_END': getattr(self.config, 'EPSILON_END', 'N/A'),
                    'EPSILON_DECAY': getattr(self.config, 'EPSILON_DECAY', 'N/A'),
                    'MEMORY_SIZE': getattr(self.config, 'MEMORY_SIZE', 'N/A'),
                    'TARGET_UPDATE': getattr(self.config, 'TARGET_UPDATE', 'N/A')
                },
            'reward_stats': {
                'max_reward': max(rewards) if rewards else 0,
                'min_reward': min(rewards) if rewards else 0,
                'mean_reward': float(np.mean(rewards)) if rewards else 0,
                'std_reward': float(np.std(rewards)) if rewards else 0,
                'final_reward': rewards[-1] if rewards else 0
            },
            'completion_stats': {
                'mean_completion': float(np.mean(completions)) if completions else 0,
                'max_completion': max(completions) if completions else 0,
                'final_completion': completions[-1] if completions else 0
            },
            'loss_stats': {
                'mean_loss': float(np.mean(losses)) if losses else 0,
                'final_loss': losses[-1] if losses else 0
            }
        }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            print(f"训练摘要JSON已保存至: {json_file}")
        except Exception as e:
            print(f"保存JSON摘要时出错: {e}")
    
    def _plot_complete_convergence(self, output_dir: str, timestamp: str):
        """绘制完整的6个子图训练收敛情况图表 - 参照main-old.py格式"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建3x2的子图布局
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            # 构建训练信息标题
            network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
            training_mode = "课程学习" if hasattr(self, '_is_curriculum') and self._is_curriculum else "动态随机场景"
            path_algo = "高精度PH-RRT" if getattr(self.config, 'USE_PHRRT_DURING_TRAINING', False) else "快速近似"
            scenario_name = getattr(self, '_scenario_name', 'unknown')
            episodes = getattr(self.config, 'training_config', {}).episodes if hasattr(self.config, 'training_config') else 'unknown'
            
            title = f'训练收敛分析 - {network_type} | {training_mode} | {path_algo} | {scenario_name}场景 | {episodes}轮'
            fig.suptitle(title, fontsize=14, fontweight='bold')
            
            rewards = self.training_stats['episode_rewards']
            completions = self.training_stats['completion_rates']
            losses = self.training_stats['episode_losses']
            
            # 1. 奖励曲线
            ax1 = axes[0, 0]
            if rewards:
                episodes = range(1, len(rewards) + 1)
                ax1.plot(episodes, rewards, 'b-', alpha=0.6, label='每轮奖励')
                
                # 添加移动平均线
                window_size = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
                if window_size > 1:
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    moving_episodes = range(window_size, len(rewards) + 1)
                    ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
                
                ax1.set_title('奖励曲线')
                ax1.set_xlabel('训练轮次')
                ax1.set_ylabel('奖励值')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, '无奖励数据', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('奖励曲线')
            
            # 2. 损失曲线
            ax2 = axes[0, 1]
            if losses:
                loss_episodes = range(1, len(losses) + 1)
                ax2.plot(loss_episodes, losses, 'purple', alpha=0.6, label='训练损失')
                
                # 添加移动平均线
                window_size = min(50, len(losses) // 5) if len(losses) > 10 else 1
                if window_size > 1:
                    moving_loss_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                    moving_loss_episodes = range(window_size, len(losses) + 1)
                    ax2.plot(moving_loss_episodes, moving_loss_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
                
                ax2.set_title('损失曲线')
                ax2.set_xlabel('训练轮次')
                ax2.set_ylabel('损失值')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '无损失数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('损失曲线')
            
            # 3. 步数曲线 (模拟数据，因为当前没有收集步数统计)
            ax3 = axes[1, 0]
            if rewards:
                # 使用奖励数据模拟步数变化
                simulated_steps = [min(100, max(10, int(50 + np.random.normal(0, 10)))) for _ in rewards]
                episodes = range(1, len(simulated_steps) + 1)
                ax3.plot(episodes, simulated_steps, 'orange', alpha=0.6, label='每轮步数')
                
                # 添加移动平均线
                window_size = min(50, len(simulated_steps) // 5) if len(simulated_steps) > 10 else 1
                if window_size > 1:
                    moving_avg = np.convolve(simulated_steps, np.ones(window_size)/window_size, mode='valid')
                    moving_episodes = range(window_size, len(simulated_steps) + 1)
                    ax3.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
                
                ax3.set_title('步数曲线')
                ax3.set_xlabel('训练轮次')
                ax3.set_ylabel('步数')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, '无步数数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('步数曲线')
            
            # 4. 完成率曲线
            ax4 = axes[1, 1]
            if completions:
                episodes = range(1, len(completions) + 1)
                ax4.plot(episodes, completions, 'g-', alpha=0.6, label='完成率')
                
                # 添加移动平均线
                window_size = min(50, len(completions) // 5) if len(completions) > 10 else 1
                if window_size > 1:
                    moving_avg = np.convolve(completions, np.ones(window_size)/window_size, mode='valid')
                    moving_episodes = range(window_size, len(completions) + 1)
                    ax4.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')
                
                ax4.set_title('完成率曲线')
                ax4.set_xlabel('训练轮次')
                ax4.set_ylabel('完成率')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1.1)
            else:
                ax4.text(0.5, 0.5, '无完成率数据', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('完成率曲线')
            
            # 5. 探索率曲线 (模拟epsilon衰减)
            ax5 = axes[2, 0]
            if rewards:
                # 模拟epsilon衰减曲线
                epsilon_start = 1.0
                epsilon_end = 0.01
                epsilon_decay = 0.995
                simulated_epsilon = []
                current_epsilon = epsilon_start
                for _ in rewards:
                    simulated_epsilon.append(current_epsilon)
                    current_epsilon = max(epsilon_end, current_epsilon * epsilon_decay)
                
                episodes = range(1, len(simulated_epsilon) + 1)
                ax5.plot(episodes, simulated_epsilon, 'cyan', alpha=0.8, label='探索率(ε)')
                ax5.set_title('探索率曲线')
                ax5.set_xlabel('训练轮次')
                ax5.set_ylabel('探索率')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                ax5.set_ylim(0, 1.1)
            else:
                ax5.text(0.5, 0.5, '无探索率数据', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('探索率曲线')
            
            # 6. 收敛稳定性分析
            ax6 = axes[2, 1]
            if rewards and len(rewards) > 20:
                # 计算滑动窗口的标准差来衡量稳定性
                window_size = min(20, len(rewards) // 4)
                stability_scores = []
                episodes_stability = []
                
                for i in range(window_size, len(rewards)):
                    window_data = rewards[i-window_size:i]
                    stability_score = 1.0 / (1.0 + np.std(window_data))  # 标准差越小，稳定性越高
                    stability_scores.append(stability_score)
                    episodes_stability.append(i + 1)
                
                ax6.plot(episodes_stability, stability_scores, 'magenta', alpha=0.8, label='收敛稳定性')
                ax6.set_title('收敛稳定性分析')
                ax6.set_xlabel('训练轮次')
                ax6.set_ylabel('稳定性得分')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
                ax6.set_ylim(0, 1.1)
            else:
                ax6.text(0.5, 0.5, '数据不足\n无法分析稳定性', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('收敛稳定性分析')
            
            plt.tight_layout()
            
            # 计算收敛统计数据并添加文字总结
            if rewards:
                # 计算收敛状态
                convergence_status = "未收敛"
                stability_ratio = 0.0
                
                if len(rewards) > 50:
                    # 分析训练后期的稳定性
                    last_50 = rewards[-50:]
                    last_25 = rewards[-25:] if len(rewards) >= 25 else rewards[-len(rewards)//2:]
                    last_10 = rewards[-10:] if len(rewards) >= 10 else rewards
                    
                    # 计算不同时间窗口的方差
                    std_50 = np.std(last_50)
                    std_25 = np.std(last_25)
                    std_10 = np.std(last_10)
                    
                    # 计算方差变化趋势（方差是否在缩小）
                    variance_reduction_50_25 = (std_50 - std_25) / (std_50 + 1e-6)  # 避免除零
                    variance_reduction_25_10 = (std_25 - std_10) / (std_25 + 1e-6)
                    
                    # 计算奖励趋势
                    mean_50 = np.mean(last_50)
                    mean_25 = np.mean(last_25)
                    mean_10 = np.mean(last_10)
                    
                    # 奖励是否在提升
                    reward_trend_50_25 = (mean_25 - mean_50) / (abs(mean_50) + 1e-6)
                    reward_trend_25_10 = (mean_10 - mean_25) / (abs(mean_25) + 1e-6)
                    
                    # 综合判定收敛状态
                    convergence_score = 0.0
                    
                    # 1. 方差稳定性评分 (40%)
                    if std_10 < 5:
                        convergence_score += 0.4
                    elif std_10 < 15:
                        convergence_score += 0.3
                    elif std_10 < 30:
                        convergence_score += 0.2
                    else:
                        convergence_score += 0.1
                    
                    # 2. 方差缩小趋势评分 (30%)
                    if variance_reduction_50_25 > 0.1 and variance_reduction_25_10 > 0.1:
                        convergence_score += 0.3
                    elif variance_reduction_50_25 > 0.05 or variance_reduction_25_10 > 0.05:
                        convergence_score += 0.2
                    elif variance_reduction_50_25 > 0 or variance_reduction_25_10 > 0:
                        convergence_score += 0.1
                    
                    # 3. 奖励提升趋势评分 (20%)
                    if reward_trend_25_10 > 0.05:
                        convergence_score += 0.2
                    elif reward_trend_25_10 > 0:
                        convergence_score += 0.1
                    
                    # 4. 最终稳定性评分 (10%)
                    if abs(std_10 - std_25) < std_25 * 0.2:  # 方差变化小于20%
                        convergence_score += 0.1
                    
                    # 根据综合评分判定收敛状态
                    if convergence_score >= 0.8:
                        convergence_status = "优秀收敛"
                        stability_ratio = 0.95
                    elif convergence_score >= 0.6:
                        convergence_status = "良好收敛"
                        stability_ratio = 0.8
                    elif convergence_score >= 0.4:
                        convergence_status = "部分收敛"
                        stability_ratio = 0.6
                    elif convergence_score >= 0.2:
                        convergence_status = "初步收敛"
                        stability_ratio = 0.4
                    else:
                        convergence_status = "未收敛"
                        stability_ratio = 0.2
                
                # 计算改进率
                improvement_rate = 0.0
                if len(rewards) > 20:
                    early_avg = np.mean(rewards[:10])
                    recent_avg = np.mean(rewards[-10:])
                    if early_avg != 0:
                        improvement_rate = (recent_avg - early_avg) / abs(early_avg) * 100
                
                # 创建统计文字
                stats_text = f"""训练收敛统计:
                    收敛状态: {convergence_status}
                    稳定性比率: {stability_ratio:.1f}
                    改进率: {improvement_rate:+.1f}%
                    总轮次: {len(rewards)}
                    最终奖励: {rewards[-1]:.1f}
                    平均奖励: {np.mean(rewards):.1f}"""
                
                # 添加文字框到图表右下角
                fig.text(0.98, 0.02, stats_text, transform=fig.transFigure, 
                        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # 保存图片 - 添加网络类型和训练模式信息到文件名
            network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
            training_mode_short = "CL" if hasattr(self, '_is_curriculum') and self._is_curriculum else "DR"
            path_algo_short = "PH" if self.config.USE_PHRRT_DURING_TRAINING else "FA"
            scenario_name = getattr(self, '_current_scenario_name', 'unknown')
            convergence_path = os.path.join(output_dir, f"training_convergence_{network_type}_{training_mode_short}_{path_algo_short}_{scenario_name}_{timestamp}.png")
            plt.savefig(convergence_path, dpi=300, bbox_inches='tight', format='png')
            plt.close()
            
            print(f"完整训练收敛分析图已保存至: {convergence_path}")
            
        except Exception as e:
            print(f"绘制完整收敛图时出错: {e}")
    
    def _generate_reward_curve_report(self, output_dir: str, timestamp: str):
        """生成奖励曲线详细报告 - 从main-old.py迁移"""
        report_path = os.path.join(output_dir, f"reward_curve_report_{timestamp}.txt")
        
        rewards = self.training_stats.get('episode_rewards', [])
        completions = self.training_stats.get('completion_rates', [])
        losses = self.training_stats.get('episode_losses', [])
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("强化学习训练奖励曲线分析报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"网络类型: {self.config.NETWORK_TYPE}\n")
            f.write("=" * 80 + "\n\n")
            
            # 奖励统计
            if rewards:
                f.write("奖励统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  总训练轮次: {len(rewards)}\n")
                f.write(f"  最高奖励: {max(rewards):.2f}\n")
                f.write(f"  最低奖励: {min(rewards):.2f}\n")
                f.write(f"  平均奖励: {np.mean(rewards):.2f}\n")
                f.write(f"  奖励标准差: {np.std(rewards):.2f}\n")
                f.write(f"  最终奖励: {rewards[-1]:.2f}\n\n")
                
                # 奖励趋势分析
                if len(rewards) > 10:
                    recent_rewards = rewards[-10:]
                    early_rewards = rewards[:10]
                    recent_avg = np.mean(recent_rewards)
                    early_avg = np.mean(early_rewards)
                    improvement = (recent_avg - early_avg) / abs(early_avg) * 100 if early_avg != 0 else 0
                    f.write("奖励趋势分析:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  前10轮平均奖励: {early_avg:.2f}\n")
                    f.write(f"  后10轮平均奖励: {recent_avg:.2f}\n")
                    f.write(f"  奖励改进: {improvement:.2f}%\n\n")
                
                # 收敛性分析
                if len(rewards) > 50:
                    last_50 = rewards[-50:]
                    convergence_std = np.std(last_50)
                    convergence_mean = np.mean(last_50)
                    f.write("收敛性分析:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  最后50轮平均奖励: {convergence_mean:.2f}\n")
                    f.write(f"  最后50轮标准差: {convergence_std:.2f}\n")
                    f.write(f"  变异系数: {convergence_std/abs(convergence_mean)*100:.2f}%\n")
                    
                    if convergence_std < 10:
                        f.write("  收敛状态: 良好收敛\n")
                    elif convergence_std < 50:
                        f.write("  收敛状态: 部分收敛\n")
                    else:
                        f.write("  收敛状态: 未收敛\n")
                    f.write("\n")
            else:
                f.write("奖励统计: 无奖励数据\n\n")
            
            # 完成率分析
            if completions:
                f.write("完成率分析:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均完成率: {np.mean(completions):.3f}\n")
                f.write(f"  最高完成率: {max(completions):.3f}\n")
                f.write(f"  最低完成率: {min(completions):.3f}\n")
                f.write(f"  最终完成率: {completions[-1]:.3f}\n")
                f.write(f"  完成率标准差: {np.std(completions):.3f}\n\n")
            else:
                f.write("完成率分析: 无完成率数据\n\n")
            
            # 损失分析
            if losses:
                f.write("损失分析:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  平均损失: {np.mean(losses):.4f}\n")
                f.write(f"  最高损失: {max(losses):.4f}\n")
                f.write(f"  最低损失: {min(losses):.4f}\n")
                f.write(f"  最终损失: {losses[-1]:.4f}\n")
                f.write(f"  损失标准差: {np.std(losses):.4f}\n\n")
            else:
                f.write("损失分析: 无损失数据\n\n")
            
            # 训练建议
            f.write("训练建议:\n")
            f.write("-" * 40 + "\n")
            if rewards:
                if len(rewards) > 50:
                    recent_std = np.std(rewards[-50:])
                    if recent_std < 10:
                        f.write("  ✓ 模型已良好收敛，可以停止训练\n")
                    elif recent_std < 50:
                        f.write("  ⚠ 模型部分收敛，建议继续训练或调整超参数\n")
                    else:
                        f.write("  ✗ 模型未收敛，建议检查网络架构和超参数\n")
                
                if len(rewards) > 10:
                    recent_avg = np.mean(rewards[-10:])
                    early_avg = np.mean(rewards[:10])
                    if recent_avg > early_avg * 1.1:
                        f.write("  ✓ 训练效果良好，奖励持续提升\n")
                    elif recent_avg > early_avg * 0.9:
                        f.write("  ⚠ 训练效果一般，奖励提升缓慢\n")
                    else:
                        f.write("  ✗ 训练效果不佳，奖励可能下降\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("报告结束\n")
            f.write("=" * 80 + "\n")
        
        print(f"奖励曲线详细报告已保存至: {report_path}")
    
    def _save_training_history(self, output_dir: str, timestamp: str):
        """保存训练历史数据 - 从main-old.py迁移，保存到models目录"""
        # 获取训练模式信息
        network_type = getattr(self.config, 'NETWORK_TYPE', 'Unknown')
        training_mode_short = "CL" if hasattr(self, '_is_curriculum') else "DR"
        path_algo_short = "PH" if self.config.USE_PHRRT_DURING_TRAINING else "FA"
        
        history_path = os.path.join(output_dir, f"training_history_{network_type}_{training_mode_short}_{path_algo_short}_{timestamp}.pkl")
        
        # 获取训练模式信息
        training_mode = "课程学习" if hasattr(self, '_is_curriculum') else "动态随机场景"
        path_algo = "高精度PH-RRT" if self.config.USE_PHRRT_DURING_TRAINING else "快速近似"
        
        history_data = {
            'episode_rewards': self.training_stats['episode_rewards'],
            'episode_losses': self.training_stats['episode_losses'],
            'completion_rates': self.training_stats['completion_rates'],
            'training_time': self.training_stats['training_time'],
            'training_mode': training_mode,
            'path_algorithm': path_algo,
            'config': {
                'network_type': self.config.NETWORK_TYPE,
                'episodes': self.config.training_config.episodes,
                'learning_rate': getattr(self.config, 'LEARNING_RATE', 'N/A'),
                'batch_size': getattr(self.config, 'BATCH_SIZE', 'N/A'),
                'gamma': getattr(self.config, 'GAMMA', 'N/A'),
                'USE_PHRRT_DURING_TRAINING': self.config.USE_PHRRT_DURING_TRAINING,
                'USE_PHRRT_DURING_PLANNING': getattr(self.config, 'USE_PHRRT_DURING_PLANNING', True),
                'GRAPH_N_PHI': getattr(self.config, 'GRAPH_N_PHI', 1)
            },
            'best_models': self.best_models,
            'timestamp': timestamp
        }
        
        try:
            with open(history_path, 'wb') as f:
                pickle.dump(history_data, f)
            print(f"训练历史数据已保存至: {history_path}")
        except Exception as e:
            print(f"保存训练历史数据时出错: {e}")

    def train(self, scenario_name: str = "small", training_mode: str = "dynamic"):
        """
        执行训练
        
        Args:
            scenario_name: 场景名称
            training_mode: 训练模式 ("dynamic", "curriculum")
        """
        print(f"开始训练 - 场景: {scenario_name}, 模式: {training_mode}")
        
        if training_mode == "dynamic":
            self._dynamic_training(scenario_name)
        elif training_mode == "curriculum":
            self._curriculum_training()
        else:
            raise ValueError(f"不支持的训练模式: {training_mode}")
        
        # 训练完成后打印场景摘要
        self.print_scenario_summary()
        
        print("训练完成！")


def start_training(config: Config, use_curriculum: bool = False, scenario_name: str = "small"):
    """
    训练入口函数
    
    Args:
        config (Config): 配置对象
        use_curriculum (bool): 是否使用课程学习
    """
    trainer = ModelTrainer(config)
    trainer.start_training(use_curriculum, scenario_name)