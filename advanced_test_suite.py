# -*- coding: utf-8 -*-
# 文件名: advanced_test_suite.py
# 描述: 先进的测试套件，用于评估和对比一个或多个模型在不同规模场景下的规划能力。
#
# ======================================================================================
#                                   如何使用
# ======================================================================================
#
# 1. 增量测试模式 (默认):
#    逐步增加无人机和目标的数量，测试模型在不同规模下的性能。
#    命令:
#    python advanced_test_suite.py --models ./output/300.pth --test-mode incremental
#    python advanced_test_suite.py --models ./output/300.pth --test-mode random --num-random-scenarios 1
# 2. 随机测试模式:
#    生成指定数量的完全随机场景（无人机和目标数量在配置范围内随机），测试模型的泛化能力。
#    命令:
#    python advanced_test_suite.py --models ./output/300.pth --test-mode random --num-random-scenarios 10
#
# 3. 多模型对比:
#    在任意模式下，提供多个模型路径，脚本会在完全相同的场景下对它们进行测试。
#    命令:
#    python advanced_test_suite.py --models model_A.pth model_B.pth --test-mode random --num-random-scenarios 5
#
# ======================================================================================

import os
import time
import argparse
import torch
import numpy as np
import pickle
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    font_names = ['SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI', 'DejaVu Sans']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font_name in font_names:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return font_name
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return "Default"

# 设置中文字体
setup_chinese_font()

# 导入项目模块
from config import Config
from entities import UAV, Target
from path_planning import CircularObstacle
from environment import UAVTaskEnv, DirectedGraph
from networks import create_network
from evaluator import ModelEvaluator, PlanVisualizer
from evaluate import evaluate_plan

def generate_test_scenario(num_uavs: int, num_targets: int, num_obstacles: int, config: Config):
    """
    【修复版本】直接生成指定数量的实体，不依赖环境重置
    确保测试场景的数量与参数完全一致
    """
    # 导入课程训练模式的场景生成函数
    from scenarios import _generate_scenario
    
    # 计算合理的资源富裕度
    resource_abundance = 1.2  # 固定为1.2倍，确保测试的一致性
    
    print(f"🏗️  生成测试场景: UAV={num_uavs}, Target={num_targets}, Obstacle={num_obstacles}")
    
    # 调用课程训练模式的场景生成函数
    scenario_dict = _generate_scenario(
        config=config,
        uav_num=num_uavs,
        target_num=num_targets,
        obstacle_num=num_obstacles,
        resource_abundance=resource_abundance
    )
    
    # 验证生成的实体数量
    actual_uavs = len(scenario_dict['uavs'])
    actual_targets = len(scenario_dict['targets'])
    actual_obstacles = len(scenario_dict['obstacles'])
    
    if (actual_uavs != num_uavs or actual_targets != num_targets or actual_obstacles != num_obstacles):
        print(f"⚠️  场景生成数量不匹配:")
        print(f"   期望: UAV={num_uavs}, Target={num_targets}, Obstacle={num_obstacles}")
        print(f"   实际: UAV={actual_uavs}, Target={actual_targets}, Obstacle={actual_obstacles}")
    else:
        print(f"✅ 场景生成成功，数量匹配")
    
    return scenario_dict['uavs'], scenario_dict['targets'], scenario_dict['obstacles']
    obstacle_centers = np.random.uniform(map_size * 0.15, map_size * 0.85, size=(num_obstacles, 2))
    for i in range(num_obstacles):
        radius = np.random.uniform(map_size * 0.02, map_size * 0.06)
        obstacles.append(CircularObstacle(center=obstacle_centers[i], radius=radius, tolerance=50.0))
        
    return uavs, targets, obstacles

class ModelTestSuiteRunner:
    """模型测试套件运行器"""
    def __init__(self, model_paths: list, config: Config, output_dir: str):
        self.model_paths = model_paths
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"测试结果将保存至: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 一次性加载所有模型
        self.networks = self._load_models()
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = PlanVisualizer(self.config)

        # 初始化CSV文件
        self.csv_path = os.path.join(self.output_dir, f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self._init_csv()

    def _load_models(self) -> dict:
        """一次性加载所有指定的模型"""
        loaded_networks = {}
        for model_path in self.model_paths:
            print(f"正在从 {model_path} 加载模型...")
            if not os.path.exists(model_path):
                print(f"警告: 模型文件未找到，跳过: {model_path}")
                continue
            
            i_dim = 64
            max_o_dim = self.config.MAX_TARGETS * self.config.MAX_UAVS * self.config.GRAPH_N_PHI
            
            network = create_network(
                self.config.NETWORK_TYPE, i_dim, self.config.hyperparameters.hidden_dim,
                max_o_dim, self.config
            ).to(self.device)
            
            try:
                try:
                    # 优先尝试使用 weights_only=False，以兼容包含非Tensor数据类型的模型文件
                    model_data = torch.load(model_path, map_location=self.device, weights_only=False)
                except TypeError:
                    # 如果PyTorch版本过旧不支持weights_only参数，则回退到原始加载方式
                    print("当前PyTorch版本不支持weights_only参数，使用默认方式加载。")
                    model_data = torch.load(model_path, map_location=self.device)
                state_dict = model_data['model_state_dict'] if isinstance(model_data, dict) and 'model_state_dict' in model_data else model_data
                network.load_state_dict(state_dict)
                network.eval()
                loaded_networks[model_path] = network
                print(f"模型 {os.path.basename(model_path)} 加载成功。")
            except Exception as e:
                print(f"加载模型 {os.path.basename(model_path)} 失败: {e}")
        
        if not loaded_networks:
            raise RuntimeError("未能成功加载任何模型，测试中止。")
        return loaded_networks

    def _init_csv(self):
        """初始化CSV文件并写入表头"""
        self.csv_fieldnames = [
            'timestamp', 'model_name', 'inference_mode', 'scenario_name',
            'num_uavs', 'num_targets', 'num_obstacles', 'resource_abundance',
            'total_reward_score', 'completion_rate', 'satisfied_targets_rate',
            'resource_utilization_rate', 'load_balance_score', 'sync_feasibility_rate',
            'total_distance', 'resource_penalty', 'is_deadlocked', 'deadlocked_uav_count',
            'inference_time_s', 'scenario_txt_path', 'result_plot_path', 'graph_plot_path'
        ]
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        print(f"结果将记录在CSV文件: {self.csv_path}")

    def _analyze_plan_details(self, final_plan: dict, uavs: list, targets: list, obstacles: list) -> dict:
        """
        分析方案的详细信息，包括路径长度、资源利用率等
        
        Returns:
            dict: 包含详细分析结果的字典
        """
        analysis = {
            'total_path_length': 0.0,
            'avg_path_length_per_uav': 0.0,
            'max_path_length': 0.0,
            'min_path_length': float('inf'),
            'active_uav_count': 0,
            'idle_uav_count': 0,
            'total_resource_consumption': np.zeros(2),
            'resource_utilization_rate': 0.0,
            'task_distribution': {},
            'path_details': {}
        }
        
        from distance_service import get_distance_service
        distance_service = get_distance_service(self.config, obstacles)
        
        active_uavs = 0
        total_path_lengths = []
        
        for uav in uavs:
            uav_tasks = final_plan.get(uav.id, [])
            if not uav_tasks:
                analysis['idle_uav_count'] += 1
                continue
                
            active_uavs += 1
            current_pos = uav.position
            uav_path_length = 0.0
            uav_resource_consumption = np.zeros(2)
            
            # 按步骤排序任务
            sorted_tasks = sorted(uav_tasks, key=lambda x: x.get('step', 0))
            
            for task in sorted_tasks:
                # 找到目标位置
                target = next((t for t in targets if t.id == task['target_id']), None)
                if target:
                    # 计算路径长度
                    distance = distance_service.calculate_distance(
                        current_pos.tolist(), target.position.tolist(), mode='planning'
                    )
                    uav_path_length += distance
                    current_pos = target.position
                    
                    # 累计资源消耗
                    resource_cost = task.get('resource_cost', np.zeros(2))
                    uav_resource_consumption += resource_cost
            
            total_path_lengths.append(uav_path_length)
            analysis['total_path_length'] += uav_path_length
            analysis['total_resource_consumption'] += uav_resource_consumption
            analysis['path_details'][uav.id] = {
                'path_length': uav_path_length,
                'task_count': len(sorted_tasks),
                'resource_consumption': uav_resource_consumption.tolist()
            }
        
        analysis['active_uav_count'] = active_uavs
        
        if total_path_lengths:
            analysis['avg_path_length_per_uav'] = analysis['total_path_length'] / active_uavs
            analysis['max_path_length'] = max(total_path_lengths)
            analysis['min_path_length'] = min(total_path_lengths)
        else:
            analysis['min_path_length'] = 0.0
            
        # 计算资源利用率
        total_initial_resources = np.sum([uav.initial_resources for uav in uavs], axis=0)
        if np.sum(total_initial_resources) > 0:
            analysis['resource_utilization_rate'] = np.sum(analysis['total_resource_consumption']) / np.sum(total_initial_resources)
        
        # 任务分布统计
        for target in targets:
            target_tasks = []
            for uav_id, tasks in final_plan.items():
                target_tasks.extend([t for t in tasks if t['target_id'] == target.id])
            analysis['task_distribution'][target.id] = len(target_tasks)
        
        return analysis

    def _save_scenario_as_txt(self, scenario_data: dict, filepath: str):
        """保存场景数据为TXT格式，重新编排便于阅读"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 标题和基本信息
                f.write("=" * 60 + "\n")
                f.write(f"场景测试报告 - {scenario_data.get('scenario_name', 'N/A')}\n")
                f.write("=" * 60 + "\n")
                f.write(f"测试时间: {scenario_data.get('timestamp', 'N/A')}\n")
                f.write(f"网络类型: {self.config.NETWORK_TYPE}\n\n")
                
                # 场景概览
                f.write("场景概览:\n")
                f.write("-" * 30 + "\n")
                f.write(f"UAV数量: {scenario_data.get('uav_count', 'N/A')}  ")
                f.write(f"目标数量: {scenario_data.get('target_count', 'N/A')}  ")
                f.write(f"障碍物数量: {scenario_data.get('obstacle_count', 'N/A')}\n\n")
                
                # 资源统计
                uav_total_res = np.sum([u.initial_resources for u in scenario_data['uavs']], axis=0)
                target_total_demand = np.sum([t.resources for t in scenario_data['targets']], axis=0)
                abundance = uav_total_res / (target_total_demand + 1e-6)
                obstacle_area = sum(np.pi * o.radius**2 for o in scenario_data['obstacles'])
                map_area = self.config.MAP_SIZE**2
                
                f.write("资源统计:\n")
                f.write("-" * 30 + "\n")
                f.write(f"总供给: {uav_total_res.astype(int)}  总需求: {target_total_demand.astype(int)}\n")
                f.write(f"资源充裕度: [{abundance[0]:.2f}, {abundance[1]:.2f}]  障碍物覆盖率: {obstacle_area / map_area:.1%}\n\n")
                
                # UAV信息（紧凑格式）
                f.write("UAV配置:\n")
                f.write("-" * 30 + "\n")
                for i, uav in enumerate(scenario_data['uavs']):
                    if i % 2 == 0 and i > 0:
                        f.write("\n")
                    f.write(f"UAV{uav.id}[{uav.position[0]:.0f},{uav.position[1]:.0f}]:{uav.initial_resources}  ")
                f.write("\n\n")
                
                # 目标信息（紧凑格式）
                f.write("目标配置:\n")
                f.write("-" * 30 + "\n")
                for i, target in enumerate(scenario_data['targets']):
                    if i % 2 == 0 and i > 0:
                        f.write("\n")
                    f.write(f"T{target.id}[{target.position[0]:.0f},{target.position[1]:.0f}]:{target.resources}  ")
                f.write("\n\n")
                
                # 障碍物信息（紧凑格式）
                if scenario_data['obstacles']:
                    f.write("障碍物配置:\n")
                    f.write("-" * 30 + "\n")
                    for i, obstacle in enumerate(scenario_data['obstacles']):
                        if i % 3 == 0 and i > 0:
                            f.write("\n")
                        f.write(f"O{i+1}[{obstacle.center[0]:.0f},{obstacle.center[1]:.0f}]r{obstacle.radius:.0f}  ")
                    f.write("\n\n")
                
                # 推理报告
                if scenario_data.get('inference_report'):
                    f.write("推理结果:\n")
                    f.write("-" * 30 + "\n")
                    f.write(scenario_data['inference_report'])
                
                f.write("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"保存TXT场景文件失败: {e}")

    def _plot_assignment_graph(self, final_plan: dict, uavs: list, targets: list, output_path: str, title: str):
        """绘制新的任务分配关系图，不显示障碍物。"""
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(16, 16))
            ax.set_facecolor("#fdfdfd")
            target_pos_map = {t.id: t.position for t in targets}
            ax.scatter([p[0] for p in target_pos_map.values()], [p[1] for p in target_pos_map.values()], s=400, c='red', alpha=0.7, label='目标', marker='o', edgecolors='black')
            for t_id, pos in target_pos_map.items():
                ax.text(pos[0], pos[1], f"T{t_id}", ha='center', va='center', color='white', fontweight='bold')
            uav_pos_map = {u.id: u.position for u in uavs}
            ax.scatter([p[0] for p in uav_pos_map.values()], [p[1] for p in uav_pos_map.values()], s=400, c='blue', alpha=0.7, label='UAV', marker='s', edgecolors='black')
            for u_id, pos in uav_pos_map.items():
                ax.text(pos[0], pos[1], f"UAV{u_id}", ha='center', va='center', color='white', fontweight='bold')
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(uavs)))
            color_map = {u.id: colors[i] for i, u in enumerate(uavs)}
            for uav_id, tasks in final_plan.items():
                if not tasks: continue
                uav = next((u for u in uavs if u.id == uav_id), None)
                if not uav: continue
                current_pos = uav.position
                sorted_tasks = sorted(tasks, key=lambda x: x.get('step', 0))
                for task in sorted_tasks:
                    target = next((t for t in targets if t.id == task['target_id']), None)
                    if not target: continue
                    target_p = target.position
                    ax.annotate("", xy=target_p, xytext=current_pos, arrowprops=dict(arrowstyle="->", color=color_map[uav_id], shrinkA=15, shrinkB=15, lw=2, connectionstyle="arc3,rad=0.1"))
                    mid_point = (current_pos + target_p) / 2
                    label = f"S{task.get('step', 'N/A')}\nD:{task.get('distance', 0):.0f}m"
                    ax.text(mid_point[0], mid_point[1] + 10, label, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc=color_map[uav_id], ec="none", alpha=0.3))
                    current_pos = target_p
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('X Coordinate (m)'); ax.set_ylabel('Y Coordinate (m)')  # 使用英文避免字体警告
            ax.legend(); ax.grid(True, linestyle='--', alpha=0.5); ax.set_aspect('equal', adjustable='box')
            
            # 抑制字体警告
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"绘制任务分配关系图失败: {e}")

    def _process_scenario(self, uavs: list, targets: list, obstacles: list, scenario_name: str):
        """对单个生成好的场景，使用所有已加载的模型进行测试和保存"""
        print(f"\n{'='*80}")
        print(f"🎯 开始处理场景: {scenario_name}")
        print(f"📊 输入实体数量: UAV={len(uavs)}, Target={len(targets)}, Obstacle={len(obstacles)}")
        
        # 计算场景资源概况
        total_uav_resources = np.sum([uav.initial_resources for uav in uavs], axis=0)
        total_target_demand = np.sum([target.resources for target in targets], axis=0)
        resource_abundance = total_uav_resources / (total_target_demand + 1e-6)
        print(f"💰 资源概况: 供给{total_uav_resources} / 需求{total_target_demand} = 充裕度{resource_abundance}")
        
        # 判断是否使用集成推理（当有多个模型时）
        if len(self.networks) > 1:
            print(f"🔀 执行集成推理 | 模型数量: {len(self.networks)}")
            
            # 【精确推理时间记录】开始
            inference_start_time = time.time()
            results = self.evaluator._ensemble_inference(self.model_paths, uavs, targets, obstacles, scenario_name=scenario_name)
            pure_inference_time = time.time() - inference_start_time
            # 【精确推理时间记录】结束
            
            if not results:
                print("❌ 集成推理失败，跳过此场景测试。")
                return
                
            # 使用第一个模型的名称作为集成推理的标识
            model_name = "ensemble_" + "_".join([os.path.basename(path)[:10] for path in self.model_paths[:3]])
            if len(self.model_paths) > 3:
                model_name += f"_and_{len(self.model_paths)-3}_more"
                
            print(f"✅ 集成推理完成，纯推理耗时: {pure_inference_time:.3f}s")
            
            # 对于集成推理，需要获取环境信息
            env = None  # 集成推理中环境信息需要从results中获取
            
        else:
            # 单模型推理逻辑
            model_path = list(self.networks.keys())[0]
            network = self.networks[model_path]
            model_name = os.path.basename(model_path)
            print(f"🤖 执行单模型推理: {model_name}")

            # 创建当前场景的环境
            print("🔄 创建推理环境...")
            env_creation_start = time.time()
            graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
            env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")
            env_creation_time = time.time() - env_creation_start
            
            # 记录环境创建后的实际实体数量
            actual_uav_count = len(env.uavs)
            actual_target_count = len(env.targets) 
            actual_obstacle_count = len(env.obstacles)
            
            print(f"🔄 环境创建完成，耗时: {env_creation_time:.3f}s")
            
            if (actual_uav_count != len(uavs) or actual_target_count != len(targets) or 
                actual_obstacle_count != len(obstacles)):
                print(f"⚠️  环境重置后实体数量发生变化:")
                print(f"   UAV: {len(uavs)} → {actual_uav_count}")
                print(f"   Target: {len(targets)} → {actual_target_count}")
                print(f"   Obstacle: {len(obstacles)} → {actual_obstacle_count}")

            # 【精确推理时间记录】开始
            print("🧠 开始神经网络推理...")
            inference_start_time = time.time()
            results = self.evaluator._run_inference(network, env, use_softmax_sampling=True, scenario_name=scenario_name)
            pure_inference_time = time.time() - inference_start_time
            # 【精确推理时间记录】结束
            
            print(f"✅ 神经网络推理完成，纯推理耗时: {pure_inference_time:.3f}s")
            
            if not results:
                print("❌ 推理失败，跳过此模型的本次测试。")
                return

        # 【开始方案分析和评估】- 不计入推理时间
        print("📊 开始方案分析和评估...")
        analysis_start_time = time.time()
        
        # 使用环境中的实际实体进行评估
        eval_uavs = env.uavs if env else uavs
        eval_targets = env.targets if env else targets
        eval_obstacles = env.obstacles if env else obstacles
        
        # 评估和保存结果
        action_sequence = results.get('action_sequence', [])
        step_details = results.get('step_details', [])
        plan_data = self.evaluator._build_execution_plan_from_action_sequence(action_sequence, eval_uavs, eval_targets, env, step_details)
        final_plan = plan_data.get('uav_assignments', {})
        metrics = evaluate_plan(final_plan, eval_uavs, eval_targets, final_uav_states=results.get('final_uav_states'))
        
        # 【增强方案信息】计算详细的路径和资源信息
        plan_analysis = self._analyze_plan_details(final_plan, eval_uavs, eval_targets, eval_obstacles)
        
        analysis_time = time.time() - analysis_start_time
        print(f"📊 方案分析完成，耗时: {analysis_time:.3f}s")
        
        # 【输出详细方案信息】
        print(f"\n📋 方案详细信息:")
        print(f"   🛣️  总路径长度: {plan_analysis['total_path_length']:.1f}m")
        print(f"   📏 平均路径长度: {plan_analysis['avg_path_length_per_uav']:.1f}m/UAV")
        print(f"   📈 最长路径: {plan_analysis['max_path_length']:.1f}m")
        print(f"   📉 最短路径: {plan_analysis['min_path_length']:.1f}m")
        print(f"   🚁 活跃UAV: {plan_analysis['active_uav_count']}/{len(eval_uavs)}")
        print(f"   😴 空闲UAV: {plan_analysis['idle_uav_count']}/{len(eval_uavs)}")
        print(f"   ⛽ 资源利用率: {plan_analysis['resource_utilization_rate']:.1%}")
        print(f"   🎯 完成率: {metrics.get('completion_rate', 0):.1%}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_suffix = f"{scenario_name}_{model_name[:20]}_{timestamp}"

        # 【开始文件保存】- 不计入推理时间
        print("💾 开始保存结果文件...")
        file_save_start = time.time()
        
        # 保存标准可视化图
        report_content, img_path = self.visualizer.save(
            final_plan, eval_uavs, eval_targets, eval_obstacles, scenario_name=f"Test_{scenario_name}",
            training_time=0, plan_generation_time=pure_inference_time,  # 使用纯推理时间
            evaluation_metrics=metrics, suffix=f"_{model_name[:20]}_{timestamp}"
        )
        final_img_path = os.path.join(self.output_dir, os.path.basename(img_path))
        if os.path.exists(img_path): os.rename(img_path, final_img_path)

            # 屏蔽assignment_graph图片生成
            # graph_plot_path = os.path.join(self.output_dir, f"assignment_graph_{file_suffix}.jpg")
            # graph_title = f'任务分配关系图\nModel: {model_name[:30]}...\nScenario: {scenario_name}'
            # self._plot_assignment_graph(final_plan, uavs, targets, graph_plot_path, graph_title)

        # 保存TXT格式的场景和结果报告，包含详细的方案分析
        scenario_txt_path = os.path.join(self.output_dir, f"scenario_report_{file_suffix}.txt")
        
        # 构建增强的报告内容，包含方案分析
        enhanced_report = report_content + f"""

方案详细分析:
{'='*50}
路径信息:
  - 总路径长度: {plan_analysis['total_path_length']:.1f}m
  - 平均路径长度: {plan_analysis['avg_path_length_per_uav']:.1f}m/UAV
  - 最长路径: {plan_analysis['max_path_length']:.1f}m
  - 最短路径: {plan_analysis['min_path_length']:.1f}m

资源利用:
  - 总资源消耗: {plan_analysis['total_resource_consumption']}
  - 资源利用率: {plan_analysis['resource_utilization_rate']:.1%}
  - 活跃UAV数量: {plan_analysis['active_uav_count']}/{len(eval_uavs)}
  - 空闲UAV数量: {plan_analysis['idle_uav_count']}/{len(eval_uavs)}

性能指标:
  - 纯推理时间: {pure_inference_time:.3f}s
  - 方案分析时间: {analysis_time:.3f}s
  - 完成率: {metrics.get('completion_rate', 0):.1%}
  - 资源利用率: {metrics.get('resource_utilization_rate', 0):.1%}

任务分布:
{chr(10).join([f"  - 目标{tid}: {count}个任务" for tid, count in plan_analysis['task_distribution'].items()])}
"""
        
        scenario_data = {
            'episode': 'Test', 'scenario_name': scenario_name, 'timestamp': timestamp,
            'uavs': eval_uavs, 'targets': eval_targets, 'obstacles': eval_obstacles,
            'uav_count': len(eval_uavs), 'target_count': len(eval_targets), 'obstacle_count': len(eval_obstacles),
            'config_info': {'obs_mode': 'graph'}, 'inference_report': enhanced_report,
            'plan_analysis': plan_analysis  # 添加详细分析数据
        }
        self._save_scenario_as_txt(scenario_data, scenario_txt_path)
        
        file_save_time = time.time() - file_save_start
        print(f"💾 文件保存完成，耗时: {file_save_time:.3f}s")
        
        # 将结果追加到CSV文件，包含增强的信息
        csv_row = {
            'timestamp': timestamp, 'model_name': model_name, 
            'inference_mode': 'ensemble_inference' if len(self.networks) > 1 else 'single_model_test',
            'scenario_name': scenario_name, 'num_uavs': len(eval_uavs), 'num_targets': len(eval_targets),
            'num_obstacles': len(eval_obstacles), 'resource_abundance': 1.2,
            'inference_time_s': round(pure_inference_time, 3),  # 使用纯推理时间，精确到毫秒
            'scenario_txt_path': os.path.basename(scenario_txt_path),
            'result_plot_path': os.path.basename(final_img_path),
            'graph_plot_path': 'disabled',  # assignment_graph已屏蔽
            'total_distance': round(plan_analysis['total_path_length'], 1),  # 添加总路径长度
            **{k: v for k, v in metrics.items() if k in self.csv_fieldnames}
        }
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(csv_row)

        # 【输出最终总结】
        total_time = time.time() - (inference_start_time - pure_inference_time)  # 从开始到现在的总时间
        print(f"\n🎉 场景 {scenario_name} 处理完成!")
        print(f"   🤖 模型: {model_name}")
        print(f"   ⏱️  纯推理时间: {pure_inference_time:.3f}s")
        print(f"   📊 方案分析时间: {analysis_time:.3f}s") 
        print(f"   💾 文件保存时间: {file_save_time:.3f}s")
        print(f"   🕐 总处理时间: {total_time:.3f}s")
        print(f"   🎯 任务完成率: {metrics.get('completion_rate', 0):.1%}")
        print(f"   🛣️  总路径长度: {plan_analysis['total_path_length']:.1f}m")
        print(f"{'='*80}")

    def run_suite(self, test_mode, num_random_scenarios, uav_range, target_range, step):
        """根据选择的模式，运行相应的测试套件"""
        if test_mode == 'incremental':
            self.run_incremental_tests(uav_range, target_range, step)
        elif test_mode == 'random':
            self.run_random_tests(num_random_scenarios)
        
        print("\n" + "="*60)
        print("所有测试已完成！")
        print(f"详细结果已保存至目录: {self.output_dir}")
        print(f"汇总数据请查看CSV文件: {self.csv_path}")
        print("="*60)

    def run_incremental_tests(self, uav_range: tuple, target_range: tuple, step: int = 1):
        """执行从最小到最大的增量测试"""
        print("\n" + "="*60 + f"\n启动增量测试模式\n" + "="*60)
        for num_uavs in range(uav_range[0], uav_range[1] + 1, step):
            for num_targets in range(target_range[0], target_range[1] + 1, step):
                if num_uavs > self.config.MAX_UAVS or num_targets > self.config.MAX_TARGETS:
                    continue
                scenario_name = f"{num_uavs}uav_{num_targets}tgt"
                num_obstacles = (num_uavs + num_targets) // 2
                uavs, targets, obstacles = generate_test_scenario(num_uavs, num_targets, num_obstacles, self.config)
                self._process_scenario(uavs, targets, obstacles, scenario_name)

    def run_random_tests(self, num_scenarios: int):
        """执行指定数量的随机场景测试"""
        print("\n" + "="*60 + f"\n启动随机测试模式 (数量: {num_scenarios})\n" + "="*60)
        
        # 【修复】使用合理的范围生成随机场景，避免超出模板限制
        for i in range(num_scenarios):
            # 使用更合理的范围，确保不会超出模板限制
            num_uavs = np.random.randint(3, min(25, self.config.MAX_UAVS) + 1)  # 3-25
            num_targets = np.random.randint(2, min(15, self.config.MAX_TARGETS) + 1)  # 2-15
            scenario_name = f"random{i+1}_{num_uavs}uav_{num_targets}tgt"
            num_obstacles = max(5, (num_uavs + num_targets) // 3)  # 确保有足够的障碍物
            
            print(f"🎲 生成随机场景 {i+1}/{num_scenarios}: {scenario_name}")
            uavs, targets, obstacles = generate_test_scenario(num_uavs, num_targets, num_obstacles, self.config)
            self._process_scenario(uavs, targets, obstacles, scenario_name)

def main():
    parser = argparse.ArgumentParser(description="多模型增量与随机测试套件")
    parser.add_argument('--models', nargs='+', required=True, help='一个或多个已训练模型的路径，用空格隔开')
    parser.add_argument('--output', type=str, default='output/test_suite_results', help='测试结果保存目录')
    parser.add_argument('--test-mode', type=str, choices=['incremental', 'random'], default='incremental', help="测试模式: 'incremental' (默认) 或 'random'")
    parser.add_argument('--num-random-scenarios', type=int, default=5, help="在 'random' 模式下，要生成的随机场景数量。")
    args = parser.parse_args()
    config = Config()
    
    # 增量测试的范围 (仅在 incremental 模式下生效)
    UAV_RANGE = (3, config.MAX_UAVS)
    TARGET_RANGE = (2, config.MAX_TARGETS)
    STEP = 2
    
    try:
        runner = ModelTestSuiteRunner(args.models, config, args.output)
        runner.run_suite(
            test_mode=args.test_mode,
            num_random_scenarios=args.num_random_scenarios,
            uav_range=UAV_RANGE,
            target_range=TARGET_RANGE,
            step=STEP
        )
    except Exception as e:
        print(f"\n测试过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()