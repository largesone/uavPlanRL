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
#
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
    根据指定的实体数量动态生成一个测试场景。
    该函数逻辑改编自项目中的 environment.py::_initialize_entities 方法，以确保场景的有效性。
    """
    uavs, targets, obstacles = [], [], []
    map_size = config.MAP_SIZE
    resource_dim = config.RESOURCE_DIM
    
    # 1. 生成目标
    target_positions = np.random.uniform(map_size * 0.2, map_size * 0.8, size=(num_targets, 2))
    for i in range(num_targets):
        resources = np.random.randint(50, 151, size=resource_dim)
        value = np.random.randint(80, 121)
        targets.append(Target(id=i + 1, position=target_positions[i], resources=resources, value=value))

    # 2. 生成无人机
    total_demand = np.sum([t.resources for t in targets], axis=0) if targets else np.zeros(resource_dim)
    resource_abundance = 1.2  # 在测试中固定资源富裕度为1.2倍，以控制变量
    total_supply = total_demand * resource_abundance
    
    uav_resources = np.zeros((num_uavs, resource_dim))
    if num_uavs > 0:
        avg_supply_per_uav = total_supply / num_uavs
        for i in range(num_uavs):
            uav_resources[i] = np.random.uniform(0.8, 1.2, size=resource_dim) * avg_supply_per_uav
    
    uav_positions = np.random.uniform(0, map_size, size=(num_uavs, 2))
    for i in range(num_uavs):
        uavs.append(UAV(
            id=i + 1, position=uav_positions[i], heading=np.random.uniform(0, 2 * np.pi),
            resources=uav_resources[i].astype(int), max_distance=config.UAV_MAX_DISTANCE,
            velocity_range=config.UAV_VELOCITY_RANGE, economic_speed=config.UAV_ECONOMIC_SPEED
        ))
        
    # 3. 生成障碍物
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

    def _save_scenario_as_txt(self, scenario_data: dict, filepath: str):
        """将场景数据详细信息保存为txt文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n" + "训练场景数据记录\n" + "=" * 80 + "\n")
                f.write(f"轮次: {scenario_data.get('episode', 'N/A')}\n")
                f.write(f"场景名称: {scenario_data.get('scenario_name', 'N/A')}\n")
                f.write(f"时间戳: {scenario_data.get('timestamp', 'N/A')}\n")
                f.write(f"UAV数量: {scenario_data.get('uav_count', 'N/A')}\n")
                f.write(f"目标数量: {scenario_data.get('target_count', 'N/A')}\n")
                f.write(f"障碍物数量: {scenario_data.get('obstacle_count', 'N/A')}\n\n")
                f.write("配置信息:\n" + "-" * 40 + "\n")
                f.write(f"网络类型: {self.config.NETWORK_TYPE}\n\n")
                f.write("UAV详细信息:\n" + "-" * 40 + "\n")
                for uav in scenario_data['uavs']:
                    f.write(f"  UAV {uav.id}: \n")
                    f.write(f"    初始位置: [{uav.position[0]:.2f}, {uav.position[1]:.2f}]\n")
                    f.write(f"    初始资源: {uav.initial_resources}\n\n")
                f.write("目标详细信息:\n" + "-" * 40 + "\n")
                for target in scenario_data['targets']:
                    f.write(f"  目标 {target.id}: \n")
                    f.write(f"    位置: [{target.position[0]:.2f}, {target.position[1]:.2f}]\n")
                    f.write(f"    初始需求资源: {target.resources}\n\n")
                f.write("障碍物详细信息:\n" + "-" * 40 + "\n")
                for i, obstacle in enumerate(scenario_data['obstacles']):
                    f.write(f"  障碍物 {i+1}: \n    类型: 圆形障碍物\n    中心: [{obstacle.center[0]:.2f}, {obstacle.center[1]:.2f}]\n    半径: {obstacle.radius:.2f}\n\n")
                f.write("场景统计信息:\n" + "-" * 40 + "\n")
                uav_total_res = np.sum([u.initial_resources for u in scenario_data['uavs']], axis=0)
                target_total_demand = np.sum([t.resources for t in scenario_data['targets']], axis=0)
                abundance = uav_total_res / (target_total_demand + 1e-6)
                obstacle_area = sum(np.pi * o.radius**2 for o in scenario_data['obstacles'])
                map_area = self.config.MAP_SIZE**2
                f.write(f"UAV初始总资源: {uav_total_res.astype(int)}\n")
                f.write(f"目标初始总需求: {target_total_demand.astype(int)}\n")
                f.write(f"资源充裕度: [{abundance[0]:.3f} {abundance[1]:.3f}]\n")
                f.write(f"障碍物覆盖率: {obstacle_area / map_area:.3f}\n")
                if scenario_data.get('inference_report'):
                    f.write("\n" + scenario_data['inference_report'])
                f.write("\n" + "=" * 80 + "\n" + "场景数据记录结束\n" + "=" * 80 + "\n")
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
            ax.set_xlabel('X 坐标 (m)'); ax.set_ylabel('Y 坐标 (m)')
            ax.legend(); ax.grid(True, linestyle='--', alpha=0.5); ax.set_aspect('equal', adjustable='box')
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"绘制任务分配关系图失败: {e}")

    def _process_scenario(self, uavs: list, targets: list, obstacles: list, scenario_name: str):
        """对单个生成好的场景，使用所有已加载的模型进行测试和保存"""
        # 判断是否使用集成推理（当有多个模型时）
        if len(self.networks) > 1:
            print(f"\n--- 执行集成推理 | 模型数量: {len(self.networks)} | 场景: {scenario_name} ---")
            
            # 执行集成推理
            start_time = time.time()
            results = self.evaluator._ensemble_inference(self.model_paths, uavs, targets, obstacles, scenario_name=scenario_name)
            inference_time = time.time() - start_time
            
            if not results:
                print("集成推理失败，跳过此场景测试。")
                return
                
            # 使用第一个模型的名称作为集成推理的标识
            model_name = "ensemble_" + "_".join([os.path.basename(path)[:10] for path in self.model_paths[:3]])
            if len(self.model_paths) > 3:
                model_name += f"_and_{len(self.model_paths)-3}_more"
        else:
            # 单模型推理逻辑保持不变
            model_path = list(self.networks.keys())[0]
            network = self.networks[model_path]
            model_name = os.path.basename(model_path)
            print(f"\n--- 测试模型: {model_name} | 场景: {scenario_name} ---")

            # 创建当前场景的环境
            graph = DirectedGraph(uavs, targets, self.config.GRAPH_N_PHI, obstacles, self.config)
            env = UAVTaskEnv(uavs, targets, graph, obstacles, self.config, obs_mode="graph")

            # 执行推理
            start_time = time.time()
            results = self.evaluator._run_inference(network, env, use_softmax_sampling=True, scenario_name=scenario_name)
            inference_time = time.time() - start_time
            
            if not results:
                print("推理失败，跳过此模型的本次测试。")
                return

            # 评估和保存结果
            final_plan = self.evaluator._build_plan_from_inference_results(results, uavs, targets)
            metrics = evaluate_plan(final_plan, uavs, targets, final_uav_states=results.get('final_uav_states'))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_suffix = f"{scenario_name}_{model_name[:20]}_{timestamp}"

            # 保存标准可视化图
            report_content, img_path = self.visualizer.save(
                final_plan, uavs, targets, obstacles, scenario_name=f"Test_{scenario_name}",
                training_time=0, plan_generation_time=inference_time,
                evaluation_metrics=metrics, suffix=f"_{model_name[:20]}_{timestamp}"
            )
            final_img_path = os.path.join(self.output_dir, os.path.basename(img_path))
            if os.path.exists(img_path): os.rename(img_path, final_img_path)

            # 保存新的任务分配关系图
            graph_plot_path = os.path.join(self.output_dir, f"assignment_graph_{file_suffix}.jpg")
            graph_title = f'任务分配关系图\nModel: {model_name[:30]}...\nScenario: {scenario_name}'
            self._plot_assignment_graph(final_plan, uavs, targets, graph_plot_path, graph_title)

            # 保存TXT格式的场景和结果报告
            scenario_txt_path = os.path.join(self.output_dir, f"scenario_report_{file_suffix}.txt")
            scenario_data = {
                'episode': 'Test', 'scenario_name': scenario_name, 'timestamp': timestamp,
                'uavs': uavs, 'targets': targets, 'obstacles': obstacles,
                'uav_count': len(uavs), 'target_count': len(targets), 'obstacle_count': len(obstacles),
                'config_info': {'obs_mode': 'graph'}, 'inference_report': report_content
            }
            self._save_scenario_as_txt(scenario_data, scenario_txt_path)
            
            # 将结果追加到CSV文件
            csv_row = {
                'timestamp': timestamp, 'model_name': model_name, 
                'inference_mode': 'ensemble_inference' if len(self.networks) > 1 else 'single_model_test',
                'scenario_name': scenario_name, 'num_uavs': len(uavs), 'num_targets': len(targets),
                'num_obstacles': len(obstacles), 'resource_abundance': 1.2,
                'inference_time_s': round(inference_time, 2),
                'scenario_txt_path': os.path.basename(scenario_txt_path),
                'result_plot_path': os.path.basename(final_img_path),
                'graph_plot_path': os.path.basename(graph_plot_path),
                **{k: v for k, v in metrics.items() if k in self.csv_fieldnames}
            }
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writerow(csv_row)

            if len(self.networks) > 1:
                print(f"集成推理 ({len(self.networks)}个模型) 在场景 {scenario_name} 的测试完成。")
            else:
                print(f"模型 {model_name} 在场景 {scenario_name} 的测试完成。")

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
        for i in range(num_scenarios):
            num_uavs = np.random.randint(2, self.config.MAX_UAVS + 1)
            num_targets = np.random.randint(2, self.config.MAX_TARGETS + 1)
            scenario_name = f"random{i+1}_{num_uavs}uav_{num_targets}tgt"
            num_obstacles = (num_uavs + num_targets) // 2
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