# -*- coding: utf-8 -*-
# 文件名: PSOSolver.py
# 描述: 基于Zhang et al. 2023论文的改进多目标PSO算法实现

import numpy as np
import time
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from entities import UAV, Target
from path_planning import PHCurveRRTPlanner, Obstacle
from main import Config, DirectedGraph
from evaluate import evaluate_plan

class Particle:
    """粒子类，表示一个解决方案"""
    def __init__(self, dimension, uavs, targets, n_phi):
        self.dimension = dimension
        self.uavs = uavs
        self.targets = targets
        self.n_phi = n_phi
        
        # 位置和速度初始化
        self.position = self._initialize_position()
        self.velocity = self._initialize_velocity()
        
        # 个体最优
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('-inf')
        
        # 适应度历史
        self.fitness_history = []
    
    def _initialize_position(self):
        """初始化粒子位置"""
        # 编码格式: [target_id, uav_id, phi_idx, resource_ratio]
        position = np.zeros(self.dimension, dtype=float)  # 改为float类型
        
        # 目标ID (0到targets数量-1)
        target_ids = list(range(len(self.targets)))
        for i in range(0, self.dimension, 4):
            position[i] = np.random.choice(target_ids)
        
        # UAV ID (0到uavs数量-1)
        uav_ids = list(range(len(self.uavs)))
        for i in range(1, self.dimension, 4):
            position[i] = np.random.choice(uav_ids)
        
        # 角度索引 (0到n_phi-1)
        for i in range(2, self.dimension, 4):
            position[i] = np.random.randint(0, self.n_phi)
        
        # 资源分配比例 (0到1)
        for i in range(3, self.dimension, 4):
            position[i] = np.random.uniform(0.1, 1.0)
            
        return position
    
    def _initialize_velocity(self):
        """初始化粒子速度"""
        velocity = np.zeros(self.dimension)
        
        # 目标ID速度
        for i in range(0, self.dimension, 4):
            velocity[i] = np.random.uniform(-0.5, 0.5)
        
        # UAV ID速度
        for i in range(1, self.dimension, 4):
            velocity[i] = np.random.uniform(-0.5, 0.5)
        
        # 角度索引速度
        for i in range(2, self.dimension, 4):
            velocity[i] = np.random.uniform(-0.5, 0.5)
        
        # 资源分配比例速度
        for i in range(3, self.dimension, 4):
            velocity[i] = np.random.uniform(-0.1, 0.1)
            
        return velocity
    
    def update_position(self):
        """更新粒子位置"""
        # 确保position是float类型
        self.position = self.position.astype(float)
        self.position += self.velocity
        
        # 边界约束
        self.position = np.clip(self.position, 0, None)
        
        # 离散化约束
        for i in range(0, self.dimension, 4):
            # 目标ID约束
            self.position[i] = int(np.clip(self.position[i], 0, len(self.targets) - 1))
        
        for i in range(1, self.dimension, 4):
            # UAV ID约束
            self.position[i] = int(np.clip(self.position[i], 0, len(self.uavs) - 1))
        
        for i in range(2, self.dimension, 4):
            # 角度索引约束
            self.position[i] = int(np.clip(self.position[i], 0, self.n_phi - 1))
        
        for i in range(3, self.dimension, 4):
            # 资源分配比例约束
            self.position[i] = np.clip(self.position[i], 0.1, 1.0)
    
    def update_velocity(self, gbest_position, w, c1, c2):
        """更新粒子速度"""
        r1, r2 = np.random.rand(2)
        
        self.velocity = (w * self.velocity + 
                        c1 * r1 * (self.pbest_position - self.position) +
                        c2 * r2 * (gbest_position - self.position))

class ImprovedPSOSolver:
    """改进的多目标PSO算法求解器"""
    
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles: List[Obstacle], config: Config):
        self.uavs = uavs
        self.targets = targets
        self.obstacles = obstacles
        self.config = config
        self.graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI)
        
        # PSO参数
        self.population_size = 100
        self.max_iterations = 200
        self.dimension = len(targets) * len(uavs) * 4  # 每个分配包含4个参数
        
        # 自适应参数
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 0.5
        self.c2_max = 2.5
        self.c2_min = 0.5
        
        # 区域分割参数
        self.area_division_enabled = True
        self.division_threshold = 0.1
        
        # 多目标权重
        self.objective_weights = {
            'completion_rate': 0.4,
            'resource_utilization': 0.3,
            'distance': 0.2,
            'load_balance': 0.1
        }
        
        print("ImprovedPSOSolver 已初始化 (基于Zhang et al. 2023论文)")
    
    def _decode_particle(self, particle: Particle) -> Dict:
        """将粒子位置解码为任务分配方案"""
        task_assignments = defaultdict(list)
        
        for i in range(0, particle.dimension, 4):
            target_id = int(particle.position[i])
            uav_id = int(particle.position[i + 1])
            phi_idx = int(particle.position[i + 2])
            resource_ratio = particle.position[i + 3]
            
            # 获取目标ID
            target = self.targets[target_id]
            
            # 添加到任务分配
            task_assignments[uav_id].append((target.id, phi_idx, resource_ratio))
        
        return dict(task_assignments)
    
    def _evaluate_fitness(self, particle: Particle) -> float:
        """评估粒子适应度（多目标综合）"""
        task_assignments = self._decode_particle(particle)
        
        # 计算各项指标
        metrics = self._calculate_objectives(task_assignments)
        
        # 综合适应度
        fitness = (self.objective_weights['completion_rate'] * metrics['completion_rate'] +
                  self.objective_weights['resource_utilization'] * metrics['resource_utilization'] +
                  self.objective_weights['distance'] * (1 - metrics['normalized_distance']) +
                  self.objective_weights['load_balance'] * metrics['load_balance'])
        
        return fitness
    
    def _calculate_objectives(self, task_assignments: Dict) -> Dict:
        """计算多目标指标"""
        if not task_assignments:
            return {
                'completion_rate': 0.0,
                'resource_utilization': 0.0,
                'normalized_distance': 1.0,
                'load_balance': 0.0
            }
        
        # 初始化状态
        uav_states = {u.id: {
            'rem_res': u.initial_resources.copy().astype(float),
            'last_vertex': (-u.id, None),
            'total_distance': 0.0
        } for u in self.uavs}
        
        target_rem_needs = {t.id: t.resources.copy().astype(float) for t in self.targets}
        
        # 执行任务分配
        total_distance = 0.0
        total_contribution = np.zeros_like(self.targets[0].resources)
        uav_contributions = []
        
        for uav_id, assignments in task_assignments.items():
            uav = next((u for u in self.uavs if u.id == uav_id), None)
            if not uav:
                continue
                
            uav_contribution = np.zeros_like(uav.initial_resources)
            
            for target_id, phi_idx, resource_ratio in assignments:
                target = next((t for t in self.targets if t.id == target_id), None)
                if not target:
                    continue
                
                # 计算距离
                start_v = uav_states[uav_id]['last_vertex']
                end_v = (target_id, self.graph.phi_set[phi_idx])
                dist = self.graph.adjacency_matrix[self.graph.vertex_to_idx[start_v], 
                                                 self.graph.vertex_to_idx[end_v]]
                
                # 计算资源贡献
                available_res = uav_states[uav_id]['rem_res']
                needed_res = target_rem_needs[target_id]
                contribution = np.minimum(available_res * resource_ratio, needed_res)
                
                # 更新状态
                uav_states[uav_id]['rem_res'] -= contribution
                target_rem_needs[target_id] -= contribution
                uav_states[uav_id]['last_vertex'] = end_v
                uav_states[uav_id]['total_distance'] += dist
                total_distance += dist
                total_contribution += contribution
                uav_contribution += contribution
            
            uav_contributions.append(np.sum(uav_contribution))
        
        # 计算指标 - 修复完成率计算
        total_demand = np.sum([t.resources for t in self.targets], axis=0)
        total_demand_safe = np.maximum(total_demand, 1e-6)
        
        # 完成率：实际贡献与需求的比值，按资源类型分别计算后取平均
        completion_rates = np.minimum(total_contribution, total_demand) / total_demand_safe
        completion_rate = np.mean(completion_rates)
        
        # 资源利用率：实际消耗与初始资源的比值
        total_initial_resources = np.sum([u.initial_resources for u in self.uavs], axis=0)
        total_initial_safe = np.maximum(total_initial_resources, 1e-6)
        
        total_consumed = np.sum([u.initial_resources - uav_states[u.id]['rem_res'] for u in self.uavs], axis=0)
        resource_utilization_rates = total_consumed / total_initial_safe
        resource_utilization = np.mean(resource_utilization_rates)
        
        # 确保数值不超过100%
        completion_rate = min(completion_rate, 1.0)
        resource_utilization = min(resource_utilization, 1.0)
        
        # 归一化距离
        max_possible_distance = sum(u.max_distance for u in self.uavs)
        normalized_distance = total_distance / max_possible_distance if max_possible_distance > 0 else 1.0
        
        # 负载均衡
        if uav_contributions:
            load_balance = 1.0 / (1.0 + np.std(uav_contributions))
        else:
            load_balance = 0.0
        
        return {
            'completion_rate': completion_rate,
            'resource_utilization': resource_utilization,
            'normalized_distance': normalized_distance,
            'load_balance': load_balance
        }
    
    def _area_division(self, particles: List[Particle]) -> List[List[Particle]]:
        """区域分割策略"""
        if not self.area_division_enabled:
            return [particles]
        
        # 计算粒子间的距离
        positions = np.array([p.position for p in particles])
        distances = np.zeros((len(particles), len(particles)))
        
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = distances[j, i] = dist
        
        # 基于距离聚类
        clusters = []
        used = set()
        
        for i in range(len(particles)):
            if i in used:
                continue
                
            cluster = [particles[i]]
            used.add(i)
            
            for j in range(i + 1, len(particles)):
                if j not in used and distances[i, j] < self.division_threshold:
                    cluster.append(particles[j])
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _adaptive_parameters(self, iteration: int, max_iterations: int) -> Tuple[float, float, float]:
        """自适应参数调整"""
        progress = iteration / max_iterations
        
        # 惯性权重线性递减
        w = self.w_max - (self.w_max - self.w_min) * progress
        
        # 学习因子自适应调整
        if progress < 0.5:
            # 前期：高探索
            c1 = self.c1_max - (self.c1_max - self.c1_min) * (progress * 2)
            c2 = self.c2_min + (self.c2_max - self.c2_min) * (progress * 2)
        else:
            # 后期：高开发
            c1 = self.c1_min + (self.c1_max - self.c1_min) * ((progress - 0.5) * 2)
            c2 = self.c2_max - (self.c2_max - self.c2_min) * ((progress - 0.5) * 2)
        
        return w, c1, c2
    
    def solve(self) -> Tuple[Dict, float, float]:
        """执行改进的PSO算法"""
        start_time = time.time()
        
        # 初始化粒子群
        particles = [Particle(self.dimension, self.uavs, self.targets, self.config.GRAPH_N_PHI) 
                   for _ in range(self.population_size)]
        
        # 全局最优
        gbest_position = particles[0].position.copy()
        gbest_fitness = float('-inf')
        
        # 收敛历史
        convergence_history = []
        
        print("开始PSO优化...")
        
        for iteration in tqdm(range(self.max_iterations), desc="PSO优化"):
            # 自适应参数调整
            w, c1, c2 = self._adaptive_parameters(iteration, self.max_iterations)
            
            # 区域分割
            clusters = self._area_division(particles)
            
            # 更新每个区域内的粒子
            for cluster in clusters:
                # 找到区域内的最优粒子
                cluster_best = max(cluster, key=lambda p: p.pbest_fitness)
                
                for particle in cluster:
                    # 评估适应度
                    fitness = self._evaluate_fitness(particle)
                    particle.fitness_history.append(fitness)
                    
                    # 更新个体最优
                    if fitness > particle.pbest_fitness:
                        particle.pbest_position = particle.position.copy()
                        particle.pbest_fitness = fitness
                    
                    # 更新全局最优
                    if fitness > gbest_fitness:
                        gbest_position = particle.position.copy()
                        gbest_fitness = fitness
                    
                    # 更新速度和位置
                    particle.update_velocity(gbest_position, w, c1, c2)
                    particle.update_position()
            
            # 记录收敛历史
            convergence_history.append(gbest_fitness)
            
            # 早停检测
            if len(convergence_history) > 20:
                recent_improvement = max(convergence_history[-20:]) - min(convergence_history[-20:])
                if recent_improvement < 0.001:
                    print(f"PSO早停于第{iteration}轮")
                    break
        
        # 生成最终方案
        best_particle = Particle(self.dimension, self.uavs, self.targets, self.config.GRAPH_N_PHI)
        best_particle.position = gbest_position
        final_assignments = self._decode_particle(best_particle)
        
        # 转换为标准格式
        final_plan = self._convert_to_standard_format(final_assignments)
        
        pso_time = time.time() - start_time
        
        # 绘制收敛曲线
        self._plot_convergence(convergence_history)
        
        return final_plan, pso_time, 0.0
    
    def _convert_to_standard_format(self, assignments: Dict) -> Dict:
        """转换为标准任务分配格式"""
        final_plan = defaultdict(list)
        
        for uav_id, assignments_list in assignments.items():
            for target_id, phi_idx, resource_ratio in assignments_list:
                # 计算实际资源消耗
                uav = next((u for u in self.uavs if u.id == uav_id), None)
                target = next((t for t in self.targets if t.id == target_id), None)
                
                if uav and target:
                    # 修复：确保resource_cost不超过实际可用资源
                    available_resources = uav.initial_resources.copy()
                    target_needs = target.resources.copy()
                    resource_cost = np.minimum(available_resources * resource_ratio, target_needs)
                    
                    # 计算实际距离
                    distance = np.linalg.norm(uav.position - target.position)
                    
                    final_plan[uav_id].append({
                        'target_id': target_id,
                        'phi_idx': phi_idx,
                        'resource_cost': resource_cost,
                        'distance': distance,
                        'speed': uav.velocity_range[1],
                        'arrival_time': 0.0,
                        'step': len(final_plan[uav_id]) + 1,
                        'is_sync_feasible': True
                    })
        
        return dict(final_plan)
    
    def _plot_convergence(self, convergence_history: List[float]):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_history, 'b-', linewidth=2)
        plt.title('PSO算法收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('最优适应度')
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        output_dir = "output/images"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/pso_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PSO收敛曲线已保存至: {output_dir}/pso_convergence.png")

# 测试代码
if __name__ == '__main__':
    """独立测试入口"""
    from main import visualize_task_assignments, set_chinese_font
    from scenarios import get_small_scenario

    set_chinese_font()
    config = Config()
    
    print("="*80)
    print(">>> 正在独立运行 Particle Swarm Optimization (PSO) 求解器 <<<")
    print("="*80)

    uav_data, target_data, obstacle_data = get_small_scenario(config.OBSTACLE_TOLERANCE)
    
    pso_solver = ImprovedPSOSolver(uav_data, target_data, obstacle_data, config)
    final_plan_pso, pso_time, scheduling_time = pso_solver.solve()
    
    print(f"求解完成。PSO优化耗时: {pso_time:.2f}s, 同步规划耗时: {scheduling_time:.2f}s")
    
    visualize_task_assignments(final_plan_pso, uav_data, target_data, obstacle_data,
                               config, "小场景_PSO测试",
                               training_time=pso_time,
                               plan_generation_time=scheduling_time,
                               save_report=True,
                               deadlocked_tasks=None) 
    print("\n===== [PSOSolver 独立测试运行结束] =====")