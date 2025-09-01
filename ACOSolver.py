# -*- coding: utf-8 -*-
# 文件名: ACOSolver.py
# 描述: 基于Duan et al. 2022论文的改进ACO算法实现

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

class Ant:
    """蚂蚁类，表示一个解决方案"""
    def __init__(self, uavs, targets, n_phi):
        self.uavs = uavs
        self.targets = targets
        self.n_phi = n_phi
        self.tour = []
        self.fitness = 0.0
        
    def construct_solution(self, pheromone_matrix, heuristic_matrix, alpha, beta):
        """构建解决方案"""
        self.tour = []
        unvisited_targets = list(range(len(self.targets)))
        uav_states = {u.id: {'rem_res': u.initial_resources.copy().astype(float)} for u in self.uavs}
        
        while unvisited_targets:
            # 选择下一个目标
            target_idx = self._select_next_target(unvisited_targets, pheromone_matrix, heuristic_matrix, alpha, beta)
            if target_idx is None:
                break
                
            # 选择UAV和角度
            uav_id, phi_idx = self._select_uav_and_angle(target_idx, uav_states)
            if uav_id is None:
                unvisited_targets.remove(target_idx)
                continue
                
            # 添加到路径
            self.tour.append((target_idx, uav_id, phi_idx))
            unvisited_targets.remove(target_idx)
            
            # 更新UAV状态
            target = self.targets[target_idx]
            contribution = np.minimum(uav_states[uav_id]['rem_res'], target.resources)
            uav_states[uav_id]['rem_res'] -= contribution
    
    def _select_next_target(self, unvisited_targets, pheromone_matrix, heuristic_matrix, alpha, beta):
        """选择下一个目标"""
        if not unvisited_targets:
            return None
            
        # 计算选择概率
        probabilities = []
        for target_idx in unvisited_targets:
            # 计算该目标的信息素和启发式信息
            pheromone = np.mean(pheromone_matrix[target_idx, :, :])
            heuristic = np.mean(heuristic_matrix[target_idx, :, :])
            
            # 计算概率
            prob = (pheromone ** alpha) * (heuristic ** beta)
            probabilities.append(prob)
        
        # 归一化概率
        total_prob = sum(probabilities)
        if total_prob == 0:
            return np.random.choice(unvisited_targets)
            
        probabilities = np.array([p / total_prob for p in probabilities], dtype=float)
        
        # 确保概率和为1
        probabilities = probabilities / np.sum(probabilities)
        
        # 轮盘赌选择
        return np.random.choice(unvisited_targets, p=probabilities)
    
    def _select_uav_and_angle(self, target_idx, uav_states):
        """选择UAV和角度"""
        target = self.targets[target_idx]
        available_uavs = []
        
        for uav in self.uavs:
            if np.any(uav_states[uav.id]['rem_res'] > 0):
                # 检查资源匹配
                if np.any(np.minimum(uav_states[uav.id]['rem_res'], target.resources) > 0):
                    available_uavs.append(uav.id)
        
        if not available_uavs:
            return None, None
        
        # 随机选择UAV和角度
        uav_id = np.random.choice(available_uavs)
        phi_idx = np.random.randint(0, self.n_phi)
        
        return uav_id, phi_idx
    
    def calculate_fitness(self, graph):
        """计算适应度"""
        if not self.tour:
            return 0.0
        
        total_distance = 0.0
        total_contribution = np.zeros_like(self.targets[0].resources)
        
        uav_states = {u.id: {'rem_res': u.initial_resources.copy().astype(float)} for u in self.uavs}
        
        for target_idx, uav_id, phi_idx in self.tour:
            target = self.targets[target_idx]
            
            # 计算距离（简化）
            uav = next((u for u in self.uavs if u.id == uav_id), None)
            if uav:
                distance = np.linalg.norm(uav.position - target.position)
                total_distance += distance
            
            # 计算资源贡献
            contribution = np.minimum(uav_states[uav_id]['rem_res'], target.resources)
            total_contribution += contribution
            uav_states[uav_id]['rem_res'] -= contribution
        
        # 计算适应度 - 修复完成率计算
        total_demand = np.sum([t.resources for t in self.targets], axis=0)
        total_demand_safe = np.maximum(total_demand, 1e-6)
        
        # 完成率：实际贡献与需求的比值，按资源类型分别计算后取平均
        completion_rates = np.minimum(total_contribution, total_demand) / total_demand_safe
        completion_rate = float(np.mean(completion_rates))
        
        # 确保完成率不超过100%
        completion_rate = min(completion_rate, 1.0)
        
        # 距离惩罚
        distance_penalty = total_distance * 0.001
        
        return float(completion_rate - distance_penalty)

class ImprovedACOSolver:
    """改进的ACO算法求解器"""
    
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles: List[Obstacle], config: Config):
        self.uavs = uavs
        self.targets = targets
        self.obstacles = obstacles
        self.config = config
        self.graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI)
        
        # ACO参数
        self.n_ants = 50
        self.max_iterations = 100
        self.alpha = 1.0  # 信息素重要程度
        self.beta = 2.0   # 启发式信息重要程度
        self.rho = 0.1    # 信息素挥发率
        self.Q = 1.0      # 信息素增量常数
        
        # 自适应参数
        self.adaptive_enabled = True
        self.diversity_threshold = 0.1
        
        # 信息素矩阵 [targets, uavs, angles]
        self.pheromone_matrix = np.ones((len(targets), len(uavs), config.GRAPH_N_PHI))
        
        # 启发式信息矩阵
        self.heuristic_matrix = self._initialize_heuristic_matrix()
        
        print("ImprovedACOSolver 已初始化 (基于Duan et al. 2022论文)")
    
    def _initialize_heuristic_matrix(self):
        """初始化启发式信息矩阵"""
        heuristic_matrix = np.zeros((len(self.targets), len(self.uavs), self.config.GRAPH_N_PHI))
        
        for i, target in enumerate(self.targets):
            for j, uav in enumerate(self.uavs):
                for k in range(self.config.GRAPH_N_PHI):
                    # 距离启发式
                    distance = np.linalg.norm(uav.position - target.position)
                    distance_heuristic = 1.0 / (1.0 + distance)
                    
                    # 资源匹配启发式
                    resource_match = np.sum(np.minimum(uav.resources, target.resources))
                    resource_heuristic = resource_match / (np.sum(uav.resources) + np.sum(target.resources))
                    
                    # 综合启发式
                    heuristic_matrix[i, j, k] = distance_heuristic * 0.7 + resource_heuristic * 0.3
        
        return heuristic_matrix
    
    def _update_pheromone(self, ants: List[Ant]):
        """更新信息素"""
        # 信息素挥发
        self.pheromone_matrix *= (1 - self.rho)
        
        # 信息素沉积
        for ant in ants:
            if ant.fitness > 0:
                delta_pheromone = self.Q / ant.fitness
                
                for target_idx, uav_id, phi_idx in ant.tour:
                    uav_idx = next(i for i, u in enumerate(self.uavs) if u.id == uav_id)
                    self.pheromone_matrix[target_idx, uav_idx, phi_idx] += delta_pheromone
    
    def _adaptive_parameters(self, iteration: int, diversity: float):
        """自适应参数调整"""
        if not self.adaptive_enabled:
            return self.alpha, self.beta, self.rho
        
        progress = iteration / self.max_iterations
        
        # 自适应alpha和beta
        if diversity < self.diversity_threshold:
            # 多样性低，增加探索
            alpha = self.alpha * (1 + progress)
            beta = self.beta * (1 - progress * 0.5)
        else:
            # 多样性高，增加开发
            alpha = self.alpha * (1 - progress * 0.5)
            beta = self.beta * (1 + progress)
        
        # 自适应rho
        rho = self.rho * (1 + progress)
        
        return alpha, beta, rho
    
    def _calculate_diversity(self, ants: List[Ant]) -> float:
        """计算种群多样性"""
        if not ants:
            return 0.0
        
        # 计算解之间的平均距离
        tours = [ant.tour for ant in ants]
        distances = []
        
        for i in range(len(tours)):
            for j in range(i + 1, len(tours)):
                # 计算两个解的距离
                distance = self._calculate_tour_distance(tours[i], tours[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_tour_distance(self, tour1, tour2):
        """计算两个解之间的距离"""
        # 简化的距离计算
        common_elements = 0
        total_elements = max(len(tour1), len(tour2))
        
        for item1 in tour1:
            for item2 in tour2:
                if item1[0] == item2[0] and item1[1] == item2[1]:  # 相同的目标和UAV
                    common_elements += 1
                    break
        
        return 1.0 - (common_elements / total_elements) if total_elements > 0 else 1.0
    
    def solve(self) -> Tuple[Dict, float, float]:
        """执行改进的ACO算法"""
        start_time = time.time()
        
        best_ant = None
        best_fitness = float('-inf')
        convergence_history = []
        
        print("开始ACO优化...")
        
        for iteration in tqdm(range(self.max_iterations), desc="ACO优化"):
            # 生成蚂蚁
            ants = [Ant(self.uavs, self.targets, self.config.GRAPH_N_PHI) for _ in range(self.n_ants)]
            
            # 构建解决方案
            alpha, beta, rho = self._adaptive_parameters(iteration, 0.0)  # 初始多样性为0
            
            for ant in ants:
                ant.construct_solution(self.pheromone_matrix, self.heuristic_matrix, alpha, beta)
                ant.fitness = ant.calculate_fitness(self.graph)
            
            # 计算多样性
            diversity = self._calculate_diversity(ants)
            
            # 更新最优解
            for ant in ants:
                if ant.fitness > best_fitness:
                    best_fitness = ant.fitness
                    best_ant = ant
            
            # 更新信息素
            self._update_pheromone(ants)
            
            # 记录收敛历史
            convergence_history.append(best_fitness)
            
            # 早停检测
            if len(convergence_history) > 20:
                recent_improvement = max(convergence_history[-20:]) - min(convergence_history[-20:])
                if recent_improvement < 0.001:
                    print(f"ACO早停于第{iteration}轮")
                    break
        
        # 生成最终方案
        if best_ant:
            final_plan = self._convert_tour_to_plan(best_ant.tour)
        else:
            final_plan = {}
        
        aco_time = time.time() - start_time
        
        # 绘制收敛曲线
        self._plot_convergence(convergence_history)
        
        return final_plan, aco_time, 0.0
    
    def _convert_tour_to_plan(self, tour: List[Tuple]) -> Dict:
        """将蚂蚁路径转换为任务分配方案"""
        final_plan = defaultdict(list)
        
        for target_idx, uav_id, phi_idx in tour:
            target = self.targets[target_idx]
            uav = next((u for u in self.uavs if u.id == uav_id), None)
            
            if uav:
                # 修复：确保resource_cost不超过实际可用资源
                available_resources = uav.initial_resources.copy()
                target_needs = target.resources.copy()
                resource_cost = np.minimum(available_resources, target_needs)
                
                # 计算实际距离
                distance = np.linalg.norm(uav.position - target.position)
                
                final_plan[uav_id].append({
                    'target_id': target.id,
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
        plt.plot(convergence_history, 'g-', linewidth=2)
        plt.title('ACO算法收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('最优适应度')
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        output_dir = "output/images"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/aco_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ACO收敛曲线已保存至: {output_dir}/aco_convergence.png")

# 测试代码
if __name__ == "__main__":
    from scenarios import get_small_scenario
    
    config = Config()
    uav_data, target_data, obstacle_data = get_small_scenario(config.OBSTACLE_TOLERANCE)
    
    aco_solver = ImprovedACOSolver(uav_data, target_data, obstacle_data, config)
    final_plan_aco, aco_time, scheduling_time = aco_solver.solve()
    
    print(f"ACO算法完成，耗时: {aco_time:.2f}s")
    print(f"最终方案: {final_plan_aco}")