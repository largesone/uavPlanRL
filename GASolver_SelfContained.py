# 文件名: GASolver_SelfContained.py
# 描述: (临时修复版) 一个采用混合保真度模拟的、自包含的遗传算法求解器。

from typing import List, Dict, Tuple
import numpy as np
import time
from collections import defaultdict
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib

from entities import UAV, Target
from path_planning import PHCurveRRTPlanner, Obstacle
# [说明] 我们从main.py中导入了DirectedGraph，但根据您的要求，我们不会修改main.py中的定义
from main import Config, visualize_task_assignments, set_chinese_font, DirectedGraph
from scenarios import get_new_experimental_scenario,get_small_scenario,get_complex_scenario


class GeneticAlgorithm_SelfContained:
    # ... 此内部类的所有代码与上一版本完全相同，此处省略以保持简洁 ...
    def __init__(self, uavs: List[UAV], targets: List[Target], graph: DirectedGraph, config: Config,
                 population_size: int, crossover_rate: float, mutation_rate: float,
                 target_execution_sequence: List[int], max_uavs_per_target: int):
        self.uavs = uavs; self.targets = targets; self.graph = graph; self.config = config; self.population_size = population_size
        self.crossover_rate = crossover_rate; self.mutation_rate = mutation_rate; self.target_execution_sequence = target_execution_sequence
        self.max_uavs_per_target = max_uavs_per_target
        self.uav_map = {u.id: u for u in uavs}; self.target_map = {t.id: t for t in targets}
        self.population = self._create_initial_population(); self.fitness_cache = {}

    def _create_initial_population(self) -> List[np.ndarray]:
        population = []
        num_genes = len(self.targets) * self.max_uavs_per_target
        for _ in range(self.population_size):
            chromosome = np.zeros((3, num_genes), dtype=int)
            chromosome[0, :] = np.repeat(list(self.target_map.keys()), self.max_uavs_per_target)
            uav_assignment_options = list(self.uav_map.keys()) + [0]
            chromosome[1, :] = np.random.choice(uav_assignment_options, size=num_genes)
            chromosome[2, :] = np.random.randint(0, self.config.GRAPH_N_PHI, size=num_genes)
            population.append(chromosome)
        return population

    def evaluate_fitness(self, chromosome: np.ndarray) -> float:
        chrom_tuple = tuple(map(tuple, chromosome))
        if chrom_tuple in self.fitness_cache: return self.fitness_cache[chrom_tuple]
        uav_status = {u.id: {'last_vertex': (-u.id, None), 'free_at': 0.0} for u in self.uavs}
        uav_rem_resources = {u.id: u.resources.copy().astype(float) for u in self.uavs}
        target_rem_needs = {t.id: t.resources.copy().astype(float) for t in self.targets}
        max_completion_time = 0.0
        total_distance = 0.0
        for target_id in self.target_execution_sequence:
            gene_indices = np.where((chromosome[0, :] == target_id) & (chromosome[1, :] != 0))[0]
            if len(gene_indices) == 0: continue
            arrival_times = []
            uav_distances = {}
            assigned_uavs_for_target = np.unique(chromosome[1, gene_indices])
            for uav_id in assigned_uavs_for_target:
                uav = self.uav_map[uav_id]
                u_state = uav_status[uav_id]
                phi_idx = chromosome[2, np.where((chromosome[0, :] == target_id) & (chromosome[1, :] == uav_id))[0][0]]
                start_v = u_state['last_vertex']
                end_v = (target_id, self.graph.phi_set[phi_idx])
                dist = self.graph.adjacency_matrix[self.graph.vertex_to_idx[start_v], self.graph.vertex_to_idx[end_v]]
                if np.isinf(dist): return 0.0
                uav_distances[uav_id] = {'dist': dist, 'phi_idx': phi_idx}
                arrival_time = u_state['free_at'] + (dist / uav.economic_speed if uav.economic_speed > 0 else float('inf'))
                arrival_times.append(arrival_time)
            if not arrival_times: continue
            sync_time = max(arrival_times)
            max_completion_time = max(max_completion_time, sync_time)
            target_need = target_rem_needs[target_id]
            for uav_id in assigned_uavs_for_target:
                uav_res = uav_rem_resources[uav_id]
                contribution = np.minimum(uav_res, target_need)
                uav_rem_resources[uav_id] -= contribution
                target_need -= contribution
            target_rem_needs[target_id] = target_need
            for uav_id in assigned_uavs_for_target:
                u_state = uav_status[uav_id]
                dist_info = uav_distances[uav_id]
                u_state['last_vertex'] = (target_id, self.graph.phi_set[dist_info['phi_idx']])
                u_state['free_at'] = sync_time
                total_distance += dist_info['dist']
        total_unfulfilled_demand = sum(np.sum(needs) for needs in target_rem_needs.values())
        cost = max_completion_time + total_distance * 0.1 + total_unfulfilled_demand * 10000
        fitness = 1.0 / (cost + 1e-6)
        self.fitness_cache[chrom_tuple] = fitness
        return fitness
    
    def selection(self, fitnesses: List[float]) -> List[np.ndarray]:
        parents = [];
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            winner_idx = i if fitnesses[i] >= fitnesses[j] else j
            parents.append(self.population[winner_idx].copy())
        return parents

    def crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        offspring = []
        for i in range(0, self.population_size, 2):
            p1, p2 = parents[i], parents[i+1]
            if random.random() < self.crossover_rate:
                c1, c2 = p1.copy(), p2.copy()
                point = random.randint(1, c1.shape[1] - 1)
                c1[:, point:], c2[:, point:] = p2[:, point:], p1[:, point:]
                offspring.extend([c1, c2])
            else:
                offspring.extend([p1.copy(), p2.copy()])
        return offspring

    def mutation(self, offspring: List[np.ndarray]) -> List[np.ndarray]:
        for chrom in offspring:
            if random.random() < self.mutation_rate:
                gene_idx = random.randint(0, chrom.shape[1] - 1)
                chrom[1, gene_idx] = random.choice(list(self.uav_map.keys()) + [0])
                chrom[2, gene_idx] = random.randint(0, self.config.GRAPH_N_PHI - 1)
        return offspring

class GASolver_SelfContained:
    """(新对比算法) 使用混合保真度模拟的遗传算法求解器封装。"""
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles: List[Obstacle], config: Config):
        self.base_uavs = [UAV(u.id, u.position, u.heading, u.resources, u.max_distance, u.velocity_range, u.economic_speed) for u in uavs]
        self.base_targets = [Target(t.id, t.position, t.resources, t.value) for t in targets]
        self.obstacles = obstacles
        self.config = config
        
        # [临时修复] 为了避免与main.py中的旧版DirectedGraph定义冲突，此处调用时不传入obstacles。
        # 这将使程序能够运行，但同时也意味着GA的进化过程将无法感知障碍物。
        self.graph = DirectedGraph(self.base_uavs, self.base_targets, n_phi=config.GRAPH_N_PHI)
        
        self.ga_instance = None
        print("GASolver_SelfContained 已初始化 (采用混合保真度模拟策略)。")
        
    def solve(self) -> Tuple[Dict, float, float]:
        start_time = time.time()
        
        target_execution_sequence = sorted([t.id for t in self.base_targets])
        self.ga_instance = GeneticAlgorithm_SelfContained(
            self.base_uavs, self.base_targets, self.graph, self.config,
            population_size=100, crossover_rate=0.8, mutation_rate=0.2,
            target_execution_sequence=target_execution_sequence,
            max_uavs_per_target=len(self.base_uavs)
        )
        
        for _ in tqdm(range(50), desc="GA(内置约束版)进化中"):
            self.ga_instance.fitness_cache.clear()
            current_fitnesses = [self.ga_instance.evaluate_fitness(c) for c in self.ga_instance.population]
            parents = self.ga_instance.selection(current_fitnesses)
            offspring = self.ga_instance.crossover(parents)
            self.ga_instance.population = self.ga_instance.mutation(offspring)
        
        ga_run_time = time.time() - start_time
        final_fitnesses = [self.ga_instance.evaluate_fitness(c) for c in self.ga_instance.population]
        best_chromosome = self.ga_instance.population[np.argmax(final_fitnesses)]
        
        print("\n正在为GA最优解生成高保真度的最终方案...")
        reconstruct_start_time = time.time()
        final_plan = self._reconstruct_plan_high_fidelity(best_chromosome, target_execution_sequence)
        reconstruct_time = time.time() - reconstruct_start_time
        print(f"方案重构完成，耗时: {reconstruct_time:.2f}s")

        return final_plan, ga_run_time, reconstruct_time

    def _reconstruct_plan_high_fidelity(self, chromosome: np.ndarray, sequence: List[int]) -> Dict:
        # (此函数逻辑保持不变)
        final_plan = defaultdict(list)
        target_map = {t.id: t for t in self.base_targets}
        uav_map = {u.id: u for u in self.base_uavs}
        uav_status = {u.id: {'pos': u.position.copy(), 'free_at': 0.0, 'heading': u.heading} for u in self.base_uavs}
        task_step_counter = defaultdict(lambda: 1)
        
        uav_rem_resources = {u.id: u.resources.copy().astype(float) for u in self.base_uavs}
        target_rem_needs = {t.id: t.resources.copy().astype(float) for t in self.base_targets}

        for target_id in sequence:
            assigned_uavs = chromosome[1, (chromosome[0, :] == target_id) & (chromosome[1, :] != 0)]
            unique_uav_ids = np.unique(assigned_uavs)
            if len(unique_uav_ids) == 0: continue

            path_planners = {}
            for uav_id in unique_uav_ids:
                u_state = uav_status[uav_id]
                phi_idx = chromosome[2, np.where((chromosome[0, :] == target_id) & (chromosome[1, :] == uav_id))[0][0]]
                planner = PHCurveRRTPlanner(u_state['pos'], target_map[target_id].position, u_state['heading'], self.graph.phi_set[phi_idx], self.obstacles, self.config)
                plan_result = planner.plan()
                if plan_result:
                    path_planners[uav_id] = {'path_points': plan_result[0], 'distance': plan_result[1], 'phi_idx': phi_idx}

            if not path_planners: continue
            
            time_windows = []
            for uav_id, plan_data in path_planners.items():
                uav = uav_map[uav_id]
                t_min = uav_status[uav_id]['free_at'] + (plan_data['distance'] / uav.velocity_range[1])
                t_max = uav_status[uav_id]['free_at'] + (plan_data['distance'] / uav.velocity_range[0])
                time_windows.append({'uav_id': uav_id, 't_min': t_min, 't_max': t_max})
            
            if not time_windows: continue
            sync_start = max(tw['t_min'] for tw in time_windows)
            sync_end = min(tw['t_max'] for tw in time_windows)
            is_feasible = sync_start <= sync_end + 1e-6
            final_sync_time = sync_start

            target_need = target_rem_needs[target_id]
            contributions = {}
            for uav_id in path_planners.keys():
                uav_res = uav_rem_resources[uav_id]
                contribution = np.minimum(uav_res, target_need)
                contributions[uav_id] = contribution
                target_need -= contribution
            
            target_rem_needs[target_id] = target_need
            
            for uav_id, plan_data in path_planners.items():
                uav = uav_map[uav_id]
                u_state = uav_status[uav_id]
                travel_time = final_sync_time - u_state['free_at']
                speed = (plan_data['distance'] / travel_time) if travel_time > 1e-6 else uav.velocity_range[1]
                
                resource_cost = contributions.get(uav_id, np.zeros_like(uav.resources, dtype=float))
                uav_rem_resources[uav_id] -= resource_cost

                final_plan[uav_id].append({
                    'target_id': target_id, 'start_pos': u_state['pos'].copy(),
                    'speed': np.clip(speed, uav.velocity_range[0], uav.velocity_range[1]),
                    'arrival_time': final_sync_time, 'step': task_step_counter[uav_id],
                    'is_sync_feasible': is_feasible, 'phi_idx': plan_data['phi_idx'],
                    'path_points': plan_data['path_points'], 'distance': plan_data['distance'], 'resource_cost': resource_cost
                })
                
                task_step_counter[uav_id] += 1
                u_state['pos'] = target_map[target_id].position.copy()
                u_state['free_at'] = final_sync_time
                u_state['heading'] = self.graph.phi_set[plan_data['phi_idx']]
        
        return final_plan

if __name__ == '__main__':
    print("="*80)
    print(">>> 正在独立运行 GASolver_SelfContained (内置约束版) <<<")
    print("="*80)
    
    set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
    config = Config()
    # uavs, targets, obstacles = get_new_experimental_scenario(config.OBSTACLE_TOLERANCE)
    uavs, targets, obstacles = get_small_scenario(config.OBSTACLE_TOLERANCE)

    solver = GASolver_SelfContained(uavs, targets, obstacles, config)
    final_plan, ga_time, scheduling_time = solver.solve()
    
    visualize_task_assignments(
        final_plan, uavs, targets, obstacles, config, 
        scenario_name="新实验场景_GA(内置约束版)测试", 
        training_time=ga_time, plan_generation_time=scheduling_time,
        deadlocked_tasks=None
    )
    print("\n===== [GASolver_SelfContained 独立测试运行结束] =====")



