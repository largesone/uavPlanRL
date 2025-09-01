# GASolver.py
# calculate_economic_sync_speeds函数关键部分，调用了该规划函数，满足协同及死锁检测规避等约束
import time
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib
# --- 核心模块从项目源文件直接导入 ---
from entities import UAV, Target
from path_planning import Obstacle, PHCurveRRTPlanner
from scenarios import get_new_experimental_scenario
from main import Config, visualize_task_assignments, set_chinese_font, calculate_economic_sync_speeds, DirectedGraph

# ==============================================================================
# [调试] 调试专用版本的协同规划函数 (日志增强)
# ==============================================================================
# 在 GASolver.py 文件中, 找到并替换 debug_calculate_economic_sync_speeds 函数

# 在 GASolver.py 文件中, 找到并替换 debug_calculate_economic_sync_speeds 函数

def debug_calculate_economic_sync_speeds(task_assignments, uavs, targets, graph, obstacles, config) -> Tuple[defaultdict, dict]:
    """
    一个带有详细日志输出的 calculate_economic_sync_speeds 副本，用于诊断死循环或性能问题。
    """
    print("\n--- [DEBUG] 进入协同规划函数 ---")
    
    def _plan_single_leg(args):
        # 内部辅助函数，调用高精度规划器
        uav_id, start_pos, target_pos, start_heading, end_heading, obstacles, config = args
        planner = PHCurveRRTPlanner(start_pos, target_pos, start_heading, end_heading, obstacles, config)
        return uav_id, planner.plan()

    final_plan = defaultdict(list)
    uav_status = {u.id: {'pos': u.position, 'free_at': 0.0, 'heading': u.heading} for u in uavs}
    remaining_tasks = {uav_id: list(tasks) for uav_id, tasks in task_assignments.items()}
    task_step_counter = defaultdict(lambda: 1)
    
    max_iterations = len(uavs) * len(targets) * 2 + 5 
    iteration_count = 0

    while any(v for v in remaining_tasks.values()):
        if iteration_count >= max_iterations:
            print("\n--- [DEBUG] ！！！错误：协同规划循环次数过多，可能存在死锁，强制终止。！！！")
            print(f"  [DEBUG] 卡住时剩余的任务: {remaining_tasks}")
            break
        
        print(f"\n--- [DEBUG] 规划循环第 {iteration_count + 1} 次 ---")
        
        # --- [二次修订] 将复杂的f-string格式化拆分为两步，彻底解决语法问题 ---
        # 1. 先创建一个用于显示的状态字典
        status_to_print = {u_id: f"{s['free_at']:.2f}s" for u_id, s in uav_status.items()}
        # 2. 在f-string中直接打印这个简单的字典
        print(f"  [DEBUG] 当前无人机状态 (free_at): {status_to_print}")
        # --- 修订结束 ---
        
        print(f"  [DEBUG] 当前剩余任务: {remaining_tasks}")

        next_target_groups = defaultdict(list)
        for uav_id, tasks in remaining_tasks.items():
            if tasks:
                next_target_groups[tasks[0][0]].append({'uav_id': uav_id, 'phi_idx': tasks[0][1]})
        
        print(f"  [DEBUG] 本轮待处理的协同任务组: {dict(next_target_groups)}")

        if not next_target_groups:
            print("  [DEBUG] ！！！警告：无更多可处理的任务组，但仍有任务剩余，循环终止。！！！")
            break

        group_arrival_times = []
        for target_id, uav_infos in next_target_groups.items():
            target = next((t for t in targets if t.id == target_id), None);_plan_results={};time_windows=[]
            if not target:continue
            for uav_info in uav_infos:
                uav_id=uav_info['uav_id']; args=(uav_id,uav_status[uav_id]['pos'],target.position,uav_status[uav_id]['heading'],graph.phi_set[uav_info['phi_idx']],obstacles,config); _,pr=_plan_single_leg(args)
                if pr: _plan_results[uav_id]={'path_points':pr[0],'distance':pr[1]}
            for uav_info in uav_infos:
                uav_id=uav_info['uav_id']
                if uav_id not in _plan_results:continue
                uav=next((u for u in uavs if u.id==uav_id),None)
                if not uav:continue
                dist,free_at=_plan_results[uav_id]['distance'],uav_status[uav_id]['free_at'];t_min=free_at+(dist/uav.velocity_range[1]);t_max=free_at+(dist/uav.velocity_range[0]) if uav.velocity_range[0]>0 else float('inf');t_econ=free_at+(dist/uav.economic_speed)
                time_windows.append({'uav_id':uav_id,'phi_idx':uav_info['phi_idx'],'t_min':t_min,'t_max':t_max,'t_econ':t_econ})
            if not time_windows:continue
            sync_start=max(tw['t_min'] for tw in time_windows);sync_end=min(tw['t_max'] for tw in time_windows);is_feasible=sync_start<=sync_end+1e-6
            final_sync_time=np.clip(np.median([tw['t_econ'] for tw in time_windows]),sync_start,sync_end) if is_feasible else sync_start
            group_arrival_times.append({'target_id':target_id,'arrival_time':final_sync_time,'uav_infos':time_windows,'is_feasible':is_feasible,'path_planners':_plan_results})

        print(f"  [DEBUG] 计算出的各任务组时间窗口及可行性:")
        if not group_arrival_times: print("    - (无有效任务组)")
        for group in group_arrival_times:
            print(f"    - 目标 {group['target_id']}: 到达时间={group['arrival_time']:.2f}, 是否可行={group['is_feasible']}")

        if not group_arrival_times:
            print("  [DEBUG] ！！！警告：无法为任何任务组计算到达时间，循环终止。！！！")
            break
        
        next_event = min(group_arrival_times, key=lambda x: x['arrival_time'])
        print(f"  [DEBUG] 本轮选择执行的事件: 目标 {next_event['target_id']} 在 {next_event['arrival_time']:.2f}s 到达")
        
        target_pos = next(t.position for t in targets if t.id == next_event['target_id'])
        for uav_info in next_event['uav_infos']:
            uav_id = uav_info['uav_id']
            if uav_id not in next_event['path_planners']: continue
            uav, plan_data = next(u for u in uavs if u.id == uav_id), next_event['path_planners'][uav_id]
            travel_time = next_event['arrival_time'] - uav_status[uav_id]['free_at']
            speed = (plan_data['distance'] / travel_time) if travel_time > 1e-6 else uav.velocity_range[1]
            final_plan[uav_id].append({'target_id': next_event['target_id'], 'start_pos': uav_status[uav_id]['pos'], 'speed': np.clip(speed, uav.velocity_range[0], uav.velocity_range[1]), 'arrival_time': next_event['arrival_time'], 'step': task_step_counter[uav_id], 'is_sync_feasible': next_event['is_feasible'], 'phi_idx': uav_info['phi_idx'], 'path_points': plan_data['path_points'], 'distance': plan_data['distance']})
            task_step_counter[uav_id] += 1
            uav_status[uav_id].update(pos=target_pos, free_at=next_event['arrival_time'], heading=graph.phi_set[uav_info['phi_idx']])
            if remaining_tasks.get(uav_id): remaining_tasks[uav_id].pop(0)
        
        iteration_count += 1
    
    print("\n--- [DEBUG] 协同规划函数执行完毕 ---\n")
    return final_plan, remaining_tasks



class GeneticAlgorithm:
    def __init__(self, uavs: List[UAV], targets: List[Target], graph: DirectedGraph, config: Config,
                 population_size: int, crossover_rate: float, mutation_rate: float, reinforcement_rate: float,
                 target_execution_sequence: List[int], max_uavs_per_target: int):
        self.uavs = uavs; self.targets = targets; self.graph = graph; self.config = config; self.population_size = population_size
        self.crossover_rate = crossover_rate; self.mutation_rate = mutation_rate; self.reinforcement_rate = reinforcement_rate
        self.target_execution_sequence = target_execution_sequence; self.max_uavs_per_target = max_uavs_per_target
        self.uav_map = {uav.id: uav for uav in self.uavs}; self.target_map = {t.id: t for t in self.targets}
        self.uav_ids = list(self.uav_map.keys()); self.n_phi = self.config.GRAPH_N_PHI
        self.population = self._create_initial_population(); self.fitness_cache = {}

    def _create_initial_population(self) -> List[np.ndarray]:
        # ... 此函数无改动 ...
        population = []
        num_genes = len(self.targets) * self.max_uavs_per_target
        for _ in range(self.population_size):
            chromosome = np.zeros((3, num_genes), dtype=int)
            chromosome[0, :] = np.repeat(list(self.target_map.keys()), self.max_uavs_per_target)
            uav_assignment_options = self.uav_ids + [0]
            chromosome[1, :] = np.random.choice(uav_assignment_options, size=num_genes)
            chromosome[2, :] = np.random.randint(0, self.n_phi, size=num_genes)
            population.append(chromosome)
        return population

# 在 GASolver.py 文件中, 找到 GeneticAlgorithm 类, 并替换其中的 evaluate_fitness 函数

    def evaluate_fitness(self, chromosome: np.ndarray) -> float:
        """[重构] 使用近似图快速评估单个染色体的适应度。"""
        chrom_tuple = tuple(map(tuple, chromosome))
        if chrom_tuple in self.fitness_cache: return self.fitness_cache[chrom_tuple]

        uav_states = {u.id: {
            'last_vertex': (-u.id, None),
            'dist': 0.0,
            'rem_res': u.initial_resources.copy().astype(float)
        } for u in self.uavs}
        target_rem_needs = {t.id: t.resources.copy().astype(float) for t in self.targets}
        
        for target_id in self.target_execution_sequence:
            gene_indices = np.where((chromosome[0, :] == target_id) & (chromosome[1, :] != 0))[0]
            if len(gene_indices) == 0:
                continue
            coalition = defaultdict(list)
            for idx in gene_indices:
                uav_id, phi_idx = chromosome[1, idx], chromosome[2, idx]
                coalition[uav_id].append(phi_idx)
            for uav_id, phi_indices in coalition.items():
                phi_idx = phi_indices[0]
                u_state = uav_states[uav_id]
                start_v, end_v = u_state['last_vertex'], (target_id, self.graph.phi_set[phi_idx])
                dist = self.graph.adjacency_matrix[self.graph.vertex_to_idx[start_v], self.graph.vertex_to_idx[end_v]]
                u_state['dist'] += dist
                u_state['last_vertex'] = end_v
            needed_res = target_rem_needs[target_id]
            sorted_coalition = sorted(list(coalition.keys()), key=lambda uid: np.sum(uav_states[uid]['rem_res']), reverse=True)
            for uav_id in sorted_coalition:
                contribution = np.minimum(uav_states[uav_id]['rem_res'], needed_res)
                uav_states[uav_id]['rem_res'] -= contribution
                needed_res -= contribution
            target_rem_needs[target_id] = needed_res

        penalty = 0
        cumulative_dist = sum(s['dist'] for s in uav_states.values())
        for uav_id, uav in self.uav_map.items():
            if uav_states[uav_id]['dist'] > uav.max_distance:
                penalty += (uav_states[uav_id]['dist'] - uav.max_distance) * 10
        
        total_unfulfilled_demand = sum(np.sum(np.maximum(0, needs)) for needs in target_rem_needs.values())
        
        # --- [修订] 大幅增加对未满足资源需求的惩罚权重 ---
        # 将惩罚系数从 1000 提升至 50000，强制算法优先满足资源需求
        penalty += total_unfulfilled_demand * 50000
        # --- 修订结束 ---
        
        max_dist = max(s['dist'] for s in uav_states.values()) if uav_states else 0
        objective_j = cumulative_dist + 0.5 * max_dist
        
        fitness = 1.0 / (objective_j + penalty + 1e-6)
        self.fitness_cache[chrom_tuple] = fitness
        return fitness
    
    def selection(self, fitnesses: List[float]) -> List[np.ndarray]:
        # ... 此函数无改动 ...
        parents = [];
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            winner_idx = i if fitnesses[i] >= fitnesses[j] else j
            parents.append(self.population[winner_idx].copy())
        return parents

    def crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        # ... 此函数无改动 ...
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
        # ... 此函数无改动 ...
        for chrom in offspring:
            if random.random() < self.mutation_rate:
                gene_idx = random.randint(0, chrom.shape[1] - 1)
                chrom[1, gene_idx] = random.choice(self.uav_ids + [0])
                chrom[2, gene_idx] = random.randint(0, self.n_phi - 1)
        return offspring
    
    # --- [修订] 重构并启用启发式增强算子，提升解的质量 ---
   # 在 GASolver.py 文件中, 找到 GeneticAlgorithm 类, 并替换其中的 heuristic_reinforcement_operator 函数

    def heuristic_reinforcement_operator(self, offspring: List[np.ndarray]) -> List[np.ndarray]:
        """[重构] 启发式增强算子，修复了变量名和内部模拟逻辑的错误。"""
        for chrom in offspring:
            if random.random() < self.reinforcement_rate:
                # 1. 诊断：找出此染色体方案中的闲置UAV
                all_assigned_uavs = set(chrom[1, chrom[1, :] != 0])
                idle_uavs = [uid for uid in self.uav_ids if uid not in all_assigned_uavs]
                if not idle_uavs:
                    continue

                # 2. 诊断：通过正确的、序列化的模拟，找出未满足的目标
                #    此模拟过程现在与 evaluate_fitness 中的逻辑完全对齐
                target_rem_needs_sim = {t.id: t.resources.copy().astype(float) for t in self.targets}
                uav_rem_res_sim = {u.id: u.initial_resources.copy().astype(float) for u in self.uavs}

                for target_id in self.target_execution_sequence:
                    # [修复] 将所有 'chromosome' 更正为 'chrom'
                    coalition_uids = chrom[1, (chrom[0, :] == target_id) & (chrom[1, :] != 0)]
                    if len(coalition_uids) == 0:
                        continue
                    
                    # [修复] 模拟资源消耗时，使用在循环中不断更新的 uav_rem_res_sim
                    needed_res = target_rem_needs_sim[target_id]
                    sorted_coalition = sorted(list(set(coalition_uids)), key=lambda uid: np.sum(uav_rem_res_sim[uid]), reverse=True)
                    
                    for uav_id in sorted_coalition:
                        contribution = np.minimum(uav_rem_res_sim[uav_id], needed_res)
                        uav_rem_res_sim[uav_id] -= contribution
                        needed_res -= contribution
                    target_rem_needs_sim[target_id] = needed_res
                
                unsatisfied_targets = {tid: np.sum(needs) for tid, needs in target_rem_needs_sim.items() if np.any(needs > 1e-5)}
                if not unsatisfied_targets:
                    continue

                # 3. 修正：将一个闲置UAV分配给需求最大的、且有空槽的目标
                target_to_fix = max(unsatisfied_targets, key=unsatisfied_targets.get)
                uav_to_assign = random.choice(idle_uavs)
                
                target_indices = np.where(chrom[0, :] == target_to_fix)[0]
                empty_slot_indices = [idx for idx in target_indices if chrom[1, idx] == 0]

                if empty_slot_indices:
                    gene_idx_to_modify = random.choice(empty_slot_indices)
                    chrom[1, gene_idx_to_modify] = uav_to_assign
                    # 随机分配一个角度
                    chrom[2, gene_idx_to_modify] = random.randint(0, self.n_phi - 1)
        return offspring

# 在 GASolver.py 文件中, 找到并替换整个 GASolver 类

class GASolver:
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles: List[Obstacle], config: Config):
        self.uavs = uavs
        self.targets = targets
        self.obstacles = obstacles
        self.config = config
        self.ga_instance = None
        self.graph = DirectedGraph(self.uavs, self.targets, n_phi=self.config.GRAPH_N_PHI)
        print("GASolver 已初始化 (采用优化的快速评估策略)。")
    
    def solve(self) -> Tuple[Dict, float, float]:
        start_time = time.time()
        print("GASolver 正在进行遗传优化...")

        target_execution_sequence = self._determine_target_execution_sequence()
        print(f"全局任务序列确定为: {target_execution_sequence}")

        self.ga_instance = GeneticAlgorithm(
            self.uavs, self.targets, self.graph, self.config,
            population_size=200,
            crossover_rate=0.8,
            mutation_rate=0.1,
            reinforcement_rate=0.4,
            target_execution_sequence=target_execution_sequence,
            max_uavs_per_target=len(self.uavs) 
        )
        
        best_fitness_history = []
        max_generations = 200

        for generation in tqdm(range(max_generations), desc="遗传算法进化中"):
            self.ga_instance.fitness_cache.clear()
            current_fitnesses = [self.ga_instance.evaluate_fitness(chrom) for chrom in self.ga_instance.population]
            best_index = np.argmax(current_fitnesses)
            best_fitness_history.append(current_fitnesses[best_index])
            elite_chromosome = self.ga_instance.population[best_index].copy()
            parents = self.ga_instance.selection(current_fitnesses)
            offspring = self.ga_instance.crossover(parents)
            offspring = self.ga_instance.mutation(offspring)
            offspring = self.ga_instance.heuristic_reinforcement_operator(offspring)
            self.ga_instance.fitness_cache.clear()
            offspring_fitnesses = [self.ga_instance.evaluate_fitness(c) for c in offspring]
            worst_index = np.argmin(offspring_fitnesses)
            offspring[worst_index] = elite_chromosome
            self.ga_instance.population = offspring

        ga_run_time = time.time() - start_time
        print(f"GA 优化完成，耗时: {ga_run_time:.2f}s")

        final_fitnesses = [self.ga_instance.evaluate_fitness(c) for c in self.ga_instance.population]
        best_chromosome = self.ga_instance.population[np.argmax(final_fitnesses)]
        
        task_assignments = defaultdict(list)
        for i in range(best_chromosome.shape[1]):
            target_id, uav_id, phi_idx = best_chromosome[:, i]
            if uav_id != 0:
                task_assignments[uav_id].append((target_id, phi_idx))
        
        # --- [最终修订] 增加逻辑来过滤连续的相同目标 ---
        final_task_assignments = defaultdict(list)
        for uav_id, tasks in task_assignments.items():
            # 1. 首先，移除 (target_id, phi_idx) 完全相同的重复任务
            unique_tasks = sorted(list(set(tasks)), key=lambda t: target_execution_sequence.index(t[0]))
            
            if not unique_tasks:
                continue

            # 2. 然后，过滤掉连续访问同一目标的任务
            #    例如，将 [(2,3), (2,2)] 修正为 [(2,3)]
            filtered_tasks = [unique_tasks[0]] # 先把第一个任务放进去
            for i in range(1, len(unique_tasks)):
                # 只有当当前任务的目标ID与上一个任务的目标ID不同时，才添加它
                if unique_tasks[i][0] != filtered_tasks[-1][0]:
                    filtered_tasks.append(unique_tasks[i])
            
            final_task_assignments[uav_id] = filtered_tasks
        # --- 修订结束 ---

        print("\n正在为GA最优解生成最终的精确协同计划...")
        print(f"  [INFO] 用于规划的最终任务序列 (已去重并过滤): {dict(final_task_assignments)}")
        post_process_start_time = time.time()
        
        # [恢复] 调用 main.py 中的官方函数，不再使用调试版
        final_plan, deadlocked_tasks = calculate_economic_sync_speeds(final_task_assignments, self.uavs, self.targets, self.graph, self.obstacles, self.config)
        planning_time = time.time() - post_process_start_time
        
        self._visualize_fitness(best_fitness_history)
        
        if deadlocked_tasks and any(deadlocked_tasks.values()):
             print("\n警告: 检测到死锁，以下无人机未能完成所有任务:", {k:v for k,v in deadlocked_tasks.items() if v})

        return final_plan, ga_run_time, planning_time

    def _determine_target_execution_sequence(self) -> List[int]:
        # ... 此函数无改动 ...
        utilities, beta = {}, 0.01
        for target in self.targets:
            sum_dist = sum(np.linalg.norm(uav.position - target.position) for uav in self.uavs)
            utilities[target.id] = target.value - beta * sum_dist
        return sorted(utilities.keys(), key=lambda t_id: utilities[t_id], reverse=True)

    def _visualize_fitness(self, history: List[float]):
        """可视化适应度进化过程"""
        # --- [修订] 在绘图前强制使用非交互式后端，防止弹窗 ---
        matplotlib.use('Agg')
        # --- 修订结束 ---
        
        plt.figure(figsize=(12, 6))
        plt.plot(history, marker='o', linestyle='-', color='b', markersize=4)
        plt.title('遗传算法适应度进化曲线 (已优化)', fontsize=16)
        plt.xlabel('进化代数 (Generation)', fontsize=12)
        plt.ylabel('种群最优适应度 (Best Fitness)', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        output_dir = "output/images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        img_filepath = os.path.join(output_dir, f"GA_Fitness_Convergence_{timestamp}.png")
        plt.savefig(img_filepath)
        print(f"适应度进化曲线图已保存至: {img_filepath}")
        plt.close()



if __name__ == '__main__':
    # ... 主函数入口无改动 ...
    set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
    print("="*80)
    print(">>> 正在独立运行 Genetic Algorithm (GA) 对比算法 (最终调试版) <<<")
    print("="*80)
    config = Config()
    uav_data, target_data, obstacle_data = get_new_experimental_scenario(config.OBSTACLE_TOLERANCE)
    print(f"\n---------- 开始场景: '新实验场景' | 求解器: GASolver (最终调试版) ----------")
    ga_solver = GASolver(uav_data, target_data, obstacle_data, config)
    final_plan_ga, ga_time, scheduling_time = ga_solver.solve()
    print(f"求解完成。GA优化耗时: {ga_time:.2f}s, 同步规划耗时: {scheduling_time:.2f}s")
    # visualize_task_assignments 应该能处理 deadlocked_tasks=None 的情况
    visualize_task_assignments(final_plan_ga, uav_data, target_data, obstacle_data,
                               config, "新实验场景_GA(调试版)测试",
                               training_time=ga_time,
                               plan_generation_time=scheduling_time,
                               save_report=True,
                               deadlocked_tasks=None) 
    print("\n===== [GASolver (最终调试版) 独立测试运行结束] =====")