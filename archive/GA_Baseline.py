import numpy as np
import random
from collections import defaultdict
from entities import UAV, Target
from typing import List, Dict, Tuple

class GA_Baseline:
    """
    纯对比基线遗传算法：
    - 距离仅用欧氏距离
    - 不感知障碍物
    - 资源分配只考虑满足度和总距离
    - 不做复杂同步
    """
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles, config):
        self.uavs = [UAV(u.id, u.position, u.heading, u.resources, u.max_distance, u.velocity_range, u.economic_speed) for u in uavs]
        self.targets = [Target(t.id, t.position, t.resources, t.value) for t in targets]
        self.config = config
        self.population_size = getattr(config, 'POPULATION_SIZE', 50)
        self.generations = getattr(config, 'GENERATIONS', 30)
        self.max_uavs_per_target = min(3, len(self.uavs))
        self.n_phi = getattr(config, 'GRAPH_N_PHI', 6)
        self.n_genes = len(self.targets) * self.max_uavs_per_target
        self.uav_ids = [u.id for u in self.uavs]
        self.target_ids = [t.id for t in self.targets]
        self.population = self._create_initial_population()

    def _create_initial_population(self):
        pop = []
        for _ in range(self.population_size):
            chrom = np.zeros((3, self.n_genes), dtype=int)
            chrom[0, :] = np.repeat(self.target_ids, self.max_uavs_per_target)
            chrom[1, :] = np.random.choice(self.uav_ids + [0], size=self.n_genes)
            chrom[2, :] = np.random.randint(0, self.n_phi, size=self.n_genes)
            pop.append(chrom)
        return pop

    def _euclidean_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def evaluate_fitness(self, chrom):
        uav_status = {u.id: {'pos': u.position.copy(), 'rem_res': u.resources.copy().astype(float)} for u in self.uavs}
        target_needs = {t.id: t.resources.copy().astype(float) for t in self.targets}
        total_dist = 0.0
        for target_id in self.target_ids:
            idxs = np.where((chrom[0, :] == target_id) & (chrom[1, :] != 0))[0]
            if len(idxs) == 0: continue
            for idx in idxs:
                uav_id = chrom[1, idx]
                uav = uav_status[uav_id]
                t = next(t for t in self.targets if t.id == target_id)
                dist = self._euclidean_dist(uav['pos'], t.position)
                total_dist += dist
                contrib = np.minimum(uav['rem_res'], target_needs[target_id])
                uav['rem_res'] -= contrib
                target_needs[target_id] -= contrib
                uav['pos'] = t.position.copy()
        unfulfilled = sum(np.sum(v) for v in target_needs.values())
        fitness = -total_dist - 1000 * unfulfilled
        return fitness

    def selection(self, fitnesses):
        idxs = np.argsort(fitnesses)[-self.population_size:]
        return [self.population[i].copy() for i in idxs]

    def crossover(self, parents):
        offspring = []
        for i in range(0, self.population_size, 2):
            p1, p2 = parents[i], parents[(i+1)%self.population_size]
            if random.random() < 0.8:
                point = random.randint(1, self.n_genes-1)
                c1, c2 = p1.copy(), p2.copy()
                c1[:, point:], c2[:, point:] = p2[:, point:], p1[:, point:]
                offspring.extend([c1, c2])
            else:
                offspring.extend([p1.copy(), p2.copy()])
        return offspring[:self.population_size]

    def mutation(self, offspring):
        for chrom in offspring:
            if random.random() < 0.2:
                idx = random.randint(0, self.n_genes-1)
                chrom[1, idx] = random.choice(self.uav_ids + [0])
                chrom[2, idx] = random.randint(0, self.n_phi-1)
        return offspring

    def solve(self) -> Tuple[Dict, float, float]:
        import time
        start = time.time()
        for _ in range(self.generations):
            fitnesses = [self.evaluate_fitness(c) for c in self.population]
            parents = self.selection(fitnesses)
            offspring = self.crossover(parents)
            self.population = self.mutation(offspring)
        fitnesses = np.array([self.evaluate_fitness(c) for c in self.population])
        best_idx = np.argmax(fitnesses)
        best_chrom = self.population[best_idx]
        plan = self._decode_plan(best_chrom)
        return plan, time.time()-start, 0.0

    def _decode_plan(self, chrom):
        plan = defaultdict(list)
        uav_status = {u.id: {'pos': u.position.copy(), 'rem_res': u.resources.copy().astype(float)} for u in self.uavs}
        for target_id in self.target_ids:
            idxs = np.where((chrom[0, :] == target_id) & (chrom[1, :] != 0))[0]
            for idx in idxs:
                uav_id = chrom[1, idx]
                uav = uav_status[uav_id]
                t = next(t for t in self.targets if t.id == target_id)
                dist = self._euclidean_dist(uav['pos'], t.position)
                contrib = np.minimum(uav['rem_res'], t.resources)
                uav['rem_res'] -= contrib
                uav['pos'] = t.position.copy()
                plan[uav_id].append({
                    'target_id': target_id,
                    'start_pos': uav['pos'].copy(),
                    'distance': dist,
                    'resource_cost': contrib,
                    'phi_idx': 0,
                    'is_sync_feasible': True,
                    'arrival_time': 0,
                    'step': 1,
                    'speed': 1.0,
                    'path_points': [uav['pos'].copy(), t.position.copy()]
                })
        return plan 