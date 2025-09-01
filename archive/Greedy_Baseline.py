import numpy as np
from collections import defaultdict
from entities import UAV, Target
from typing import List, Dict, Tuple

class Greedy_Baseline:
    """
    纯对比基线贪婪算法：
    - 距离仅用欧氏距离
    - 不感知障碍物
    - 每次分配最近且有资源的无人机-目标对
    - 不做复杂同步
    """
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles, config):
        self.uavs = [UAV(u.id, u.position, u.heading, u.resources, u.max_distance, u.velocity_range, u.economic_speed) for u in uavs]
        self.targets = [Target(t.id, t.position, t.resources, t.value) for t in targets]
        self.config = config
        self.uav_ids = [u.id for u in self.uavs]
        self.target_ids = [t.id for t in self.targets]

    def _euclidean_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def solve(self) -> Tuple[Dict, float, float]:
        import time
        start = time.time()
        uav_status = {u.id: {'pos': u.position.copy(), 'rem_res': u.resources.copy().astype(float)} for u in self.uavs}
        target_needs = {t.id: t.resources.copy().astype(float) for t in self.targets}
        plan = defaultdict(list)
        while any(np.any(needs > 1e-6) for needs in target_needs.values()):
            best = None
            best_dist = float('inf')
            for uav_id in self.uav_ids:
                uav = uav_status[uav_id]
                if not np.any(uav['rem_res'] > 1e-6):
                    continue
                for target_id in self.target_ids:
                    if not np.any(target_needs[target_id] > 1e-6):
                        continue
                    dist = self._euclidean_dist(uav['pos'], next(t for t in self.targets if t.id == target_id).position)
                    if dist < best_dist:
                        best = (uav_id, target_id, dist)
                        best_dist = dist
            if best is None:
                break
            uav_id, target_id, dist = best
            uav = uav_status[uav_id]
            t = next(t for t in self.targets if t.id == target_id)
            contrib = np.minimum(uav['rem_res'], target_needs[target_id])
            uav['rem_res'] -= contrib
            target_needs[target_id] -= contrib
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
            uav['pos'] = t.position.copy()
        return plan, time.time()-start, 0.0 