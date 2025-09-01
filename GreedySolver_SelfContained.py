# 文件名: GreedySolver_SelfContained.py
# 描述: (已修复) 一个将路径和时间约束内置的、自包含的贪婪求解器。

from typing import List, Dict, Tuple
import numpy as np
import time
from collections import defaultdict

from entities import UAV, Target
from path_planning import PHCurveRRTPlanner, Obstacle
from main import Config, visualize_task_assignments, set_chinese_font, DirectedGraph
from scenarios import get_new_experimental_scenario

class GreedySolver_SelfContained:
    """
    (新对比算法) 一个将路径和时间约束内置的、自包含的贪婪求解器。
    (v2.0 已修复数据类型错误)
    """
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles: List[Obstacle], graph: DirectedGraph, config: Config):
        # [一致性] 深度复制场景实体，确保求解器不修改外部原始数据
        self.uavs = [UAV(u.id, u.position, u.heading, u.resources, u.max_distance, u.velocity_range, u.economic_speed) for u in uavs]
        self.targets = [Target(t.id, t.position, t.resources, t.value) for t in targets]
        self.obstacles = obstacles
        self.graph = graph
        self.config = config
        self.uav_map = {u.id: u for u in self.uavs}
        self.target_map = {t.id: t for t in self.targets}
        # [新机制] 内部维护无人机状态
        self.uav_status = {u.id: {'pos': u.position.copy(), 'free_at': 0.0, 'heading': u.heading} for u in self.uavs}
        # [修复] 创建一个无人机剩余资源的浮点类型副本，用于计算，避免类型冲突
        self.uav_rem_resources = {u.id: u.resources.copy().astype(float) for u in self.uavs}
        print("GreedySolver_SelfContained 已初始化 (采用内置约束的“先到先得”策略)。")

    def _plan_single_leg(self, start_pos, target_pos, start_heading, end_heading):
        planner = PHCurveRRTPlanner(start_pos, target_pos, start_heading, end_heading, self.obstacles, self.config)
        return planner.plan()

    def solve(self) -> Tuple[Dict, float, float]:
        print("GreedySolver_SelfContained 正在生成方案...")
        start_time = time.time()
        
        target_rem_needs = {t.id: t.resources.copy().astype(float) for t in self.targets}
        final_plan = defaultdict(list)
        task_step_counter = defaultdict(lambda: 1)

        while any(np.any(needs > 1e-6) for needs in target_rem_needs.values()):
            possible_moves = []
            
            for uav in self.uavs:
                u_state = self.uav_status[uav.id]
                uav_res = self.uav_rem_resources[uav.id]
                if not np.any(uav_res > 1e-6):
                    continue

                for target in self.targets:
                    if not np.any(target_rem_needs[target.id] > 1e-6) or \
                       not np.any(np.minimum(uav_res, target_rem_needs[target.id]) > 1e-6):
                        continue
                    
                    for phi_idx in range(self.graph.n_phi):
                        phi_angle = phi_idx * (2 * np.pi / self.config.GRAPH_N_PHI)
                        plan_result = self._plan_single_leg(u_state['pos'], target.position, u_state['heading'], phi_angle)
                        
                        if plan_result:
                            _, distance = plan_result
                            travel_time = distance / uav.velocity_range[1]
                            arrival_time = u_state['free_at'] + travel_time
                            possible_moves.append({
                                'uav_id': uav.id, 'target_id': target.id, 'phi_idx': phi_idx,
                                'cost': arrival_time, 'plan_result': plan_result
                            })

            if not possible_moves:
                print("警告: 已没有可行的无人机分配，但仍有目标未满足，规划终止。")
                break
                
            best_move = min(possible_moves, key=lambda x: x['cost'])
            
            uav_id, target_id = best_move['uav_id'], best_move['target_id']
            u_state = self.uav_status[uav_id]
            target_pos = self.target_map[target_id].position
            
            # [修复] 使用浮点类型的资源副本进行计算
            uav_res = self.uav_rem_resources[uav_id]
            contribution = np.minimum(uav_res, target_rem_needs[target_id])
            self.uav_rem_resources[uav_id] -= contribution # 在副本上操作
            target_rem_needs[target_id] -= contribution
            
            path_points, distance = best_move['plan_result']
            arrival_time = best_move['cost']
            travel_time = arrival_time - u_state['free_at']
            speed = (distance / travel_time) if travel_time > 1e-6 else self.uav_map[uav_id].velocity_range[1]

            final_plan[uav_id].append({
                'target_id': target_id, 'start_pos': u_state['pos'].copy(),
                'speed': np.clip(speed, self.uav_map[uav_id].velocity_range[0], self.uav_map[uav_id].velocity_range[1]),
                'arrival_time': arrival_time, 'step': task_step_counter[uav_id],
                'is_sync_feasible': True, 'phi_idx': best_move['phi_idx'],
                'path_points': path_points, 'distance': distance, 'resource_cost': contribution
            })

            task_step_counter[uav_id] += 1
            u_state['pos'] = target_pos.copy()
            u_state['free_at'] = arrival_time
            u_state['heading'] = best_move['phi_idx'] * (2 * np.pi / self.config.GRAPH_N_PHI)

        generation_time = time.time() - start_time
        print(f"方案生成完成，耗时: {generation_time:.4f}s")
        return final_plan, generation_time, 0.0

if __name__ == '__main__':
    print("="*80)
    print(">>> 正在独立运行 GreedySolver_SelfContained (内置约束版) <<<")
    print("="*80)
    
    set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
    config = Config()
    uavs, targets, obstacles = get_new_experimental_scenario(config.OBSTACLE_TOLERANCE)
    graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI)
    
    solver = GreedySolver_SelfContained(uavs, targets, obstacles, graph, config)
    final_plan, generation_time, _ = solver.solve()

    visualize_task_assignments(
        final_plan, 
        uavs, # [说明] 此处传入原始uavs用于显示初始资源
        targets, obstacles, config, 
        scenario_name="新实验场景_Greedy(内置约束版)测试", 
        training_time=generation_time, plan_generation_time=0,
        deadlocked_tasks=None
    )
    print("\n===== [GreedySolver_SelfContained 独立测试运行结束] =====")