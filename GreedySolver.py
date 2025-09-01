# comparison_solver.py

# --- 基础模块导入 ---
from typing import List, Dict, Tuple
import numpy as np
import time
from collections import defaultdict

# --- [已纠正] 从各自的源文件直接导入所需模块 ---
try:
    # 从 entities.py 导入核心实体
    from entities import UAV, Target
    
    # 从 path_planning.py 导入官方路径规划器和障碍物基类
    from path_planning import RRTPlanner, Obstacle
    
    # [优化] 从 main.py 导入配置类、官方可视化函数、以及核心的图结构和同步规划函数
    from main import Config, visualize_task_assignments, DirectedGraph, calculate_economic_sync_speeds, set_chinese_font
    
    # 从 scenarios.py 导入新的默认测试场景
    from scenarios import get_new_experimental_scenario

except ImportError as e:
    print("="*80)
    print("错误：无法从项目文件 (main.py, entities.py, path_planning.py, scenarios.py) 导入必要的模块。")
    print("请确保 comparison_solver.py 与这些文件在同一个文件夹下。")
    print(f"具体错误: {e}")
    print("="*80)
    # 如果导入失败，程序将无法运行，直接退出
    exit()


class GreedySolver:
    """
    (已优化) 一个基于贪婪策略的任务序列生成器。
    
    核心逻辑:
    在每个决策步骤，从所有可能的 (无人机-目标-角度) 分配中，选择成本最低的一个。
    其输出与 GraphRLSolver 一致，为一个 task_assignments 字典，后续将由
    calculate_economic_sync_speeds 函数处理以生成最终的协同计划。
    """
    def __init__(self, uavs: List[UAV], targets: List[Target], graph: DirectedGraph, config: Config):
        self.uavs = uavs
        self.targets = targets
        self.graph = graph  # [优化] 使用与RL相同的图结构来快速估算距离
        self.config = config
        print("GreedySolver 已初始化 (采用优化的两阶段策略)。")



    def solve(self) -> Tuple[Dict, float]:
        """
        [优化] 执行贪婪算法以生成任务分配序列。
        """
        print("GreedySolver 正在生成任务序列...")
        start_time = time.time()
        
        uav_states = {u.id: {'rem_res': u.initial_resources.copy().astype(float), 'last_vertex': (-u.id, None)} for u in self.uavs}
        target_rem_needs = {t.id: t.resources.copy().astype(float) for t in self.targets}
        task_assignments = defaultdict(list)
        
        while True:
            if all(np.all(needs <= 1e-5) for needs in target_rem_needs.values()):
                print("所有目标资源需求均已满足。任务序列生成完毕。")
                break

            possible_moves = []
            for uav in self.uavs:
                u_state = uav_states[uav.id]
                if not np.any(u_state['rem_res'] > 1e-5):
                    continue

                for target in self.targets:
                    target_need = target_rem_needs[target.id]
                    if not np.any(target_need > 1e-5):
                        continue
                    
                    # --- [修订] 增加关键判断，防止无效分配 ---
                    # 只有当无人机的剩余资源与目标的需求有重叠时，才认为此移动有效
                    potential_contribution = np.minimum(u_state['rem_res'], target_need)
                    if not np.any(potential_contribution > 1e-5):
                        continue
                    # --- 修订结束 ---

                    for phi_idx in range(self.graph.n_phi):
                        start_v = u_state['last_vertex']
                        end_v = (target.id, self.graph.phi_set[phi_idx])
                        dist = self.graph.adjacency_matrix[self.graph.vertex_to_idx[start_v], self.graph.vertex_to_idx[end_v]]
                        cost = dist / uav.economic_speed if uav.economic_speed > 0 else float('inf')
                        possible_moves.append({'uav_id': uav.id, 'target_id': target.id, 'phi_idx': phi_idx, 'cost': cost})
            
            if not possible_moves:
                print("警告: 已没有可行的无人机分配，但仍有目标未满足。")
                break
                
            best_move = min(possible_moves, key=lambda x: x['cost'])
            
            uav_id, target_id, phi_idx = best_move['uav_id'], best_move['target_id'], best_move['phi_idx']
            u_state = uav_states[uav_id]
            
            task_assignments[uav_id].append((target_id, phi_idx))
            
            contribution = np.minimum(u_state['rem_res'], target_rem_needs[target_id])
            u_state['rem_res'] -= contribution
            target_rem_needs[target_id] -= contribution
            u_state['last_vertex'] = (target_id, self.graph.phi_set[phi_idx])
            
        generation_time = time.time() - start_time
        return task_assignments, generation_time

# ==============================================================================
# 独立测试函数入口
# ==============================================================================

if __name__ == '__main__':
    """
    当此文件被直接运行时，此部分代码将被执行。
    """
    print("="*80)
    print(">>> 正在独立运行 GreedySolver (采用优化的协同策略) <<<")
    print("="*80)
    
    set_chinese_font(manual_font_path='C:/Windows/Fonts/simhei.ttf')
    config = Config()

    print("\n---------- 开始场景: '新实验场景' | 求解器: GreedySolver ----------")
    # 1. 获取场景数据
    uavs, targets, obstacles = get_new_experimental_scenario(config.OBSTACLE_TOLERANCE)
    
    # 2. [优化] 创建与RL Solver相同的图结构
    graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI)
    
    # 3. [优化] 实例化求解器，并传入图
    greedy_solver = GreedySolver(uavs, targets, graph, config)
    
    # 4. [优化] 第一阶段：生成任务序列
    task_assignments, generation_time = greedy_solver.solve()
    print(f"贪婪任务序列生成完成，耗时: {generation_time:.4f}s")
    print("生成的任务序列:", {k:v for k,v in task_assignments.items() if v})

    # 5. [优化] 第二阶段：调用官方的同步规划函数生成最终计划
    print("\n正在调用官方 `calculate_economic_sync_speeds` 函数进行协同规划...")
    plan_generation_start_time = time.time()
    final_plan, deadlocked_tasks = calculate_economic_sync_speeds(
        task_assignments, uavs, targets, graph, obstacles, config
    )
    plan_generation_time = time.time() - plan_generation_start_time
    print(f"协同规划完成，耗时: {plan_generation_time:.4f}s")

    # 6. 可视化最终结果
    visualize_task_assignments(
        final_plan, 
        uavs, 
        targets, 
        obstacles,
        config, 
        scenario_name="新实验场景_Greedy协同优化测试", 
        training_time=generation_time, # 此处用序列生成时间代替训练时间
        plan_generation_time=plan_generation_time,
        deadlocked_tasks=deadlocked_tasks
    )

    print("\n===== [GreedySolver 独立测试运行结束] =====")