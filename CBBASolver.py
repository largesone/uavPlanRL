# -*- coding: utf-8 -*-
# 文件名: CBBASolver.py
# 描述: 基于Choi et al. 2009论文的改进CBBA算法实现

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

class CBBA_Agent:
    """CBBA智能体类"""
    def __init__(self, uav_id, uav, targets, n_phi):
        self.uav_id = uav_id
        self.uav = uav
        self.targets = targets
        self.n_phi = n_phi
        
        # 任务分配状态
        self.winning_bids = {}  # {target_id: (bid_value, uav_id)}
        self.winning_agents = {}  # {target_id: uav_id}
        self.bundle = []  # 当前智能体的任务包
        self.path = []  # 当前智能体的路径
        self.times = {}  # {target_id: time}
        
        # 共识状态
        self.iteration = 0
        self.max_iterations = 50
        
    def calculate_bid(self, target_id, phi_idx):
        """计算对目标的出价 - 重新设计出价机制"""
        target = next((t for t in self.targets if t.id == target_id), None)
        if not target:
            return 0.0
        
        # 计算距离成本
        distance = np.linalg.norm(self.uav.position - target.position)
        
        # 计算资源匹配度 - 改进计算方式
        available_resources = self.uav.resources.copy()
        target_needs = target.resources.copy()
        
        # 计算每种资源的匹配度
        resource_matches = np.minimum(available_resources, target_needs)
        total_match = np.sum(resource_matches)
        total_available = np.sum(available_resources)
        total_needed = np.sum(target_needs)
        
        # 资源匹配率 - 改进计算
        if total_available > 0 and total_needed > 0:
            resource_ratio = total_match / min(total_available, total_needed)
        else:
            resource_ratio = 0.0
        
        # 计算时间成本
        time_cost = distance / self.uav.velocity_range[1] if self.uav.velocity_range[1] > 0 else 0.0
        
        # 重新设计出价机制 - 基于边际效用
        # 1. 基础价值：目标价值
        base_value = target.value if hasattr(target, 'value') else 100.0
        
        # 2. 资源匹配奖励：匹配度越高，奖励越大
        resource_bonus = resource_ratio * base_value * 2.0
        
        # 3. 距离惩罚：距离越远，惩罚越大
        distance_penalty = distance * 0.05
        
        # 4. 时间惩罚：时间成本
        time_penalty = time_cost * 5.0
        
        # 5. 边际效用：考虑当前UAV的剩余资源
        remaining_resources = np.sum(available_resources)
        if remaining_resources > 0:
            marginal_utility = (total_match / remaining_resources) * base_value
        else:
            marginal_utility = 0.0
        
        # 综合出价计算
        bid = base_value + resource_bonus + marginal_utility - distance_penalty - time_penalty
        
        # 确保出价不为负，且有一定的最小值
        return max(1.0, float(bid))
    
    def update_bundle(self):
        """更新任务包"""
        self.bundle = []
        self.path = []
        self.times = {}
        
        # 获取所有可用目标
        available_targets = []
        for target in self.targets:
            if target.id not in self.winning_agents or self.winning_agents[target.id] == self.uav_id:
                available_targets.append(target.id)
        
        # 贪心构建任务包
        while len(self.bundle) < 5:  # 限制任务包大小
            best_target = None
            best_phi = None
            best_bid = -float('inf')
            
            for target_id in available_targets:
                for phi_idx in range(self.n_phi):
                    bid = self.calculate_bid(target_id, phi_idx)
                    if bid > best_bid:
                        best_bid = bid
                        best_target = target_id
                        best_phi = phi_idx
            
            if best_target is not None and best_bid > 0:
                self.bundle.append((best_target, best_phi))
                self.path.append((best_target, best_phi))
                available_targets.remove(best_target)
            else:
                break
    
    def update_winning_bids(self, target_id, bid_value, agent_id):
        """更新获胜出价"""
        if target_id not in self.winning_bids or bid_value > self.winning_bids[target_id][0]:
            self.winning_bids[target_id] = (bid_value, agent_id)
            self.winning_agents[target_id] = agent_id

class ImprovedCBBASolver:
    """改进的CBBA算法求解器"""
    
    def __init__(self, uavs: List[UAV], targets: List[Target], obstacles: List[Obstacle], config: Config):
        self.uavs = uavs
        self.targets = targets
        self.obstacles = obstacles
        self.config = config
        self.graph = DirectedGraph(uavs, targets, n_phi=config.GRAPH_N_PHI)
        
        # CBBA参数
        self.max_iterations = 50
        self.consensus_threshold = 0.01
        
        # 初始化智能体
        self.agents = {}
        for uav in uavs:
            self.agents[uav.id] = CBBA_Agent(uav.id, uav, targets, config.GRAPH_N_PHI)
        
        print("ImprovedCBBASolver 已初始化 (基于Choi et al. 2009论文)")
    
    def _consensus_phase(self):
        """共识阶段"""
        consensus_reached = False
        iteration = 0
        
        while not consensus_reached and iteration < self.max_iterations:
            iteration += 1
            changes = 0
            
            # 每个智能体更新自己的任务包
            for agent in self.agents.values():
                agent.update_bundle()
            
            # 交换出价信息
            for agent_id, agent in self.agents.items():
                for target_id, phi_idx in agent.bundle:
                    bid_value = agent.calculate_bid(target_id, phi_idx)
                    
                    # 广播出价给其他智能体
                    for other_agent in self.agents.values():
                        if other_agent.uav_id != agent_id:
                            other_agent.update_winning_bids(target_id, bid_value, agent_id)
            
            # 检查共识
            for agent in self.agents.values():
                for target_id, phi_idx in agent.bundle:
                    if target_id in agent.winning_agents:
                        if agent.winning_agents[target_id] != agent.uav_id:
                            # 失去任务，需要重新分配
                            agent.bundle.remove((target_id, phi_idx))
                            changes += 1
            
            if changes == 0:
                consensus_reached = True
        
        return iteration
    
    def _conflict_resolution(self):
        """冲突解决"""
        # 解决任务冲突
        target_assignments = defaultdict(list)
        
        for agent in self.agents.values():
            for target_id, phi_idx in agent.bundle:
                if target_id in agent.winning_agents and agent.winning_agents[target_id] == agent.uav_id:
                    target_assignments[agent.uav_id].append((target_id, phi_idx))
        
        return dict(target_assignments)
    
    def solve(self) -> Tuple[Dict, float, float]:
        """执行改进的CBBA算法"""
        start_time = time.time()
        
        print("开始CBBA优化...")
        
        # 执行共识阶段
        iterations = self._consensus_phase()
        print(f"共识阶段完成，迭代次数: {iterations}")
        
        # 解决冲突
        final_assignments = self._conflict_resolution()
        
        # 转换为标准格式
        final_plan = self._convert_to_standard_format(final_assignments)
        
        cbba_time = time.time() - start_time
        
        # 绘制收敛曲线
        self._plot_convergence(iterations)
        
        return final_plan, cbba_time, 0.0
    
    def _convert_to_standard_format(self, assignments: Dict) -> Dict:
        """转换为标准任务分配格式"""
        final_plan = defaultdict(list)
        
        for uav_id, assignments_list in assignments.items():
            for target_id, phi_idx in assignments_list:
                uav = next((u for u in self.uavs if u.id == uav_id), None)
                target = next((t for t in self.targets if t.id == target_id), None)
                
                if uav and target:
                    resource_cost = np.minimum(uav.initial_resources, target.resources)
                    
                    final_plan[uav_id].append({
                        'target_id': target_id,
                        'phi_idx': phi_idx,
                        'resource_cost': resource_cost,
                        'distance': 0.0,  # 将在后续计算中更新
                        'speed': uav.velocity_range[1],
                        'arrival_time': 0.0,
                        'step': len(final_plan[uav_id]) + 1,
                        'is_sync_feasible': True
                    })
        
        return dict(final_plan)
    
    def _plot_convergence(self, iterations: int):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot([iterations], [1.0], 'ro', markersize=10)
        plt.title('CBBA算法收敛')
        plt.xlabel('迭代次数')
        plt.ylabel('共识状态')
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        output_dir = "output/images"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/cbba_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CBBA收敛曲线已保存至: {output_dir}/cbba_convergence.png")

# 测试代码
if __name__ == "__main__":
    from scenarios import get_small_scenario
    
    config = Config()
    uav_data, target_data, obstacle_data = get_small_scenario(config.OBSTACLE_TOLERANCE)
    
    cbba_solver = ImprovedCBBASolver(uav_data, target_data, obstacle_data, config)
    final_plan_cbba, cbba_time, scheduling_time = cbba_solver.solve()
    
    print(f"CBBA算法完成，耗时: {cbba_time:.2f}s")
    print(f"最终方案: {final_plan_cbba}")