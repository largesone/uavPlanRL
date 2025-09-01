# -*- coding: utf-8 -*-
# 文件名: solvers.py
# 描述: 包含解决任务分配问题的核心算法，主要是基于强化学习的求解器。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import time
from typing import Optional, Callable

from environment import UAVTaskEnv

# =============================================================================
# section 4: 强化学习求解器
# =============================================================================

class ReplayBuffer:
    """经验回放池，用于存储和采样DQN的训练数据"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """(已修订) 简单的深度Q网络，包含批归一化和Dropout"""
    def __init__(self, i_dim, h_dim, o_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(i_dim, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(h_dim, h_dim // 2)
        self.bn2 = nn.BatchNorm1d(h_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(h_dim // 2, o_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)

class GraphRLSolver:
    """(已修订) 基于图和深度强化学习的无人机任务分配求解器"""
    def __init__(self, uavs, targets, graph, obstacles, i_dim, h_dim, o_dim, config, obs_mode="flat"):
        self.uavs, self.targets, self.graph, self.obstacles, self.config = uavs, targets, graph, obstacles, config
        self.obs_mode = obs_mode
        self.env = UAVTaskEnv(uavs, targets, graph, obstacles, config, obs_mode=obs_mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 根据网络类型创建网络
        from networks import create_network
        self.policy_net = create_network(config.NETWORK_TYPE, i_dim, h_dim, o_dim, config).to(self.device)
        self.target_net = create_network(config.NETWORK_TYPE, i_dim, h_dim, o_dim, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        memory_capacity = getattr(config, 'MEMORY_CAPACITY', config.training_config.memory_size)
        self.memory = ReplayBuffer(memory_capacity)
        self.epsilon = getattr(config, 'EPSILON_START', config.training_config.epsilon_start)
        self.batch_size = getattr(config, 'BATCH_SIZE', config.training_config.batch_size)
        
        # 添加动作映射
        self.target_id_map = {t.id: i for i, t in enumerate(self.env.targets)}
        self.uav_id_map = {u.id: i for i, u in enumerate(self.env.uavs)}
        
        # 添加损失跟踪
        self._last_loss = None

        self.action_logger = None

    def _action_to_index(self, a):
        """将动作转换为索引"""
        t_idx, u_idx, p_idx = self.target_id_map[a[0]], self.uav_id_map[a[1]], a[2]
        return t_idx * (len(self.env.uavs) * self.graph.n_phi) + u_idx * self.graph.n_phi + p_idx
    
    def _index_to_action(self, i):
        """将索引转换为动作"""
        n_u, n_p = len(self.env.uavs), self.graph.n_phi
        t_idx, u_idx, p_idx = i // (n_u * n_p), (i % (n_u * n_p)) // n_p, i % n_p
        return (self.env.targets[t_idx].id, self.env.uavs[u_idx].id, p_idx)
    
    def select_action(self, state):
        """使用Epsilon-Greedy策略选择动作 - 支持动作掩码，优化Action=0问题"""
        # 处理状态输入
        if self.obs_mode == "graph":
            # 图模式：处理字典状态
            processed_state = self._process_graph_state(state)
        else:
            # 扁平模式：直接处理张量
            if isinstance(state, np.ndarray):
                processed_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                processed_state = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
        
        # 使用eval模式避免BatchNorm问题
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(processed_state)
        self.policy_net.train()
        
        # 获取有效动作掩码
        action_mask = self.env.get_action_mask()
        valid_indices = np.where(action_mask)[0]
        
        # 优化：如果没有有效动作，尝试重新生成动作掩码
        if len(valid_indices) == 0:
            # 记录无效动作情况
            self._invalid_action_count = getattr(self, '_invalid_action_count', 0) + 1
            
            # 尝试重新计算动作掩码（可能是环境状态变化导致的）
            try:
                action_mask = self.env.get_action_mask()
                valid_indices = np.where(action_mask)[0]
            except Exception as e:
                if getattr(self.config, 'ENABLE_DEBUG', True):
                    print(f"重新计算动作掩码失败: {e}")
            
            # 如果仍然没有有效动作，返回一个默认动作（通常是第一个动作）
            if len(valid_indices) == 0:
                # 返回动作空间中的第一个动作，而不是0
                first_action = 0 if q_values.size(1) > 0 else 0
                return torch.tensor([[first_action]], device=self.device, dtype=torch.long), q_values
        
        # 探索：随机选择有效动作
        if random.random() < self.epsilon:
            if len(valid_indices) > 0:
                random_action = int(np.random.choice(valid_indices))
                return torch.tensor([[random_action]], device=self.device, dtype=torch.long), q_values
            else:
                # 如果没有有效动作，返回第一个有效动作
                first_valid = valid_indices[0] if len(valid_indices) > 0 else 0
                return torch.tensor([[first_valid]], device=self.device, dtype=torch.long), q_values
        
        # 利用：选择Q值最高的有效动作
        if len(valid_indices) > 0:
            # 将无效动作的Q值设为负无穷
            masked_q_values = q_values.clone()
            invalid_mask = torch.ones(masked_q_values.size(), dtype=torch.bool, device=self.device)
            for idx in valid_indices:
                if idx < masked_q_values.size(1):
                    invalid_mask[0, idx] = False
            
            masked_q_values[invalid_mask] = float('-inf')
            
            # 选择Q值最高的动作
            best_action = masked_q_values.argmax().item()
            
            # 确保选择的动作是有效的
            if best_action not in valid_indices:
                # 如果最佳动作无效，选择第一个有效动作
                best_action = valid_indices[0] if len(valid_indices) > 0 else 0
            
            return torch.tensor([[best_action]], device=self.device, dtype=torch.long), q_values
        else:
            # 如果没有有效动作，返回第一个动作
            first_action = 0 if q_values.size(1) > 0 else 0
            return torch.tensor([[first_action]], device=self.device, dtype=torch.long), q_values

    def optimize_model(self):
        """从经验回放池中采样并优化模型 - 支持图模式"""
        batch_size = getattr(self.config, 'BATCH_SIZE', self.config.training_config.batch_size)
        if len(self.memory) < batch_size: 
            return None
        
        transitions = self.memory.sample(batch_size)
        batch = tuple(zip(*transitions))
        
        # 处理状态批次
        if self.obs_mode == "graph":
            # 图模式：合并字典状态
            state_batch = self._merge_graph_states(batch[0])
            next_states_batch = self._merge_graph_states(batch[3])
        else:
            # 扁平模式：直接拼接
            state_batch = torch.cat(batch[0])
            next_states_batch = torch.cat(batch[3])
        
        action_batch = torch.cat(batch[1])
        # 处理奖励批次，确保是张量
        reward_tensors = []
        for reward in batch[2]:
            if isinstance(reward, (int, float, np.integer, np.floating)):
                reward_tensors.append(torch.tensor([reward], dtype=torch.float32).to(self.device))
            else:
                reward_tensors.append(reward)
        reward_batch = torch.cat(reward_tensors)
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.bool)

        # 计算当前Q值
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算下一状态的Q值
        next_q_values = torch.zeros(batch_size, device=self.device)
        non_final_mask = ~done_batch
        
        if non_final_mask.sum() > 0:
            if self.obs_mode == "graph":
                # 图模式：提取非终止状态
                non_final_next_states = self._extract_non_final_graph_states(next_states_batch, non_final_mask)
            else:
                # 扁平模式：直接索引
                non_final_next_states = next_states_batch[non_final_mask]
            
            if non_final_next_states is not None:
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # 计算目标Q值
        gamma = getattr(self.config, 'GAMMA', self.config.training_config.gamma)
        expected_q_values = (next_q_values * gamma) + reward_batch
        
        # 计算损失
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _merge_graph_states(self, state_list):
        """合并图结构状态列表为批次"""
        if not state_list or not isinstance(state_list[0], dict):
            return None
        
        merged_state = {}
        first_state = state_list[0]
        batch_size = len(state_list)
        
        for key in first_state.keys():
            if key == "masks":
                # 处理掩码字典
                merged_masks = {}
                for mask_key in first_state[key].keys():
                    mask_tensors = []
                    for state in state_list:
                        mask_value = state[key][mask_key]
                        if isinstance(mask_value, np.ndarray):
                            mask_tensors.append(torch.FloatTensor(mask_value).to(self.device))
                        elif isinstance(mask_value, (int, float, np.integer, np.floating)):
                            # 处理标量值
                            mask_tensors.append(torch.tensor([mask_value], dtype=torch.float32).to(self.device))
                        else:
                            mask_tensors.append(mask_value)
                    
                    # 合并张量，确保批次维度正确
                    if mask_tensors:
                        if mask_tensors[0].dim() == 1:
                            # 一维张量：直接拼接
                            merged_masks[mask_key] = torch.stack(mask_tensors, dim=0)
                        else:
                            # 多维张量：在第一个维度拼接
                            merged_masks[mask_key] = torch.cat(mask_tensors, dim=0)
                merged_state[key] = merged_masks
            else:
                # 处理其他张量
                tensors = []
                for state in state_list:
                    value = state[key]
                    if isinstance(value, np.ndarray):
                        tensors.append(torch.FloatTensor(value).to(self.device))
                    else:
                        tensors.append(value)
                
                # 合并张量，确保批次维度正确
                if tensors:
                    if tensors[0].dim() == 2:
                        # 二维张量：堆叠以创建批次维度
                        merged_state[key] = torch.stack(tensors, dim=0)
                    else:
                        # 其他维度：堆叠
                        merged_state[key] = torch.stack(tensors, dim=0)
        
        return merged_state
    
    def _extract_non_final_graph_states(self, graph_states, mask):
        """从图状态批次中提取非终止状态"""
        if not isinstance(graph_states, dict):
            return None
        
        extracted_state = {}
        for key, value in graph_states.items():
            if key == "masks":
                # 处理掩码字典
                extracted_masks = {}
                for mask_key, mask_value in value.items():
                    try:
                        batch_size = mask.shape[0]
                        
                        # 确保掩码值的第一个维度与批次掩码匹配
                        if mask_value.shape[0] != batch_size:
                            if mask_key == "interaction_mask":
                                # interaction_mask 特殊处理：可能是展平的交互矩阵
                                if len(mask_value.shape) == 2 and mask_value.shape[0] > batch_size:
                                    # 如果是 [N_uav*N_target, features] 形式，重新整形
                                    n_features = mask_value.shape[1]
                                    # 尝试重新整形为 [batch_size, -1]
                                    try:
                                        reshaped_size = mask_value.shape[0] // batch_size
                                        mask_value = mask_value.view(batch_size, reshaped_size, n_features)
                                        mask_value = mask_value.view(batch_size, -1)  # 展平后两个维度
                                    except:
                                        # 如果重新整形失败，直接截取
                                        mask_value = mask_value[:batch_size]
                                else:
                                    # 其他情况：调整到正确的批次大小
                                    if mask_value.shape[0] > batch_size:
                                        mask_value = mask_value[:batch_size]
                                    elif mask_value.shape[0] < batch_size:
                                        # 重复最后一行到所需大小
                                        repeat_count = batch_size - mask_value.shape[0]
                                        last_row = mask_value[-1:].expand(repeat_count, *mask_value.shape[1:])
                                        mask_value = torch.cat([mask_value, last_row], dim=0)
                            else:
                                # 其他掩码的标准处理
                                if mask_value.shape[0] > batch_size:
                                    mask_value = mask_value[:batch_size]
                                elif mask_value.shape[0] < batch_size:
                                    repeat_count = batch_size - mask_value.shape[0]
                                    last_row = mask_value[-1:].expand(repeat_count, *mask_value.shape[1:])
                                    mask_value = torch.cat([mask_value, last_row], dim=0)
                        
                        # 应用掩码提取非终止状态
                        extracted_masks[mask_key] = mask_value[mask]
                        
                    except Exception as e:
                        print(f"警告: 处理掩码 {mask_key} 时出错: {e}")
                        print(f"  mask_value.shape: {mask_value.shape}, mask.shape: {mask.shape}")
                        # 如果出错，跳过这个掩码或使用默认值
                        if mask.sum() > 0:
                            # 创建一个默认的掩码值
                            default_shape = list(mask_value.shape)
                            default_shape[0] = mask.sum().item()
                            extracted_masks[mask_key] = torch.zeros(default_shape, 
                                                                  dtype=mask_value.dtype, 
                                                                  device=mask_value.device)
                
                extracted_state[key] = extracted_masks
            else:
                # 处理其他张量
                try:
                    extracted_state[key] = value[mask]
                except Exception as e:
                    print(f"警告: 处理状态键 {key} 时出错: {e}")
                    # 创建默认值
                    if mask.sum() > 0:
                        default_shape = list(value.shape)
                        default_shape[0] = mask.sum().item()
                        extracted_state[key] = torch.zeros(default_shape, 
                                                         dtype=value.dtype, 
                                                         device=value.device)
        
        return extracted_state

    def _process_graph_state(self, state):
        """处理图模式的状态输入"""
        if isinstance(state, dict):
            # 处理字典状态
            processed_state = {}
            for key, value in state.items():
                if key == "masks":
                    # 处理masks字典
                    processed_state[key] = {}
                    for mask_key, mask_value in value.items():
                        if isinstance(mask_value, torch.Tensor):
                            processed_state[key][mask_key] = mask_value.unsqueeze(0)
                        elif isinstance(mask_value, np.ndarray):
                            processed_state[key][mask_key] = torch.FloatTensor(mask_value).unsqueeze(0).to(self.device)
                        else:
                            processed_state[key][mask_key] = mask_value
                else:
                    # 处理其他张量
                    if isinstance(value, torch.Tensor):
                        processed_state[key] = value.unsqueeze(0)
                    elif isinstance(value, np.ndarray):
                        processed_state[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                    else:
                        processed_state[key] = value
            return processed_state
        else:
            # 如果不是字典，按扁平模式处理
            if isinstance(state, np.ndarray):
                return torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                return state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)

    def train(self, episodes: int, patience: int, log_interval: int, model_save_path: str, 
              on_episode_end: Optional[Callable[[int, float, float], None]] = None, 
              on_episode_start: Optional[Callable[[int], None]] = None,
              scenario_name: str = 'medium') -> float:
        """(已修订) 完整的训练循环，包含早停、模型保存和训练停滞检测

        兼容图/扁平观测与五元组环境步进返回。
        
        新增功能:
        - 训练停滞检测：监控连续零动作，达到15次时提前终止当前轮次
        - 增强错误处理：对动作解析失败等异常情况进行处理
        - 详细日志记录：记录提前终止原因和统计信息
        
        Args:
            episodes: 训练轮次数
            patience: 早停耐心值
            log_interval: 日志输出间隔
            model_save_path: 模型保存路径
            on_episode_end: 轮次结束回调函数
            on_episode_start: 轮次开始回调函数（用于记录初始场景状态）
            scenario_name: 场景名称
            
        Returns:
            float: 训练总耗时（秒）
            
        注意:
            - 连续15次选择动作0将触发提前终止机制
            - 提前终止不会影响整体训练流程，会继续下一轮训练
        """
        start_time = time.time()
        best_reward = -np.inf
        patience_counter = 0

        for i_episode in range(1, episodes + 1):
            # 调用轮次开始回调（用于记录初始场景状态）
            if on_episode_start is not None:
                try:
                    on_episode_start(i_episode)
                except Exception as e:
                    print(f"Episode start callback 失败: {e}")
            
            # 初始化训练状态监控变量（带错误处理）
            try:
                consecutive_zero_actions = 0
            except Exception as e:
                print(f"警告: 状态监控变量初始化失败: {e}")
                consecutive_zero_actions = 0
            
            # 【修复】初始化状态 - 传递轮次信息确保每个轮次只重置一次
            reset_options = {
                'scenario_name': scenario_name,
                'episode': i_episode
            } if scenario_name else {'episode': i_episode}
            reset_result = self.env.reset(options=reset_options)

            self.graph = self.env.graph

            if getattr(self.config, 'ENABLE_DEBUG', False):
                print(f"--- DEBUG POINT 1 (post-reset) ---")
                print(f"Solver graph ID: {id(self.graph)}, Env graph ID: {id(self.env.graph)}")
                print(f"Solver graph nodes: {len(self.graph.nodes) if self.graph else 'None'}")
                print(f"Env uavs: {len(self.env.uavs)}, Env targets: {len(self.env.targets)}")
                print(f"------------------------------------")
            # --- ↑↑↑ 调试代码结束 ↑↑↑ ---

            if isinstance(reset_result, tuple):
                state, info = reset_result
            else:
                state = reset_result
                info = {}
            episode_start_time = time.time()
            if self.obs_mode == "graph":
                state_tensor = {}
                for key, value in state.items():
                    if key == "masks":
                        mask_tensor = {}
                        for mask_key, mask_value in value.items():
                            # 确保掩码有正确的批次维度
                            if isinstance(mask_value, np.ndarray):
                                if mask_value.ndim == 1:
                                    mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0).to(self.device)
                                else:
                                    mask_tensor[mask_key] = torch.tensor(mask_value).to(self.device)
                            else:
                                mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0).to(self.device)
                        state_tensor[key] = mask_tensor
                    else:
                        # 确保张量有正确的批次维度
                        if isinstance(value, np.ndarray):
                            if value.ndim == 2:  # [N_entities, features]
                                state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                            elif value.ndim == 3:  # [N_uav, N_target, features]
                                state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                            else:
                                state_tensor[key] = torch.FloatTensor(value).to(self.device)
                        else:
                            state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                state = state_tensor
            else:
                state = torch.tensor([state], device=self.device, dtype=torch.float32)

            episode_reward = 0.0
            step_counter = 0
            episode_rewards_this_episode = []  # 用于记录每步奖励
            total_base_reward = 0.0  # 累积基础奖励
            total_shaping_reward = 0.0  # 累积塑形奖励
            action_sequence = [] # 用于记录本轮的动作序列

            while True:
                # 步骤1: 创建UAV资源状态的快照
                uav_resources_snapshot = {uav.id: uav.resources.copy() for uav in self.env.uavs}
                uav_positions_snapshot = {uav.id: uav.current_position.copy() for uav in self.env.uavs} # <--- 新增此行
                
                # 1. 首先获取有效动作列表，用于日志记录
                action_mask = self.env.get_action_mask()
                valid_indices = np.where(action_mask)[0]

                # print(f"可选动作列表 (共 {len(valid_indices)} 个):")
                if len(valid_indices) == 0:
                    if getattr(self.config, 'ENABLE_DEBUG', True):
                        print("  [警告] 未发现任何有效动作！")
                else:
                    # 为了更清晰，我们将解码每个有效动作并打印
                    decoded_actions = []
                    for action_idx in valid_indices:
                        try:
                            # 使用环境的解码函数
                            target_idx, uav_idx, _ = self.env.decode_action(action_idx)
                            target_id = self.env.targets[target_idx].id
                            uav_id = self.env.uavs[uav_idx].id
                            decoded_actions.append(f"  - Action({action_idx:3d}): UAV({uav_id}) -> Target({target_id})")
                        except IndexError:
                            decoded_actions.append(f"  - Action({action_idx:3d}): 解码错误")
                    
                    # # 打印解码后的动作，在后面已经输出了
                    # if getattr(self.config, 'ENABLE_DEBUG', True):
                    #     for line in decoded_actions:
                    #         print(line)

                # 计算动作执行前的完成率
                pre_action_completion_rate = None
                if hasattr(self.env, 'get_completion_rate'):
                    pre_action_completion_rate = self.env.get_completion_rate()
                else:
                    completed_targets_pre = sum(1 for t in self.env.targets if np.all(t.remaining_resources <= 0))
                    pre_action_completion_rate = completed_targets_pre / len(self.env.targets)

                # 2. 智能体根据状态选择动作                
                action, q_values = self.select_action(state)
                action_sequence.append(action.item()) # <记录选择的动作

                # ==================== 新增调试信息打印模块 ====================
                if getattr(self.config, 'ENABLE_DEBUG', True):
                    print(f"\n[TRAIN_DEBUG] Episode {i_episode}, Step {step_counter + 1} | 动作候选列表 (Epsilon: {self.epsilon:.3f})")
                    
                    if len(valid_indices) == 0:
                        print("  [警告] 当前步骤无有效动作可选。")
                    
                    for i, action_idx in enumerate(valid_indices):
                        try:
                            # 解码动作
                            target_idx, uav_idx, _ = self.env.decode_action(action_idx)
                            target = self.env.targets[target_idx]
                            uav = self.env.uavs[uav_idx]
                            
                            # 提取对应的Q值
                            q_value = q_values[0, action_idx].item()
                            
                            # 计算上下文信息 (与evaluator.py中逻辑相同)
                            uav_pos = uav.current_position
                            target_pos = target.position
                            distance = np.linalg.norm(uav_pos - target_pos)
                            uav_resources = uav.resources
                            target_needs = target.remaining_resources
                            possible_contribution = np.minimum(uav_resources, target_needs)
                            
                            # 打印详细信息
                            print(f"  > 动作 {i+1} (索引 {action_idx}): UAV {uav.id} -> Target {target.id}"
                                f"      - Q值: {q_value: .4f}, 距离: {distance:.1f}m, "
                                f"UAV资源: {uav_resources}, 目标需求: {target_needs}, "
                                f"潜在贡献: {possible_contribution}")
                        except Exception as e:
                            print(f"  > 动作 {i+1} (索引 {action_idx}): 处理时发生错误: {e}")
                    
                    # 打印最终选择
                    chosen_action_idx = action.item()
                    print(f"  >> [最终选择]: 动作索引 = {chosen_action_idx} (Q值: {q_values[0, chosen_action_idx].item():.4f})")
                    print("-" * 60)
                # ==========================================================
                # 3. 执行动作并获取奖励
                next_state, reward, done, truncated, info = self.env.step(action.item())

                
                # ==================== 单行奖励构成调试输出模块 ====================
                if getattr(self.config, 'ENABLE_DEBUG', True):
                    reward_breakdown = info.get('reward_breakdown', {})
                    if reward_breakdown:
                        parts = []
                        # 智能地遍历所有可能的奖励构成部分
                        for component_dict_key in ['layer1_breakdown', 'layer2_breakdown', 'extra_rewards', 'simple_breakdown']:
                            component_dict = reward_breakdown.get(component_dict_key, {})
                            if component_dict:
                                for key, value in component_dict.items():
                                    parts.append(f"{key}={value:.1f}")
                                    # # 只显示有显著贡献的项，保持行整洁
                                    # if isinstance(value, (int, float)) and abs(value) > 0.01:
                                    #     parts.append(f"{key}={value:.1f}")
                        
                        breakdown_str = " | ".join(parts)
                        # 使用更紧凑的 [REWARD] 标签
                        print(f"  [REWARD] Step {step_counter + 1}: Total={reward:.2f} [ {breakdown_str} ]")
                    else:
                        # 如果没有详细分解信息，也保持单行输出
                        print(f"  [REWARD] Step {step_counter + 1}: Total={reward:.2f} [无详细构成]")
                    
                    # 使用不同的分隔符，以便和Q值调试信息区分开
                    print("=" * 80)
                # =====================================================================
                 

                # 计算动作执行后的完成率
                post_action_completion_rate = None
                if hasattr(self.env, 'get_completion_rate'):
                    post_action_completion_rate = self.env.get_completion_rate()
                else:
                    completed_targets_post = sum(1 for t in self.env.targets if np.all(t.remaining_resources <= 0))
                    post_action_completion_rate = completed_targets_post / len(self.env.targets)
                
                # [新增] 调用动作日志记录，包含奖励信息和执行前后的完成率
                if hasattr(self, 'action_logger') and self.action_logger is not None:
                    # 准备步骤信息，包含奖励分解
                    step_info = {
                        'reward_breakdown': info.get('reward_breakdown', {}),
                        'base_reward': info.get('base_reward', reward * 0.8),
                        'shaping_reward': info.get('shaping_reward', reward * 0.2)
                    }
                    
                    self.action_logger(
                        episode=i_episode,
                        step=step_counter,
                        valid_actions=valid_indices,
                        chosen_action=action.item(),
                        env=self.env,
                        reward=reward,
                        step_info=step_info,                        
                        pre_action_completion_rate=pre_action_completion_rate,
                        post_action_completion_rate=post_action_completion_rate,
                        uav_resources_snapshot=uav_resources_snapshot, # [新增] 传递快照
                        uav_positions_snapshot=uav_positions_snapshot
                    ) 
                # 动作监控和计数逻辑（带错误处理）
                try:
                    action_value = action.item() if hasattr(action, 'item') else action
                    if action_value == 0:
                        consecutive_zero_actions += 1
                    else:
                        consecutive_zero_actions = 0
                except Exception as e:
                    if getattr(self.config, 'ENABLE_DEBUG', True):
                        print(f"警告: 动作解析失败: {e}，使用默认非零动作逻辑")
                    consecutive_zero_actions = 0  # 默认为非零动作
                
                # 提前终止条件检查（带错误处理）
                try:
                    if consecutive_zero_actions >= self.config.STAGNATION_THRESHOLD:
                        if getattr(self.config, 'ENABLE_DEBUG', True):
                            print(f"第 {i_episode} 轮因策略停滞被提前终止（连续 {consecutive_zero_actions} 次零动作）。")
                        # 1. 定义惩罚
                        reward = self.config.DEADLOCK_PENALTY
                        reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
                        
                        # 2. 存储“失败”的经验。这是一个终止状态(done=True)。
                        #    这是让智能体学习“什么不能做”的关键。
                        self.memory.push(state, action, reward_tensor, state, True)
                        
                        # 3. 累加这个惩罚到总奖励中，以便日志记录
                        episode_reward += reward
                        
                        # 4. 跳出循环，结束本轮
                        break
                except Exception as e:
                    if getattr(self.config, 'ENABLE_DEBUG', True):
                        print(f"警告: 提前终止检查失败: {e}，继续训练")
                    pass
                
                # 环境步进（五元返回）
                episode_reward += reward
                episode_rewards_this_episode.append(reward)  # 记录每步奖励

                # 累积基础奖励和塑形奖励
                base_reward = info.get('base_reward', reward * 0.8)
                shaping_reward = info.get('shaping_reward', reward * 0.2)
                total_base_reward += base_reward
                total_shaping_reward += shaping_reward

                # 记录每步奖励信息（如果trainer提供了回调）
                if hasattr(self, 'step_logger') and self.step_logger is not None:
                    try:
                        # 使用动作执行后的完成率
                        current_completion_rate = post_action_completion_rate
                        
                        # 准备步骤信息
                        step_info = {
                            'reward_breakdown': info.get('reward_breakdown', {}),
                            'base_reward': base_reward,
                            'shaping_reward': shaping_reward
                        }
                        
                        # 计算剩余目标数量（基于动作执行后的状态）
                        completed_targets_post = sum(1 for t in self.env.targets if np.all(t.remaining_resources <= 0))
                        remaining_targets = len(self.env.targets) - completed_targets_post
                        
                        env_info = {
                            'completion_rate': current_completion_rate,
                            'remaining_targets': remaining_targets,
                            'epsilon': self.epsilon
                        }
                        
                        # 记录步骤奖励
                        self.step_logger(step_counter, i_episode, action.item(), reward, step_info, env_info)
                    except Exception as e:
                        if getattr(self.config, 'ENABLE_DEBUG', True):
                            print(f"[WARNING] 步骤奖励记录失败: {e}")

                # 处理next_state为张量/字典张量
                if self.obs_mode == "graph":
                    next_state_tensor = {}
                    for key, value in next_state.items():
                        if key == "masks":
                            mask_tensor = {}
                            for mask_key, mask_value in value.items():
                                # 确保掩码有正确的批次维度
                                if isinstance(mask_value, np.ndarray):
                                    if mask_value.ndim == 1:
                                        mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0).to(self.device)
                                    else:
                                        mask_tensor[mask_key] = torch.tensor(mask_value).to(self.device)
                                else:
                                    mask_tensor[mask_key] = torch.tensor(mask_value).unsqueeze(0).to(self.device)
                            next_state_tensor[key] = mask_tensor
                        else:
                            # 确保张量有正确的批次维度
                            if isinstance(value, np.ndarray):
                                if value.ndim == 2:  # [N_entities, features]
                                    next_state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                                elif value.ndim == 3:  # [N_uav, N_target, features]
                                    next_state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                                else:
                                    next_state_tensor[key] = torch.FloatTensor(value).to(self.device)
                            else:
                                next_state_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                else:
                    next_state_tensor = torch.tensor([next_state], device=self.device, dtype=torch.float32)

                # 存储经验并优化
                reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
                self.memory.push(state, action, reward_tensor, next_state_tensor, done or truncated)
                state = next_state_tensor
                opt_loss = self.optimize_model()
                # 保存最近的损失值供trainer使用
                if opt_loss is not None:
                    self._last_loss = opt_loss

                step_counter += 1
                if done or truncated:
                    break

            # 本轮耗时
            episode_elapsed = time.time() - episode_start_time

            # 保存信息供回调函数使用
            self.step_counter = step_counter
            self.episode_elapsed = episode_elapsed
            self.total_base_reward = total_base_reward
            self.total_shaping_reward = total_shaping_reward

            # 计算完成率（用于回调和日志）- 使用环境的标准计算方法
            if hasattr(self.env, 'get_completion_rate'):
                completion_rate = self.env.get_completion_rate()
            else:
                # 后备方案：使用简单的目标数量比率
                completed_targets = sum(1 for t in self.env.targets if np.all(t.remaining_resources <= 0))
                total_targets = max(1, len(self.env.targets))
                completion_rate = completed_targets / total_targets

            # 回调：记录每轮奖励与完成率
            if on_episode_end is not None:
                try:
                    episode_info = {
                        'action_sequence': action_sequence,
                        'final_env': self.env  # 传递最终的环境状态
                    }
                    on_episode_end(i_episode, float(episode_reward), float(completion_rate), episode_info)
                    
                except Exception:
                    pass

            # 更新探索率
            epsilon_end = getattr(self.config, 'EPSILON_END', self.config.training_config.epsilon_end)
            epsilon_decay = getattr(self.config, 'EPSILON_DECAY', self.config.training_config.epsilon_decay)
            self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)

            # 同步目标网络
            target_update_freq = getattr(self.config, 'TARGET_UPDATE_INTERVAL', self.config.training_config.target_update_freq)
            if i_episode % target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # 分级日志输出控制
            log_level = getattr(self.config, 'LOG_LEVEL', 'simple')
            log_episode_detail = getattr(self.config, 'LOG_EPISODE_DETAIL', False)
            log_reward_detail = getattr(self.config, 'LOG_REWARD_DETAIL', False)
            
            # 轮次信息输出控制
            if log_episode_detail:
                progress_pct = i_episode / episodes * 100
                final_potential = 0.0

                breakdown = getattr(self.env, '_last_reward_breakdown', {}) or {}
                
                # 奖励子项信息格式化
                def _fmt_breakdown(d, max_items=6):
                    try:
                        items = list(d.items())
                        parts = []
                        for k, v in items[:max_items]:
                            if isinstance(v, (int, float)):
                                parts.append(f"{k}={v:.1f}")
                            else:
                                parts.append(f"{k}={v}")
                        if len(items) > max_items:
                            parts.append("...")
                        return ", ".join(parts)
                    except Exception:
                        return str(d)

                # 获取场景关键信息
                num_uavs = len(self.env.uavs)
                num_targets = len(self.env.targets)
                num_obstacles = len(self.env.obstacles) if hasattr(self.env, 'obstacles') and self.env.obstacles else 0
                
                # 计算资源充裕度比率（向量格式）- 使用初始状态数据
                if hasattr(self.env, '_initial_uav_resources') and hasattr(self.env, '_initial_target_demands'):
                    # 使用保存的初始状态数据
                    uav_resources_vector = self.env._initial_uav_resources
                    target_demand_vector = self.env._initial_target_demands
                else:
                    # 后备方案：使用当前状态（可能已被消耗）
                    uav_resources_vector = np.sum([uav.resources for uav in self.env.uavs], axis=0)
                    target_demand_vector = np.sum([target.resources for target in self.env.targets], axis=0)
                resource_abundance_ratio = uav_resources_vector / (target_demand_vector + 1e-6)
                
                # 添加路径算法标注
                path_algo = "高精度PH-RRT" if getattr(self.config, 'USE_PHRRT_DURING_TRAINING', False) else "快速近似"

                # 根据日志级别输出不同详细程度的信息
                if log_level == 'minimal':
                    # 最小输出：只输出关键信息
                    pass  # 不输出任何信息
                elif log_level == 'simple':
                    # 简洁模式：输出基本轮次信息
                    print(
                        f"Episode {i_episode:6d}/{episodes:d} ({progress_pct:5.1f}%) [{path_algo}]: "
                        f"无人机={num_uavs}, 目标={num_targets}, 障碍={num_obstacles}, "
                        f"资源充裕度={resource_abundance_ratio[0]:.2f}|{resource_abundance_ratio[1]:.2f}, "
                        f"步数={step_counter:2d}, 总奖励={episode_reward:7.1f}, 基础奖励={total_base_reward:6.1f}, "
                        f"塑形奖励={total_shaping_reward:6.1f}, 势能={final_potential:.3f}, 完成率={completion_rate:.3f}, "
                        f"探索率={self.epsilon:.3f}, 用时={episode_elapsed:.1f}s",
                        flush=True
                    )
                    
                    # 根据配置决定是否输出奖励分解
                    if log_reward_detail and hasattr(self.config, 'NETWORK_TYPE') and self.config.NETWORK_TYPE == 'ZeroShotGNN':
                        layer1_bd = _fmt_breakdown(breakdown.get('layer1_breakdown', {})) if 'layer1_breakdown' in breakdown else ''
                        layer2_bd = _fmt_breakdown(breakdown.get('layer2_breakdown', {})) if 'layer2_breakdown' in breakdown else ''
                        extra_bd = _fmt_breakdown(breakdown.get('extra_rewards', {})) if 'extra_rewards' in breakdown else ''
                        
                        if layer1_bd:
                            print(f"  分解-L1: {layer1_bd}", flush=True)
                        if layer2_bd:
                            print(f"  分解-L2: {layer2_bd}", flush=True)
                        if extra_bd:
                            print(f"  额外: {extra_bd}", flush=True)
                    elif log_reward_detail:
                        # 非ZeroShotGNN的简单分解
                        simple_bd = breakdown.get('simple_breakdown', {})
                        if simple_bd:
                            def _fmt_simple(d):
                                parts = []
                                for k, v in list(d.items())[:8]:
                                    parts.append(f"{k}={v:.1f}" if isinstance(v, (int, float)) else f"{k}={v}")
                                if len(d) > 8:
                                    parts.append("...")
                                return ", ".join(parts)
                            simple_str = _fmt_simple(simple_bd)
                            print(f"  分解: {simple_str}", flush=True)
                else:
                    # detailed和debug级别：输出完整信息
                    print(
                        f"Episode {i_episode:6d}/{episodes:d} ({progress_pct:5.1f}%) [{path_algo}]: "
                        f"无人机={num_uavs}, 目标={num_targets}, 障碍={num_obstacles}, "
                        f"资源充裕度={resource_abundance_ratio[0]:.2f}|{resource_abundance_ratio[1]:.2f}, "
                        f"步数={step_counter:2d}, 总奖励={episode_reward:7.1f}, 基础奖励={total_base_reward:6.1f}, "
                        f"塑形奖励={total_shaping_reward:6.1f}, 势能={final_potential:.3f}, 完成率={completion_rate:.3f}, "
                        f"探索率={self.epsilon:.3f}, 用时={episode_elapsed:.1f}s",
                        flush=True
                    )
                    
                    # 输出完整的奖励分解
                    if hasattr(self.config, 'NETWORK_TYPE') and self.config.NETWORK_TYPE == 'ZeroShotGNN':
                        layer1_bd = _fmt_breakdown(breakdown.get('layer1_breakdown', {})) if 'layer1_breakdown' in breakdown else ''
                        layer2_bd = _fmt_breakdown(breakdown.get('layer2_breakdown', {})) if 'layer2_breakdown' in breakdown else ''
                        extra_bd = _fmt_breakdown(breakdown.get('extra_rewards', {})) if 'extra_rewards' in breakdown else ''
                        
                        if layer1_bd:
                            print(f"  分解-L1: {layer1_bd}", flush=True)
                        if layer2_bd:
                            print(f"  分解-L2: {layer2_bd}", flush=True)
                        if extra_bd:
                            print(f"  额外: {extra_bd}", flush=True)
                    else:
                        # 非ZeroShotGNN的完整分解
                        simple_bd = breakdown.get('simple_breakdown', {})
                        if simple_bd:
                            def _fmt_simple(d):
                                parts = []
                                for k, v in list(d.items())[:8]:
                                    parts.append(f"{k}={v:.1f}" if isinstance(v, (int, float)) else f"{k}={v}")
                                if len(d) > 8:
                                    parts.append("...")
                                return ", ".join(parts)
                            simple_str = _fmt_simple(simple_bd)
                            print(f"  分解: {simple_str}", flush=True)

            # 早停与保存
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                self.save_model(model_save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if getattr(self.config, 'ENABLE_DEBUG', True):
                    print(f"早停触发于第 {i_episode} 回合。", flush=True)
                break

        training_time = time.time() - start_time
        if getattr(self.config, 'ENABLE_DEBUG', True):
            print(f"训练完成，耗时: {training_time:.2f}秒", flush=True)
        return training_time

    def get_task_assignments(self):
            self.policy_net.eval()
            
            # 1. [修正] 正确处理环境重置的返回值
            
            reset_result = self.env.reset(options={'silent_reset': True})
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
            assignments = {u.id: [] for u in self.env.uavs}
            done, step = False, 0
            max_steps = len(self.env.targets) * len(self.env.uavs)  # 添加最大步数防止死循环

            while not done and step < max_steps:
                # 2. [核心修正] 根据 obs_mode 处理状态
                if self.obs_mode == "graph":
                    # 对于图模式，使用专用的状态处理函数
                    processed_state = self._process_graph_state(state)
                else:
                    # 保持对扁平模式的兼容
                    processed_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_values = self.policy_net(processed_state)
                
                # 3. [修正] 使用动作掩码选择最优的有效动作（纯利用，无探索）
                action_mask = self.env.get_action_mask()
                valid_indices = np.where(action_mask)[0]

                if len(valid_indices) == 0:
                    break  # 如果没有有效动作，则终止分配

                # 将无效动作的Q值设为负无穷
                masked_q_values = q_values.clone()
                masked_q_values[0, ~action_mask] = float('-inf')
                
                action_idx = masked_q_values.argmax().item()
                action = self._index_to_action(action_idx)
                
                target_id, uav_id, phi_idx = action
                
                # 4. [修正] 确保实体索引有效
                if not (0 <= uav_id-1 < len(self.env.uavs) and 0 <= target_id-1 < len(self.env.targets)):
                    step += 1
                    continue

                uav = self.env.uavs[uav_id - 1]
                target = self.env.targets[target_id - 1]

                if np.all(uav.resources <= 0) or np.all(target.remaining_resources <= 0):
                    step += 1
                    continue

                contribution = np.minimum(uav.resources, target.remaining_resources)
                if np.any(contribution > 0):
                    assignments[uav_id].append((target_id, phi_idx))

                # 5. [修正] 正确解包 env.step 的五元组返回值
                next_state, _, done, truncated, _ = self.env.step(action_idx)
                state = next_state
                done = done or truncated  # 任何终止条件都应结束循环

                step += 1
                
            self.policy_net.train()
            return assignments
    
    def save_model(self, path):
        """(已修订) 保存模型，并确保目录存在"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        # print(f"模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            try:
                # 首先尝试使用weights_only=False加载
                model_state = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError as type_error:
                # 处理PyTorch 2.6版本中weights_only参数的变化
                if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                    print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                    model_state = torch.load(path, map_location=self.device)
                else:
                    raise
            except Exception as load_error:
                # 如果失败，尝试使用weights_only=True加载
                print(f"使用weights_only=False加载失败，尝试使用weights_only=True: {load_error}")
                try:
                    model_state = torch.load(path, map_location=self.device, weights_only=True)
                except TypeError as type_error:
                    # 处理PyTorch 2.6版本中weights_only参数的变化
                    if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                        print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                        model_state = torch.load(path, map_location=self.device)
                    else:
                        raise
            
            self.policy_net.load_state_dict(model_state)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.eval() # 设置为评估模式
            self.target_net.eval()
            # print(f"模型已从 {path} 加载。")
            return True
        return False

# Helper function for train loop
from itertools import count