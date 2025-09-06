# -*- coding: utf-8 -*-
# 文件名: environment.py
# 描述: 定义强化学习的环境，包括场景的有向图表示和任务环境本身。

import numpy as np
import itertools
from scipy.spatial.distance import cdist
from typing import Union, Dict, Any, Literal
import gymnasium as gym
from gymnasium import spaces

from entities import UAV, Target
from path_planning import PHCurveRRTPlanner, CircularObstacle

# =============================================================================
# section 3: 场景建模与强化学习环境
# =============================================================================

class DirectedGraph:
    """(已修订) 使用numpy高效构建和管理任务场景的有向图"""
    def __init__(self, uavs, targets, n_phi, obstacles, config):
        self.uavs, self.targets, self.config = uavs, targets, config
        self.n_phi = n_phi
        self.n_uavs, self.n_targets = len(uavs), len(targets)

        if self.n_uavs == 0 and self.n_targets == 0:
            self.uav_positions = np.empty((0, 2))
            self.target_positions = np.empty((0, 2))
            self.nodes = []
            self.node_positions = np.empty((0, 2))
            self.node_map = {}
            self.dist_matrix = np.empty((0, 0))
            self.adj_matrix = np.empty((0, 0))
            self.phi_matrix = np.empty((0, 0))
            return  # 提前退出，不再执行后续计算        

        self.uav_positions = np.array([u.position for u in uavs])
        self.target_positions = np.array([t.position for t in targets])
        
        self.nodes = uavs + targets
        self.node_positions = np.vstack([self.uav_positions, self.target_positions])
        self.node_map = {node.id: i for i, node in enumerate(self.nodes)}

        self.dist_matrix = self._calculate_distances(obstacles)
        self.adj_matrix = self._build_adjacency_matrix()
        self.phi_matrix = self._calculate_phi_matrix()

    def _calculate_distances(self, obstacles):
        """计算所有节点间的距离，使用统一的距离计算服务"""
        from distance_service import get_distance_service
        
        # 获取距离计算服务
        distance_service = get_distance_service(self.config, obstacles)
        
        # 使用距离计算服务计算距离矩阵
        dist_matrix = distance_service.calculate_distance_matrix(
            positions=self.node_positions.tolist(),
            mode='training'  # 在训练环境中使用训练模式
        )
        
        return dist_matrix

    def _build_adjacency_matrix(self):
        """构建邻接矩阵，UAV可以飞到任何目标，目标之间不能互飞"""
        adj = np.zeros((len(self.nodes), len(self.nodes)))
        adj[:self.n_uavs, self.n_uavs:] = 1
        return adj

    def _calculate_phi_matrix(self):
        """(已修订) 高效计算所有节点对之间的相对方向分区(phi值)"""
        delta = self.node_positions[:, np.newaxis, :] - self.node_positions[np.newaxis, :, :]
        angles = np.arctan2(delta[..., 1], delta[..., 0])
        phi_matrix = np.floor((angles % (2 * np.pi)) / (2 * np.pi / self.config.GRAPH_N_PHI))
        return phi_matrix.astype(int)

    def get_dist(self, from_node_id, to_node_id):
        """获取两个节点间的距离"""
        return self.dist_matrix[self.node_map[from_node_id], self.node_map[to_node_id]]

class UAVTaskEnv(gym.Env):
    """
    (已修订) 无人机协同任务分配的强化学习环境
    
    支持双模式观测系统：
    - "flat" 模式：传统扁平向量观测，确保FCN向后兼容性
    - "graph" 模式：结构化图观测，支持TransformerGNN架构和可变数量实体
    """
    def __init__(self, uavs, targets, graph, obstacles, config, obs_mode: Literal["flat", "graph"] = "flat", max_steps=None):
        """
        初始化UAV任务环境
        
        Args:
            uavs: UAV实体列表
            targets: 目标实体列表  
            graph: 有向图对象
            obstacles: 障碍物列表
            config: 配置对象，支持TOP_K_UAVS参数用于动作空间剪枝
            obs_mode: 观测模式，"flat"为扁平向量模式，"graph"为图结构模式
            max_steps: 最大步数，如果为None则使用默认计算逻辑
            
        配置参数:
            TOP_K_UAVS (int): 动作空间剪枝参数，每个目标考虑的最近无人机数量，默认为5
            
        注意:
            - 动作空间剪枝功能通过TOP_K_UAVS参数控制，可显著减少复杂场景下的动作空间大小
            - 增强奖励塑形包含接近奖励机制，鼓励无人机向目标移动
        """
        self.uavs = uavs
        self.targets = targets
        self.graph = graph
        self.obstacles = obstacles
        self.config = config
        self.obs_mode = obs_mode
        self.step_count = 0
        
        # 如果外部提供了max_steps，则使用外部值，否则使用默认计算逻辑
        if max_steps is not None:
            self.max_steps = max_steps
        else:
            self.max_steps = len(targets) * len(uavs) * 2
            
        self.invalid_action_penalty = -5.0  # 从-75.0大幅减少到-5.0
        
        # 动作空间剪枝配置参数 - 直接使用config中的动态参数
        # 参数验证：确保TOP_K_UAVS为正整数且不超过无人机总数
        if not isinstance(config.TOP_K_UAVS, int) or config.TOP_K_UAVS <= 0:
            # 如果TOP_K_UAVS无效，设置为默认值1
            config.TOP_K_UAVS = 1
            # print(f"警告: TOP_K_UAVS无效，设置为默认值: {config.TOP_K_UAVS}")
        if len(uavs) > 0 and config.TOP_K_UAVS > len(uavs):
            # print(f"警告: TOP_K_UAVS ({config.TOP_K_UAVS}) 大于无人机总数 ({len(uavs)})，调整为无人机总数")
            config.TOP_K_UAVS = len(uavs)
        
        # 计算动作空间大小
        if self.graph is not None:
            self.n_actions = len(targets) * len(uavs) * self.graph.n_phi
        else:
            # 当graph为None时，使用默认的n_phi=1
            self.n_actions = len(targets) * len(uavs) * 1
        
        # 确保动作空间大小为正数
        if self.n_actions <= 0:
            self.n_actions = 1  # 默认最小动作空间
        
        # 动态创建观测空间
        self.observation_space = self._create_observation_space()
        
        # 定义动作空间
        self.action_space = spaces.Discrete(self.n_actions)
    
    def _create_observation_space(self) -> spaces.Space:
        """
        动态观测空间创建的工厂模式
        
        根据obs_mode参数创建相应的观测空间：
        - "flat": 扁平向量观测空间，确保FCN向后兼容性
        - "graph": 字典结构观测空间，支持可变数量实体
        
        Returns:
            gym.spaces.Space: 对应模式的观测空间
        """
        if self.obs_mode == "flat":
            return self._create_flat_observation_space()
        elif self.obs_mode == "graph":
            return self._create_graph_observation_space()
        else:
            raise ValueError(f"不支持的观测模式: {self.obs_mode}。支持的模式: ['flat', 'graph']")
    
    def _create_flat_observation_space(self) -> spaces.Box:
        """
        创建扁平向量观测空间，维持现有实现的向后兼容性
        
        状态组成：
        - 目标信息：position(2) + resources(2) + value(1) + remaining_resources(2) = 7 * n_targets
        - UAV信息：position(2) + heading(1) + resources(2) + max_distance(1) + velocity_range(2) = 8 * n_uavs  
        - 协同信息：分配状态 = 1 * n_targets * n_uavs
        - 全局信息：10个全局状态特征
        
        Returns:
            spaces.Box: 扁平向量观测空间
        """
        n_targets = len(self.targets)
        n_uavs = len(self.uavs)
        
        # 计算状态维度
        target_dim = 7 * n_targets  # 每个目标7个特征
        uav_dim = 8 * n_uavs        # 每个UAV 8个特征
        collaboration_dim = n_targets * n_uavs  # 协同分配状态
        global_dim = 10             # 全局状态特征
        
        total_dim = target_dim + uav_dim + collaboration_dim + global_dim
        
        # 创建观测空间，使用合理的边界值
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def _create_graph_observation_space(self) -> spaces.Dict:
        """
        创建图结构观测空间，支持可变数量实体
        
        图模式状态结构：
        - uav_features: [N_uav, uav_feature_dim] - UAV实体特征（归一化）
        - target_features: [N_target, target_feature_dim] - 目标实体特征（归一化）
        - relative_positions: [N_uav, N_target, 2] - 归一化相对位置向量
        - distances: [N_uav, N_target] - 归一化距离矩阵
        - masks: 有效实体掩码字典
        
        Returns:
            spaces.Dict: 图结构观测空间
        """
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        # UAV特征维度：position(2) + heading(1) + resources_ratio(2) + max_distance_norm(1) + 
        #              velocity_norm(2) + is_alive(1) + is_idle(1) = 10
        uav_feature_dim = 10
        
        # 目标特征维度：position(2) + resources_ratio(2) + value_norm(1) + 
        #              remaining_ratio(2) + is_visible(1) = 8  
        target_feature_dim = 8
        
        return spaces.Dict({
            # UAV实体特征矩阵 [N_uav, uav_feature_dim]
            "uav_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(n_uavs, uav_feature_dim),
                dtype=np.float32
            ),
            
            # 目标实体特征矩阵 [N_target, target_feature_dim]
            "target_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(n_targets, target_feature_dim),
                dtype=np.float32
            ),
            
            # 相对位置矩阵 [N_uav, N_target, 2] - 归一化相对位置向量
            "relative_positions": spaces.Box(
                low=-1.0, high=1.0,
                shape=(n_uavs, n_targets, 2),
                dtype=np.float32
            ),
            
            # 距离矩阵 [N_uav, N_target] - 归一化距离
            "distances": spaces.Box(
                low=0.0, high=1.0,
                shape=(n_uavs, n_targets),
                dtype=np.float32
            ),
            
            # 掩码字典
            "masks": spaces.Dict({
                # UAV有效性掩码 [N_uav] - 1表示有效，0表示无效
                "uav_mask": spaces.Box(
                    low=0, high=1,
                    shape=(n_uavs,),
                    dtype=np.int32
                ),
                
                # 目标有效性掩码 [N_target] - 1表示有效，0表示无效
                "target_mask": spaces.Box(
                    low=0, high=1,
                    shape=(n_targets,),
                    dtype=np.int32
                )
            })
        })
    def _update_action_space(self):
        """【新增】根据当前实体更新动作空间大小的辅助函数"""
        if self.graph is not None:
            self.n_actions = len(self.targets) * len(self.uavs) * self.graph.n_phi
        else:
            self.n_actions = len(self.targets) * len(self.uavs) * 1
        
        if self.n_actions <= 0:
            self.n_actions = 1 # 默认最小动作空间

        self.action_space = spaces.Discrete(self.n_actions)

        self.max_steps = len(self.targets) * len(self.uavs) * 2

        if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
            print(f"[RESET DEBUG] Action space updated to size: {self.n_actions}")        


    def reset(self, seed=None, options=None):
        """
        重置环境到一个新的初始状态。
        - 如果提供了 'scenario' 选项（来自课程学习），则加载该预设场景。
        - 否则，根据 'scenario_name' 选项（来自动态模式）动态生成一个新场景。
        """
        # 【修复】防止同一轮次多次重置 - 改进版本
        requested_episode = options.get('episode', -1) if options else -1
        
        # 【修复】只有在明确要求跳过重置时才跳过，否则允许正常重置
        skip_reset = options.get('skip_reset', False) if options else False
        if skip_reset:
            return self._get_state(), self._get_info()
        
        # 如果有轮次信息且已经重置过相同轮次，则跳过（但允许强制重置）
        force_reset = options.get('force_reset', False) if options else False
        if (requested_episode != -1 and 
            hasattr(self, '_last_reset_episode') and 
            self._last_reset_episode == requested_episode and 
            not force_reset):
            # 静默跳过重复重置，不输出调试信息减少日志噪音
            return self._get_state(), self._get_info()
        
        # 如果没有轮次信息，使用调用计数防止短时间内多次重置
        if not hasattr(self, '_reset_call_count'):
            self._reset_call_count = 0
            self._last_reset_time = 0
        
        import time
        current_time = time.time()
        
        # 如果在很短时间内（0.1秒）多次调用reset，跳过后续调用
        if (current_time - self._last_reset_time < 0.1 and 
            self._reset_call_count > 0 and 
            requested_episode == -1):
            # 静默跳过短时间重复重置，减少日志噪音
            return self._get_state(), self._get_info()
        
        self._reset_call_count += 1
        self._last_reset_time = current_time
        
        # 打印重置调试信息 - 受ENABLE_DEBUG控制，支持静默重置
        silent_reset = options.get('silent_reset', False) if options else False
        if getattr(self.config, 'ENABLE_DEBUG', True) and not silent_reset:
            print(f"--- 重置场景 (轮次: {requested_episode}) ---")

        super().reset(seed=seed)

        # 【修复】优先使用传入的scenario_name，确保使用正确的场景级别
        scenario_name = 'hard'  # 默认使用hard模式，与配置保持一致
        
        # 首先检查options参数
        if options and 'scenario_name' in options:
            scenario_name = options['scenario_name']
            if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
                print(f"[RESET DEBUG] 使用options中的场景名称: {scenario_name}")
        # 其次检查环境实例是否有保存的场景名称
        elif hasattr(self, '_current_scenario_name') and self._current_scenario_name:
            scenario_name = self._current_scenario_name
            if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
                print(f"[RESET DEBUG] 使用实例保存的场景名称: {scenario_name}")
        else:
            if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
                print(f"[RESET DEBUG] 使用默认场景名称: {scenario_name}")

        # 保存当前场景名称和轮次信息
        self._current_scenario_name = scenario_name
        self._current_episode = requested_episode
        self._last_reset_episode = requested_episode

        # 【修复】检查是否需要重新生成场景
        need_regenerate_scenario = False
        
        if options and 'scenario' in options:
            # 课程学习模式：直接使用传入的、预先生成好的场景
            scenario = options['scenario']
            self.uavs = scenario['uavs']
            self.targets = scenario['targets']
            self.obstacles = scenario['obstacles']
            need_regenerate_scenario = False
        elif scenario_name in ['small', 'balanced', 'complex']:
            # 静态预定义场景：直接加载，不重新生成
            from scenarios import get_small_scenario, get_balanced_scenario, get_complex_scenario
            obstacle_tolerance = getattr(self.config, 'OBSTACLE_TOLERANCE', 50.0)
            
            if scenario_name == 'small':
                self.uavs, self.targets, self.obstacles = get_small_scenario(obstacle_tolerance)
            elif scenario_name == 'balanced':
                self.uavs, self.targets, self.obstacles = get_balanced_scenario(obstacle_tolerance)
            elif scenario_name == 'complex':
                self.uavs, self.targets, self.obstacles = get_complex_scenario(obstacle_tolerance)
            need_regenerate_scenario = False
        else:
            # 动态随机场景模式：每次都重新生成场景
            is_dynamic_mode = getattr(self.config, 'TRAINING_MODE', '') == 'dynamic'
            is_dynamic_scenario = scenario_name in ['easy', 'medium', 'hard'] # 新增行
            
            if is_dynamic_mode or is_dynamic_scenario: # 修改条件
                # 动态模式下每个episode都重新生成场景
                need_regenerate_scenario = True
                if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False) and not silent_reset:
                    print(f"[RESET DEBUG] 生成新场景数据")
                self._initialize_entities(scenario_name)
                self._scenario_initialized = True

                # --- 新增的修复逻辑：用新生成的实体来重新创建图对象，每当生成新实体后，必须重建图对象并更新动作空间 ---
                if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False) and not silent_reset:
                    print(f"[RESET DEBUG] Re-creating DirectedGraph for new scene.")
                self.graph = DirectedGraph(self.uavs, self.targets, self.config.GRAPH_N_PHI, self.obstacles, self.config)
                self._update_action_space()
                
                # # 核心修复：根据新的实体重新计算动作空间大小
                # if self.graph is not None:
                #     self.n_actions = len(self.targets) * len(self.uavs) * self.graph.n_phi
                # else:
                #     self.n_actions = len(self.targets) * len(self.uavs) * 1
                
                # if self.n_actions <= 0:
                #     self.n_actions = 1 # 默认最小动作空间

                # self.action_space = spaces.Discrete(self.n_actions)                 
                
            else:
                # 非动态模式：检查是否已有场景数据
                if (hasattr(self, 'uavs') and self.uavs and 
                    hasattr(self, 'targets') and self.targets and
                    hasattr(self, 'obstacles') and self.obstacles and
                    hasattr(self, '_scenario_initialized') and self._scenario_initialized):
                    # 已有场景数据，不重新生成
                    need_regenerate_scenario = False
                    if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False) and not silent_reset:
                        print(f"[RESET DEBUG] 使用已有场景数据: UAV={len(self.uavs)}, 目标={len(self.targets)}, 障碍={len(self.obstacles)}")
                else:
                    # 需要生成新场景
                    need_regenerate_scenario = True
                    if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False) and not silent_reset:
                        print(f"[RESET DEBUG] 生成新场景数据")
                    self._initialize_entities(scenario_name)
                    self._scenario_initialized = True
                    # --- 在这里也需要添加图对象重建逻辑 ---
                    if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False) and not silent_reset:
                        print(f"[RESET DEBUG] Re-creating DirectedGraph for new static scene.")
                    self.graph = DirectedGraph(self.uavs, self.targets, self.config.GRAPH_N_PHI, self.obstacles, self.config)
                    self._update_action_space()
                    # # 核心修复：根据新的实体重新计算动作空间大小
                    # if self.graph is not None:
                    #     self.n_actions = len(self.targets) * len(self.uavs) * self.graph.n_phi
                    # else:
                    #     self.n_actions = len(self.targets) * len(self.uavs) * 1
                    
                    # if self.n_actions <= 0:
                    #     self.n_actions = 1 # 默认最小动作空间

                    # self.action_space = spaces.Discrete(self.n_actions)
                    
                    # --- 修复逻辑结束 ---
        # [新增] 强制重置每个实体的内部状态
        for uav in self.uavs:
            # 假设UAV对象有reset方法或直接重置属性
            if hasattr(uav, 'reset'):
                uav.reset()
            else: # 后备方案
                uav.resources = uav.initial_resources.copy()
                uav.current_position = uav.position.copy()

        for target in self.targets:
            # 假设Target对象有reset方法或直接重置属性
            if hasattr(target, 'reset'):
                target.reset()
            else: # 后备方案
                target.remaining_resources = target.resources.copy()
                target.allocated_uavs = []

        # 重置环境内部状态
        self.steps = 0
        self.step_count = 0  # 修复：重置步数计数器
        self.active_uav_ids = set()

        # 保存初始状态数据用于资源充裕度计算
        self._initial_uav_resources = np.sum([uav.resources for uav in self.uavs], axis=0)
        self._initial_target_demands = np.sum([target.resources for target in self.targets], axis=0)

        self.initial_total_demand = sum(np.sum(t.resources) for t in self.targets)
        self.current_remaining_demand = self.initial_total_demand

        initial_obs = self._get_state()
        self.last_potential = self._calculate_potential()

        # 【新增】输出最终场景重置结果
        if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False) and not silent_reset:
            print(f"[RESET DEBUG] 场景重置完成 - 轮次: {requested_episode}, "
                  f"场景: {scenario_name}, "
                  f"UAV={len(self.uavs)}, 目标={len(self.targets)}, 障碍={len(self.obstacles)}")

        return initial_obs, self._get_info()

    def _get_state(self) -> Union[np.ndarray, Dict[str, Any]]:
        """
        获取当前状态，根据obs_mode返回不同格式
        
        Returns:
            Union[np.ndarray, Dict]: 
                - "flat"模式：扁平向量状态
                - "graph"模式：结构化图状态字典
        """
        if self.obs_mode == "flat":
            state = self._get_flat_state()
            # NaN检查已注释掉以提高训练效率
            # if getattr(self.config, 'debug_mode', False):
            #     assert not np.isnan(state).any(), "FATAL: State contains NaN!"
            #     assert np.isfinite(state).all(), "FATAL: State contains Infinity!"
            return state
        elif self.obs_mode == "graph":
            state = self._get_graph_state()
            # 只在调试模式下进行NaN检查，提高训练效率
            # if getattr(self.config, 'debug_mode', False):
            #     self._validate_graph_state(state)
            return state
        else:
            raise ValueError(f"不支持的观测模式: {self.obs_mode}")

    def _get_info(self) -> Dict[str, Any]:
        """
        获取辅助信息，通常为空字典或包含调试信息。
        
        Returns:
            Dict[str, Any]: 辅助信息字典。
        """
        return {}
    
    def _get_flat_state(self) -> np.ndarray:
        """
        获取扁平向量状态，维持现有实现的向后兼容性
        
        Returns:
            np.ndarray: 扁平向量状态
        """
        state = []
        
        # 目标信息
        for target in self.targets:
            target_state = [
                target.position[0], target.position[1],
                target.resources[0], target.resources[1],
                target.value,
                target.remaining_resources[0], target.remaining_resources[1]
            ]
            state.extend(target_state)
        
        # UAV信息
        for uav in self.uavs:
            uav_state = [
                uav.current_position[0], uav.current_position[1],
                uav.heading,
                uav.resources[0], uav.resources[1],
                uav.max_distance,
                uav.velocity_range[0], uav.velocity_range[1]
            ]
            state.extend(uav_state)
        
        # 协同信息
        for target in self.targets:
            for uav in self.uavs:
                is_assigned = any(
                    (uav.id, phi_idx) in target.allocated_uavs 
                    for phi_idx in range(self.graph.n_phi)
                )
                state.append(1.0 if is_assigned else 0.0)
        
        # 全局状态信息 - 修复：使用标准评估指标的完成率计算方法
        total_targets = len(self.targets)
        completed_targets = sum(
            1 for target in self.targets 
            if np.all(target.remaining_resources <= 0)
        )
        
        # 【修复】使用统一的标准完成率计算方法，确保与其他模块一致
        completion_rate = self._calculate_standard_completion_rate()
        
        global_state = [
            self.step_count,
            completion_rate,
            len([u for u in self.uavs if np.any(u.resources > 0)]),
            sum(np.sum(target.remaining_resources) for target in self.targets),
            sum(np.sum(uav.resources) for uav in self.uavs),
            completed_targets,
            total_targets,
            self.max_steps - self.step_count,
            np.mean([uav.heading for uav in self.uavs]),
            np.std([uav.heading for uav in self.uavs])
        ]
        state.extend(global_state)
        
        return np.array(state, dtype=np.float32)
    
    def _get_graph_state(self) -> Dict[str, Any]:
        """
        获取图结构状态，支持TransformerGNN架构
        
        实现尺度不变的状态表示：
        - 移除绝对坐标，使用归一化相对位置
        - 实体特征仅包含归一化的自身属性
        - 添加鲁棒性掩码机制，支持通信/感知失效场景
        - 使用固定维度确保批处理兼容性
        
        Returns:
            Dict[str, Any]: 图结构状态字典
        """
        # 使用固定的最大数量，确保维度一致性
        max_uavs = getattr(self.config, 'MAX_UAVS', 10)
        max_targets = getattr(self.config, 'MAX_TARGETS', 15)
        
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        # 计算地图尺度用于归一化（假设地图为正方形）
        map_size = getattr(self.config, 'MAP_SIZE', 1000.0)
        

        # [新增] 步骤1: 计算全局资源供需情况
        total_remaining_demand = np.sum([t.remaining_resources for t in self.targets if np.any(t.remaining_resources > 0)], axis=0)
        total_available_supply = np.sum([u.resources for u in self.uavs if np.any(u.resources > 0)], axis=0) 
        # === 1. UAV特征矩阵 [max_uavs, uav_feature_dim] ===
        # 将特征维度从 10 增加到 12，以容纳新的全局信息
        uav_feature_dim = 12 
        uav_features = np.zeros((max_uavs, uav_feature_dim), dtype=np.float32)

        for i, uav in enumerate(self.uavs):
            # 归一化位置 [0, 1]
            norm_pos = np.array(uav.current_position) / map_size
            
            # 归一化朝向 [0, 1]
            norm_heading = uav.heading / (2 * np.pi)
            
            # 资源比例 [0, 1]
            initial_resources = getattr(uav, 'initial_resources', uav.resources + 1e-6)
            # 确保数组维度正确
            uav_resources = np.atleast_1d(uav.resources)
            initial_resources = np.atleast_1d(initial_resources)
            resource_ratio = uav_resources / (initial_resources + 1e-6)
            
            # 归一化最大距离 [0, 1]
            norm_max_distance = uav.max_distance / map_size
            
            # 归一化速度范围 [0, 1]
            max_velocity = 100.0  # 假设最大速度
            norm_velocity = np.array(uav.velocity_range) / max_velocity
            
            # 鲁棒性掩码：is_alive位（0/1），标识无人机通信/感知状态
            is_alive = self._calculate_uav_alive_status(uav, i)
            
            # 空闲状态特征：检查UAV是否有资源但未分配任务，1.0代表空闲，0.0代表非空闲
            is_idle = self._calculate_uav_idle_status(uav)

            # [新增] 步骤2: 计算新的全局特征
            # 特征1: 当前无人机资源占机队总供给的比例
            # 确保数组维度正确
            uav_resources = np.atleast_1d(uav.resources)
            total_supply = np.atleast_1d(total_available_supply)
            supply_ratio = uav_resources / (total_supply + 1e-6)
            
            # 重新读取刚修复的uav_resources和remaining_demand
            # 特征2: 当前无人机资源能满足全局总需求的比例 (衡量稀缺性)
            # 确保资源数组维度正确
            uav_resources = np.atleast_1d(uav.resources)
            remaining_demand = np.atleast_1d(total_remaining_demand)
            
            scarcity_metric_res1 = (uav_resources[0] / (remaining_demand[0] + 1e-6) 
                                   if len(uav_resources) > 0 and len(remaining_demand) > 0 else 0.0)
            scarcity_metric_res2 = (uav_resources[1] / (remaining_demand[1] + 1e-6) 
                                   if len(uav_resources) > 1 and len(remaining_demand) > 1 else 0.0)

            uav_features[i] = [
                norm_pos[0], norm_pos[1],           # 归一化位置 (2)
                norm_heading,                       # 归一化朝向 (1)
                resource_ratio[0] if len(resource_ratio) > 0 else 0.0, resource_ratio[1] if len(resource_ratio) > 1 else 0.0, # 资源比例 (2)
                norm_max_distance,                  # 归一化最大距离 (1)
                norm_velocity[0], norm_velocity[1], # 归一化速度 (2)
                is_alive,                          # 存活状态 (1)
                is_idle,                            # 空闲状态 (1) - 新增
                 # [新增] 新的全局视野特征 (2)
                np.clip(scarcity_metric_res1, 0, 1), # 资源1稀缺度指标
                np.clip(scarcity_metric_res2, 0, 1)  # 资源2稀缺度指标
            ]
        
        # === 2. 目标特征矩阵 [max_targets, target_feature_dim] ===
        target_features = np.zeros((max_targets, 8), dtype=np.float32)
        
        for i, target in enumerate(self.targets):
            # 归一化位置 [0, 1]
            norm_pos = np.array(target.position) / map_size
            
            # 资源比例 [0, 1]
            initial_resources = target.resources + 1e-6
            resource_ratio = target.resources / initial_resources
            
            # 归一化价值 [0, 1]（假设最大价值为1000）
            max_value = 1000.0
            norm_value = min(target.value / max_value, 1.0)
            
            # 剩余资源比例 [0, 1]
            remaining_ratio = target.remaining_resources / initial_resources
            
            # 鲁棒性掩码：is_visible位（0/1），标识目标可见性状态
            is_visible = self._calculate_target_visibility_status(target, i)
            
            target_features[i] = [
                norm_pos[0], norm_pos[1],                    # 归一化位置 (2)
                resource_ratio[0] if len(resource_ratio) > 0 else 0.0, resource_ratio[1] if len(resource_ratio) > 1 else 0.0,        # 资源比例 (2)
                norm_value,                                  # 归一化价值 (1)
                remaining_ratio[0] if len(remaining_ratio) > 0 else 0.0, remaining_ratio[1] if len(remaining_ratio) > 1 else 0.0,      # 剩余资源比例 (2)
                is_visible                                   # 可见性状态 (1)
            ]
        
        # === 3. 相对位置矩阵 [max_uavs, max_targets, 2] - 向量化操作 ===
        relative_positions = np.zeros((max_uavs, max_targets, 2), dtype=np.float32)
        
        if n_uavs > 0 and n_targets > 0:
            # 向量化计算：获取所有UAV和目标位置
            uav_positions = np.array([uav.current_position for uav in self.uavs])  # [n_uavs, 2]
            target_positions = np.array([target.position for target in self.targets])  # [n_targets, 2]
            
            # 广播计算相对位置：target_pos[None, :, :] - uav_pos[:, None, :]
            rel_pos_matrix = target_positions[None, :, :] - uav_positions[:, None, :]  # [n_uavs, n_targets, 2]
            
            # 归一化到 [-1, 1] 范围
            relative_positions[:n_uavs, :n_targets] = rel_pos_matrix / map_size
        
        # === 4. 距离矩阵 [max_uavs, max_targets] - 向量化操作 ===
        distances = np.zeros((max_uavs, max_targets), dtype=np.float32)
        
        if n_uavs > 0 and n_targets > 0:
            # 向量化计算欧几里得距离
            # 使用广播计算所有UAV-目标对的距离
            dist_matrix = np.linalg.norm(rel_pos_matrix, axis=2)  # [n_uavs, n_targets]
            
            # 归一化到 [0, 1] 范围并裁剪
            distances[:n_uavs, :n_targets] = np.clip(dist_matrix / map_size, 0.0, 1.0)
        
        # === 5. 增强的掩码字典 ===
        masks = self._calculate_robust_masks()
        
        # 构建图状态字典
        graph_state = {
            "uav_features": uav_features,
            "target_features": target_features,
            "relative_positions": relative_positions,
            "distances": distances,
            "masks": masks
        }
        
        return graph_state
    
    def _validate_graph_state(self, state: Dict[str, Any]):
        """
        验证图状态字典中的所有数组不包含NaN或无穷大值
        
        Args:
            state: 图状态字典
        """
        for key, value in state.items():
            if key == "masks":
                # 处理掩码字典
                for mask_key, mask_value in value.items():
                    if isinstance(mask_value, np.ndarray):
                        # assert not np.isnan(mask_value).any(), f"FATAL: State[{key}][{mask_key}] contains NaN!"
                        # assert np.isfinite(mask_value).all(), f"FATAL: State[{key}][{mask_key}] contains Infinity!"
                        pass
            else:
                # 处理其他数组
                if isinstance(value, np.ndarray):
                    # assert not np.isnan(value).any(), f"FATAL: State[{key}] contains NaN!"
                    # assert np.isfinite(value).all(), f"FATAL: State[{key}] contains Infinity!"
                    pass

    # def step(self, action):
    #     """执行一步动作 - 支持可选PBRS的稳定版本"""
    #     self.step_count += 1
        
    #     # === PBRS 前置计算 (保留原逻辑) ===
    #     enable_pbrs = getattr(self.config, 'ENABLE_PBRS', False)
    #     potential_before = 0.0
    #     if enable_pbrs:
    #         potential_before = self._calculate_potential()
        
    #     # 转换动作
    #     target_idx, uav_idx, phi_idx = self._action_to_assignment(action)
    #     target = self.targets[target_idx]
    #     uav = self.uavs[uav_idx]
        
    #     # 由于动作掩码已经保证了动作的有效性，这里直接计算实际贡献
    #     # 移除了原有的无效动作检查逻辑，提升训练效率
    #     actual_contribution = np.minimum(uav.resources, target.remaining_resources)
        
    #     # [新增] 调试断言，确保贡献值不为负
    #     assert np.all(actual_contribution >= 0), f"贡献值出现负数: {actual_contribution}"        

    #     # 记录目标完成前的状态
    #     was_satisfied = np.all(target.remaining_resources <= 0)
        
    #     # [新增] 记录更新前的资源，用于验证
    #     uav_res_before = uav.resources.copy()
    #     target_need_before = target.remaining_resources.copy()        
        
        
    #     # 计算路径长度
    #     path_len = np.linalg.norm(np.array(uav.current_position) - np.array(target.position))
    #     travel_time = path_len / uav.velocity_range[1] if uav.velocity_range[1] > 0 else 0.0
        
    #     # 更新状态
    #     uav.resources = uav.resources.astype(np.float64) - actual_contribution.astype(np.float64)
    #     target.remaining_resources = target.remaining_resources.astype(np.float64) - actual_contribution.astype(np.float64)
        
    #     if uav.id not in {a[0] for a in target.allocated_uavs}:
    #         target.allocated_uavs.append((uav.id, phi_idx))
    #     uav.task_sequence.append((target_idx, phi_idx))
    #     uav.current_position = np.array(target.position).copy()
    #     uav.heading = phi_idx * (2 * np.pi / self.graph.n_phi)
        
    #     # 检查是否完成所有目标
    #     total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
    #     total_targets = len(self.targets)
    #     done = bool(total_satisfied == total_targets)
        
    #     # === 可选PBRS：记录动作后势能并计算塑形奖励 ===
    #     potential_after = 0.0
    #     shaping_reward = 0.0
        
    #     if enable_pbrs:
    #         if pbrs_type == 'simple':
    #             potential_after = self._calculate_simple_potential()
    #         elif pbrs_type == 'progress':
    #             potential_after = self._calculate_progress_potential()
    #         elif pbrs_type == 'synergy':
    #             potential_after = self._calculate_potential()  # 新的协同战备势函数
    #         else:
    #             potential_after = self._calculate_potential()  # 默认使用协同战备势函数
            
    #         # 计算PBRS塑形奖励：γ * Φ(s') - Φ(s)
    #         gamma = getattr(self.config, 'GAMMA', 0.99)
    #         raw_shaping_reward = gamma * potential_after - potential_before
            
    #         # 添加数值稳定性检查和裁剪
    #         if np.isnan(raw_shaping_reward) or np.isinf(raw_shaping_reward):
    #             shaping_reward = 0.0
    #             print(f"警告: 塑形奖励为NaN/Inf，已重置为0")
    #         else:
    #             # 裁剪塑形奖励到合理范围
    #             clip_min = getattr(self.config, 'PBRS_REWARD_CLIP_MIN', -5.0)  # 更保守的裁剪范围
    #             clip_max = getattr(self.config, 'PBRS_REWARD_CLIP_MAX', 5.0)
    #             shaping_reward = np.clip(raw_shaping_reward, clip_min, clip_max)
        
    #     # 计算基础奖励 - 根据网络类型选择奖励函数
    #     network_type = getattr(self.config, 'NETWORK_TYPE', 'FCN')
    #     if network_type == 'ZeroShotGNN':
    #         # 使用双层奖励函数，提供详细的奖励分解
    #         # [修改] 将action作为参数传入，用于计算“持续探索”奖励
    #         base_reward = self._calculate_synergistic_reward(target, uav, actual_contribution, path_len, 
    #                                                        was_satisfied, travel_time, done, action)
    #     else:
    #         # 其他网络使用简单奖励函数
    #         base_reward = self._calculate_simple_reward(target, uav, actual_contribution, path_len, 
    #                                                    was_satisfied, travel_time, done)
        
    #     # 总奖励 = 基础奖励 + 塑形奖励
    #     raw_total_reward = base_reward + shaping_reward
        
    #     # 应用奖励归一化（紧急稳定性修复）
    #     total_reward = raw_total_reward
    #     if getattr(self.config, 'REWARD_NORMALIZATION', False):
    #         reward_scale = getattr(self.config, 'REWARD_SCALE', 1.0)
    #         total_reward *= reward_scale
        
    #     # 统一的最终裁剪和数值稳定性检查
    #     if np.isnan(total_reward) or np.isinf(total_reward):
    #         print(f"警告: 总奖励为NaN/Inf ({total_reward})，重置为0")
    #         total_reward = 0.0
        
    #     # 最终裁剪：扩大奖励范围，确保奖励系统平滑且平衡
    #     # 调整裁剪范围以适应大奖励值（最终成功奖励500.0 + 其他奖励）
    #     clip_min = getattr(self.config, 'REWARD_CLIP_MIN', -500.0)
    #     clip_max = getattr(self.config, 'REWARD_CLIP_MAX', 2000.0)
    #     final_total_reward = np.clip(total_reward, clip_min, clip_max)
        
    #     # 获取并更新奖励分解信息，确保一致性
    #     reward_breakdown = getattr(self, '_last_reward_breakdown', {})
        
    #     # 计算奖励处理信息
    #     reward_scale = getattr(self.config, 'REWARD_SCALE', 1.0) if getattr(self.config, 'REWARD_NORMALIZATION', False) else 1.0
    #     was_clipped = final_total_reward != total_reward
        
    #     # 更新奖励分解信息 - 增强版日志记录
    #     if reward_breakdown:
    #         # 获取Base奖励的裁剪信息
    #         base_pre_clip = reward_breakdown.get('raw_total_reward', base_reward)
    #         base_post_clip = reward_breakdown.get('base_reward', base_reward)
    #         base_was_clipped = reward_breakdown.get('was_base_clipped', False)
            
    #         reward_breakdown.update({
    #             'raw_total_reward': raw_total_reward,
    #             'pre_clip_reward': total_reward,  # 裁剪前的奖励值
    #             'post_clip_reward': final_total_reward,  # 裁剪后的奖励值
    #             'normalized_reward': total_reward,
    #             'final_total_reward': final_total_reward,
    #             'reward_scale': reward_scale,
    #             'was_clipped': was_clipped,
    #             'clip_amount': total_reward - final_total_reward if was_clipped else 0.0,  # 裁剪量
    #             'clip_min': clip_min,
    #             'clip_max': clip_max,
    #             'clip_effectiveness': 'effective' if was_clipped else 'no_clip_needed',  # 裁剪有效性
    #             # Base奖励的裁剪信息
    #             'base_pre_clip': base_pre_clip,
    #             'base_post_clip': base_post_clip,
    #             'was_base_clipped': base_was_clipped
    #         })
    #     else:
    #         # 如果没有分解信息，创建基本的处理信息
    #         reward_breakdown = {
    #             'raw_total_reward': raw_total_reward,
    #             'pre_clip_reward': total_reward,
    #             'post_clip_reward': final_total_reward,
    #             'normalized_reward': total_reward,
    #             'final_total_reward': final_total_reward,
    #             'reward_scale': reward_scale,
    #             'was_clipped': was_clipped,
    #             'clip_amount': total_reward - final_total_reward if was_clipped else 0.0,
    #             'clip_min': clip_min,
    #             'clip_max': clip_max,
    #             'clip_effectiveness': 'effective' if was_clipped else 'no_clip_needed',
    #             # Base奖励的基本信息（如果没有详细分解）
    #             'base_pre_clip': base_reward,
    #             'base_post_clip': base_reward,
    #             'was_base_clipped': False
    #         }
        
    #     # 检查是否超时
    #     truncated = self.step_count >= self.max_steps
        
    #     # 构建详细信息字典
    #     info = {
    #         'target_id': int(target_idx),
    #         'uav_id': int(uav_idx),
    #         'phi_idx': int(phi_idx),
    #         'actual_contribution': float(np.sum(actual_contribution)),
    #         'path_length': float(path_len),
    #         'travel_time': float(travel_time),
    #         'done': bool(done),
            
    #         # PBRS相关信息
    #         'pbrs_enabled': enable_pbrs,
    #         'base_reward': float(base_reward),
    #         'shaping_reward': float(shaping_reward),
    #         'potential_before': float(potential_before),
    #         'potential_after': float(potential_after),
    #         'total_reward': float(final_total_reward),  # 使用最终的总奖励
            
    #         # 详细奖励分解信息
    #         'reward_breakdown': reward_breakdown
    #     }
        
    #     # 验证奖励计算的完整一致性
    #     if getattr(self.config, 'ENABLE_REWARD_DEBUG', False):
    #         self._validate_reward_consistency(base_reward, shaping_reward, final_total_reward, reward_breakdown)
        
    #     # 保存最后一步的信息供main.py使用
    #     self._last_step_info = info
        
    #     # 统一汇报超出范围的动作数量
    #     invalid_action_count = getattr(self, '_invalid_action_count', 0)
    #     if invalid_action_count > 0:
    #         # 减少输出频率，只在以下情况输出：
    #         # 1. 每50步输出一次
    #         # 2. 无效动作数量超过阈值（比如超过5个）
    #         # 3. 在课程训练模式下，只在第一阶段输出
    #         should_output = False
            
    #         # 检查是否为课程训练模式
    #         is_curriculum = getattr(self, '_is_curriculum_training', False)
    #         current_stage = getattr(self, '_current_curriculum_stage', 'unknown')
            
    #         if self.step_count % 50 == 0:  # 每50步输出一次
    #             should_output = True
    #         elif invalid_action_count > 5:  # 无效动作数量超过阈值
    #             should_output = True
    #         elif is_curriculum and current_stage == 'easy' and self.step_count % 20 == 0:  # 课程训练第一阶段
    #             should_output = True
            
    #         if False: #should_output:  #屏蔽输出
    #             # 根据训练模式提供不同的提示信息
    #             if is_curriculum:
    #                 print(f"课程训练[{current_stage}] - 步骤{self.step_count}: 跳过{invalid_action_count}个无效动作")
    #             else:
    #                 print(f"动态训练 - 步骤{self.step_count}: 跳过{invalid_action_count}个无效动作")
            
    #         # 重置计数器
    #         self._invalid_action_count = 0
        
    #     # 奖励NaN检查已注释掉以提高训练效率
    #     # if getattr(self.config, 'debug_mode', False):
    #     #     assert not np.isnan(final_total_reward), "FATAL: Reward became NaN!"
    #     #     assert np.isfinite(final_total_reward), "FATAL: Reward became Infinity!"
        
    #     return self._get_state(), final_total_reward, done, truncated, info



    def step(self, action):
        """执行一步动作 - [已修复并加入控制台调试版本]"""
        # =================================================================
        # section 1: 初始化和动作解码
        # =================================================================
        self.step_count += 1
        
        # 分级日志输出控制
        log_level = getattr(self.config, 'LOG_LEVEL', 'simple')
        log_episode_detail = getattr(self.config, 'LOG_EPISODE_DETAIL', False)
        log_reward_detail = getattr(self.config, 'LOG_REWARD_DETAIL', False)
        log_debug_detail = getattr(self.config, 'ENABLE_DEBUG', False)
        
        # 只在启用奖励详情时输出步级别的详细信息
        if log_debug_detail:
            print("\n" + "="*80)
            print(f"--- [ENV.STEP DEBUG] Episode: (无法直接获取), UAVs:{len(self.uavs)},Targets:{len(self.targets)},Step: {self.step_count} ---")
            print(f"--- Action Received: {action} ---")
        
        try:
            target_idx, uav_idx, phi_idx = self._action_to_assignment(action)
            target = self.targets[target_idx]
            uav = self.uavs[uav_idx]
            if log_debug_detail:
                print(f"[DEBUG] Action Decoded: UAV ID={uav.id} -> Target ID={target.id}")
        except IndexError as e:
            if getattr(self.config, 'ENABLE_DEBUG', False):
                print(f"[FATAL ERROR] Action decoding failed: {e}")
            # 返回一个带有巨大惩罚的终止状态
            return self.get_state(), -500.0, True, False, {'error': 'Invalid action index'}

        # =================================================================
        # section 2: 状态快照 (执行前)
        # =================================================================
        uav_res_before = uav.resources.copy()
        target_need_before = target.remaining_resources.copy()
        
        if log_debug_detail:
            print("--- State (Before Execution) ---" 
                f"[DEBUG] UAV {uav.id} Resources: {uav_res_before}"
                f" Target {target.id} Needs: {target_need_before}")

        # =================================================================
        # section 3: 核心计算与状态更新
        # =================================================================
        # 原子性地计算资源转移向量
        resource_transfer_vector = np.minimum(uav_res_before, target_need_before)
        
        if log_debug_detail:
            print("--- Core Calculation ---"
                 f" Calculated Transfer Vector: {resource_transfer_vector}")
        
        # 增加断言，确保贡献值不为负
        assert np.all(resource_transfer_vector >= 0), f"FATAL: Negative contribution calculated: {resource_transfer_vector}"
        
        # 记录目标完成前的状态
        was_satisfied = np.all(target.remaining_resources <= 0)
        
        # 一致性地更新无人机和目标的状态
        try:
            uav.resources = (uav_res_before.astype(np.float64) - 
                            resource_transfer_vector.astype(np.float64))
            
            target.remaining_resources = (target_need_before.astype(np.float64) - 
                                        resource_transfer_vector.astype(np.float64))

            # 确保资源不会变为负数 (处理浮点数精度问题)
            uav.resources = np.maximum(uav.resources, 0)
            target.remaining_resources = np.maximum(target.remaining_resources, 0)

        except Exception as e:
            if getattr(self.config, 'ENABLE_DEBUG', False):
                print(f"[FATAL ERROR] State update failed: {e}")
            return self._get_state(), -500.0, True, False, {'error': 'State update failed'}

        # =================================================================
        # section 4: 状态快照 (执行后)
        # =================================================================
        if log_debug_detail:
            print("--- State (After Execution) ---" 
                f"[DEBUG] UAV {uav.id} Resources Now: {uav.resources}"
                f" Target {target.id} Needs Now: {target.remaining_resources}")

        # 验证更新是否正确 - 只在启用奖励详情时输出
        if log_debug_detail:
            expected_uav_res = uav_res_before - resource_transfer_vector
            expected_target_need = target_need_before - resource_transfer_vector
            if not np.allclose(uav.resources, expected_uav_res) or not np.allclose(target.remaining_resources, expected_target_need):
                print("[CRITICAL WARNING] State update mismatch detected!")
                print(f"  > Expected UAV Res: {expected_uav_res}")
                print(f"  > Expected Target Need: {expected_target_need}")

        # 其他状态更新
        path_len = np.linalg.norm(np.array(uav.current_position) - np.array(target.position))
        travel_time = path_len / uav.velocity_range[1] if uav.velocity_range[1] > 0 else 0.0
        
        if uav.id not in {a[0] for a in target.allocated_uavs}:
            target.allocated_uavs.append((uav.id, phi_idx))
            
            # 【修改后的代码】更新目标的"在途"资源量
            # 我们假设一旦派遣，无人机会承诺其能提供的所有相关资源
            potential_contribution = np.minimum(uav.resources, target.remaining_resources)
            target.in_flight_resources += potential_contribution
            
        # 【新增代码】在动作执行后，立即更新无人机的当前位置为目标位置
        # 这是修复无效飞行Bug的关键，确保下一步决策基于正确的位置状态
        uav.current_position = np.array(target.position).copy()
        
        uav.task_sequence.append((target_idx, phi_idx))        
        uav.heading = phi_idx * (2 * np.pi / self.graph.n_phi)

        # =================================================================
        # section 5: 奖励计算与返回
        # =================================================================
        total_satisfied = sum(np.all(t.remaining_resources <= 0) for t in self.targets)
        done = bool(total_satisfied == len(self.targets))
        truncated = self.step_count >= self.max_steps
        
        # PBRS & Base Reward (这里使用正确的 resource_transfer_vector 作为贡献)
        actual_contribution = resource_transfer_vector
        
        enable_pbrs = getattr(self.config, 'ENABLE_PBRS', False)
        shaping_reward = 0.0
        if enable_pbrs:
            # PBRS 计算需要 pre 和 post 状态, 这里简化
            pass
        
        completion_rate_after = self.get_completion_rate() # 获取当前完成率
        base_reward = self._calculate_synergistic_reward(target, uav, actual_contribution, path_len, 
                                                    was_satisfied, travel_time, done, action,
                                                    self.step_count, completion_rate_after) # <--- 传入参数
        
        total_reward = base_reward + shaping_reward
        final_total_reward = np.clip(total_reward, 
                                    getattr(self.config, 'REWARD_CLIP_MIN', -500.0),
                                    getattr(self.config, 'REWARD_CLIP_MAX', 2000.0))
        
        if getattr(self.config, 'ENABLE_DEBUG', True):
            print("--- Reward & Termination ---" 
                f"[DEBUG] Base Reward: {base_reward:.2f}, Shaping Reward: {shaping_reward:.2f}"
                f" Final Total Reward: {final_total_reward:.2f}"
                f" Done: {done}, Truncated: {truncated}")
            print("="*80 + "\n")
        
        # 获取详细的奖励分解信息
        reward_breakdown = getattr(self, '_last_reward_breakdown', {})
        
        info = {
            'target_id': int(target_idx),
            'uav_id': int(uav_idx),
            'phi_idx': int(phi_idx),
            'actual_contribution': float(np.sum(actual_contribution)),
            'path_length': float(path_len),
            'travel_time': float(travel_time),
            'done': bool(done),
            'base_reward': float(base_reward),
            'shaping_reward': float(shaping_reward),
            'total_reward': float(final_total_reward),
            'reward_breakdown': reward_breakdown
        }
        
        return self._get_state(), final_total_reward, done, truncated, info


    def _validate_reward_consistency(self, base_reward, shaping_reward, final_total_reward, reward_breakdown):
        """
        验证奖励计算的完整一致性
        
        Args:
            base_reward: 基础奖励
            shaping_reward: 塑形奖励
            final_total_reward: 最终总奖励
            reward_breakdown: 奖励分解信息
        """
        try:
            # 1. 验证基础奖励与分解的一致性
            if 'layer1_total' in reward_breakdown and 'layer2_total' in reward_breakdown:
                # 双层奖励系统
                layer1_total = reward_breakdown['layer1_total']
                layer2_total = reward_breakdown['layer2_total']
                extra_rewards_sum = reward_breakdown.get('extra_rewards_sum', 0.0)
                expected_base = layer1_total + layer2_total + extra_rewards_sum
                
                base_diff = abs(base_reward - expected_base)
                if base_diff > 1e-6:
                    if getattr(self.config, 'ENABLE_DEBUG', True):
                        print(f"⚠️ [VALIDATION] 双层基础奖励不一致! 期望: {expected_base:.6f}, 实际: {base_reward:.6f}, 差异: {base_diff:.6f}")
                
                # 验证第一层分解
                layer1_breakdown = reward_breakdown.get('layer1_breakdown', {})
                if layer1_breakdown:
                    layer1_sum = sum(layer1_breakdown.values())
                    layer1_diff = abs(layer1_total - layer1_sum)
                    if layer1_diff > 1e-6:
                        if getattr(self.config, 'ENABLE_DEBUG', True):
                            print(f"⚠️ [VALIDATION] 第一层分解不一致! 期望: {layer1_total:.6f}, 分解总和: {layer1_sum:.6f}, 差异: {layer1_diff:.6f}")
                
                # 验证第二层分解
                layer2_breakdown = reward_breakdown.get('layer2_breakdown', {})
                if layer2_breakdown:
                    layer2_sum = sum(layer2_breakdown.values())
                    layer2_diff = abs(layer2_total - layer2_sum)
                    if layer2_diff > 1e-6:
                        if getattr(self.config, 'ENABLE_DEBUG', True):
                            print(f"⚠️ [VALIDATION] 第二层分解不一致! 期望: {layer2_total:.6f}, 分解总和: {layer2_sum:.6f}, 差异: {layer2_diff:.6f}")
            
            elif 'simple_breakdown' in reward_breakdown:
                # 简单奖励系统
                simple_breakdown = reward_breakdown['simple_breakdown']
                breakdown_sum = sum(simple_breakdown.values())
                base_diff = abs(base_reward - breakdown_sum)
                if base_diff > 1e-6:
                    if getattr(self.config, 'ENABLE_DEBUG', True):
                        print(f"⚠️ [VALIDATION] 简单基础奖励不一致! 期望: {breakdown_sum:.6f}, 实际: {base_reward:.6f}, 差异: {base_diff:.6f}")
            
            # 2. 验证总奖励计算的一致性
            raw_total = reward_breakdown.get('raw_total_reward', base_reward + shaping_reward)
            normalized_reward = reward_breakdown.get('normalized_reward', raw_total)
            
            # 考虑归一化的影响
            reward_scale = reward_breakdown.get('reward_scale', 1.0)
            expected_normalized = raw_total * reward_scale
            norm_diff = abs(normalized_reward - expected_normalized)
            if getattr(self.config, 'ENABLE_DEBUG', True) and norm_diff > 1e-6:
                print(f"⚠️ [VALIDATION] 归一化计算不一致! 期望: {expected_normalized:.6f}, 实际: {normalized_reward:.6f}, 差异: {norm_diff:.6f}")
            
            # 3. 验证最终裁剪的一致性
            was_clipped = reward_breakdown.get('was_clipped', False)
            if was_clipped:
                clip_min = reward_breakdown.get('clip_min', -500.0)
                clip_max = reward_breakdown.get('clip_max', 2000.0)
                expected_clipped = np.clip(normalized_reward, clip_min, clip_max)
                clip_diff = abs(final_total_reward - expected_clipped)
                if clip_diff > 1e-6:
                    print(f"⚠️ [VALIDATION] 裁剪计算不一致! 期望: {expected_clipped:.6f}, 实际: {final_total_reward:.6f}, 差异: {clip_diff:.6f}")
            
        except Exception as e:
            print(f"⚠️ [VALIDATION] 奖励一致性验证异常: {e}")

    def _calculate_simple_reward(self, target, uav, actual_contribution, path_len, 
                                was_satisfied, travel_time, done):
        """
        优化奖励函数 - 首要满足资源需求，其次路径最短，输出详细奖励分解
        """
        # 初始化奖励分解字典
        breakdown = {
            '最终成功': 0.0,
            '目标完成': 0.0,
            '基础贡献': 0.0,
            '比例奖励': 0.0,
            '效率奖励': 0.0,
            '路径成本': 0.0
        }
        
        # 1. 最终成功的巨大奖励
        if done:
            breakdown['最终成功'] = 100.0
        
        # 2. 单个目标完成奖励
        now_satisfied = np.all(target.remaining_resources <= 0)
        if now_satisfied and not was_satisfied:
            breakdown['目标完成'] = 30.0
        
        # 3. 资源贡献奖励
        contribution_amount = np.sum(actual_contribution)
        if contribution_amount > 0:
            # 基础贡献奖励
            breakdown['基础贡献'] = 5.0
            
            # 贡献量比例奖励
            target_total_need = np.sum(target.resources)
            if target_total_need > 0:
                contribution_ratio = contribution_amount / target_total_need
                breakdown['比例奖励'] = 10.0 * contribution_ratio
            
            # 资源匹配奖励
            uav_total_resources = np.sum(uav.resources) + contribution_amount
            if uav_total_resources > 0:
                efficiency_ratio = contribution_amount / uav_total_resources
                breakdown['效率奖励'] = 5.0 * efficiency_ratio
        
        # 4. 路径成本
        if contribution_amount > 0:
            breakdown['路径成本'] = -0.002 * path_len
        
        # 计算总奖励
        raw_total_reward = sum(breakdown.values())
        
        # 验证分解总和的一致性
        breakdown_sum = sum(breakdown.values())
        breakdown_diff = abs(raw_total_reward - breakdown_sum)
        
        if breakdown_diff > 1e-6:
            print(f"⚠️ [DEBUG] 简单奖励分解总和不一致! 差异: {breakdown_diff:.6f}")
            raw_total_reward = breakdown_sum  # 使用分解总和确保一致性
        
        # 对Base_Reward进行温和裁剪，确保其值不会过大
        # 调整裁剪范围以适应更大的奖励值
        clip_min = getattr(self.config, 'BASE_REWARD_CLIP_MIN', -200.0)
        clip_max = getattr(self.config, 'BASE_REWARD_CLIP_MAX', 600.0)
        clipped_reward = np.clip(raw_total_reward, clip_min, clip_max)

        # 数值稳定性检查
        if np.isnan(clipped_reward) or np.isinf(clipped_reward):
            clipped_reward = 0.0
        
        # 确保返回值为有限数值
        final_reward = float(clipped_reward)
        if not np.isfinite(final_reward):
            final_reward = 0.0
        
        # 保存详细的奖励分解信息用于分析
        self._last_reward_breakdown = {
            'simple_breakdown': breakdown,
            'breakdown_sum': breakdown_sum,
            'raw_total_reward': raw_total_reward,
            'clipped_reward': clipped_reward,
            'base_reward': final_reward,
            'contribution_amount': contribution_amount,
            'target_satisfied': now_satisfied and not was_satisfied,
            'final_success': done,
            'was_base_clipped': clipped_reward != raw_total_reward,
            'base_clip_min': clip_min,
            'base_clip_max': clip_max,
            'breakdown_consistent': breakdown_diff <= 1e-6
        }
        
        return final_reward
    def _calculate_synergistic_reward(self, target, uav, actual_contribution, path_len, 
                                        was_satisfied, travel_time, done, action,
                                        step_count: int, completion_rate: float): # <--- 新增接收参数
        """
        [双层奖励方案] - 优先满足资源匹配，再进行优化选择
        
        第一层：资源匹配奖励 - 确保无人机资源与目标需求匹配
        第二层：优化选择奖励 - 在满足匹配的基础上进行路径、协同等优化
        
        Args:
            target: 目标对象
            uav: UAV对象
            actual_contribution: 实际贡献的资源
            path_len: 路径长度
            was_satisfied: 目标之前是否已满足
            travel_time: 旅行时间
            done: 是否完成
            
        Returns:
            float: 最终的Base_Reward值
        """
        
        # 启用调试模式
        debug_mode = getattr(self.config, 'ENABLE_REWARD_DEBUG', False)
        
        try:
            # ========== 第一层：资源匹配奖励 ==========
            layer1_reward, layer1_breakdown = self._calculate_layer1_matching_reward(target, uav, actual_contribution)
            
            # ========== 第二层：优化选择奖励 ==========
            layer2_reward, layer2_breakdown = self._calculate_layer2_optimization_reward(
                target, uav, actual_contribution, path_len, was_satisfied, travel_time, done,
                step_count, completion_rate) # <--- 传递参数

            # 检查子奖励的数值有效性
            if np.isnan(layer1_reward) or np.isinf(layer1_reward):
                if debug_mode:
                    print("⚠️ [DEBUG] layer1_reward出现数值问题，已重置为0")
                layer1_reward = 0.0
            
            if np.isnan(layer2_reward) or np.isinf(layer2_reward):
                if debug_mode:
                    print("⚠️ [DEBUG] layer2_reward出现数值问题，已重置为0")
                layer2_reward = 0.0

            # 将双层奖励合成作为Base_Reward
            base_reward = layer1_reward + layer2_reward
            
            # 初始化额外奖励分解
            extra_rewards = {
                '最终成功奖励': 0.0,
                '接近完成奖励': 0.0,
                '回合结束惩罚': 0.0,
                '持续探索奖励': 0.0  # [新增] 初始化持续探索奖励
            }
            # [新增] 如果模型做出了有实际贡献的动作，则给予持续探索奖励
            if np.sum(actual_contribution) > 0:
                extra_rewards['持续探索奖励'] = 0.5  # 给予一个小的正向激励

            # === 最终成功奖励 - 直接计入Base_Reward ===
            all_targets_satisfied = all(np.all(t.remaining_resources <= 0) for t in self.targets)
            if done and all_targets_satisfied:
                final_success_bonus = 150.0  # 最终成功奖励
                base_reward += final_success_bonus
                extra_rewards['最终成功奖励'] = final_success_bonus
                if debug_mode:
                    print(f"🏆 [DEBUG] 最终成功奖励: {final_success_bonus}")
            
            # === 90%完成时给予部分最终奖励 - 直接计入Base_Reward ===
            # 使用标准完成率计算方法
            completion_ratio = self._calculate_standard_completion_rate()
            
            if completion_ratio >= 0.9 and not (done and all_targets_satisfied):
                near_completion_bonus = 100.0 * (completion_ratio - 0.9) / 0.1
                base_reward += near_completion_bonus
                extra_rewards['接近完成奖励'] = near_completion_bonus
                if debug_mode:
                    print(f"🎯 [DEBUG] 接近完成奖励: {near_completion_bonus:.1f} (完成率: {completion_ratio:.2%})")
            
            # === 回合结束惩罚 - 直接计入Base_Reward ===
            truncated = self.step_count >= self.max_steps
            if truncated and not all_targets_satisfied:
                num_unmet_targets = len([t for t in self.targets if np.any(t.remaining_resources > 0)])
                final_incompletion_penalty = -200.0 * num_unmet_targets
                base_reward += final_incompletion_penalty
                extra_rewards['回合结束惩罚'] = final_incompletion_penalty
            
            # 验证基础奖励计算的一致性
            expected_base_reward = layer1_reward + layer2_reward + sum(extra_rewards.values())
            base_reward_diff = abs(base_reward - expected_base_reward)
            
            if base_reward_diff > 1e-6:
                if debug_mode:
                    print(f"⚠️ [DEBUG] 基础奖励计算不一致! 期望: {expected_base_reward:.6f}, 实际: {base_reward:.6f}, 差异: {base_reward_diff:.6f}")
                # 修正基础奖励以确保一致性
                base_reward = expected_base_reward
            
            # 保存详细的奖励分解信息用于分析
            self._last_reward_breakdown = {
                'layer1_total': layer1_reward,
                'layer1_breakdown': layer1_breakdown,
                'layer2_total': layer2_reward,
                'layer2_breakdown': layer2_breakdown,
                'extra_rewards': extra_rewards,
                'extra_rewards_sum': sum(extra_rewards.values()),
                'base_reward': base_reward,
                'raw_layer_sum': layer1_reward + layer2_reward,
                'expected_base_reward': expected_base_reward,
                'base_reward_consistent': base_reward_diff <= 1e-6
            }
            
            # 确保返回值为有限数值，但不进行裁剪（由step函数统一处理）
            if np.isnan(base_reward) or np.isinf(base_reward):
                base_reward = 0.0
                if debug_mode:
                    print(f"⚠️ [DEBUG] base_reward非有限值，重置为0")
            
            return float(base_reward)
        
        except Exception as e:
            if debug_mode:
                print(f"⚠️ [DEBUG] 奖励计算异常: {e}")
            return 0.0   

    def _calculate_resource_waste_penalty(self, breakdown: Dict[str, Any], step_count: int, completion_rate: float) -> float:
        """
        计算资源浪费惩罚
        
        当存在空闲资源和未满足需求同时存在时进行扣分，
        激励模型主动解决这一问题
        
        Returns:
            float: 资源浪费惩罚值（负数）
        """
        penalty = 0.0
        
        # 条件1: 仅在任务完成率低于90%时考虑此惩罚
        # 条件2: 仅在轮次的前10步考虑此惩罚
        if completion_rate >= 0.9 or step_count >= 10:
            breakdown['资源浪费'] = 0.0
            return 0.0

        # 统计空闲UAV（有资源但未分配任务）
        idle_uavs = []
        for uav in self.uavs:
            if np.any(uav.resources > 0):  # UAV有资源
                # 检查是否有分配的任务
                has_tasks = any(uav.id in [uav_info[0] for uav_info in target.allocated_uavs] 
                              for target in self.targets)
                if not has_tasks:
                    idle_uavs.append(uav)
        
        # 统计未满足的目标需求
        unmet_targets = []
        for target in self.targets:
            if np.any(target.remaining_resources > 0):  # 目标还有需求
                unmet_targets.append(target)
        
        # 如果同时存在空闲资源和未满足需求，施加惩罚
        if len(idle_uavs) > 0 and len(unmet_targets) > 0:
            # 计算可匹配的资源-需求对数量
            matchable_pairs = 0
            for uav in idle_uavs:
                for target in unmet_targets:
                    # 检查UAV是否能满足目标的某种资源需求
                    for phi_idx in range(len(uav.resources)):
                        if (uav.resources[phi_idx] > 0 and 
                            target.remaining_resources[phi_idx] > 0):
                            matchable_pairs += 1
                            break  # 每个UAV-目标对只计算一次
            
            # 基于可匹配对数量计算惩罚
            if matchable_pairs > 0:
                # 增强惩罚强度，确保不被其他奖励抵消
                waste_severity = min(matchable_pairs, 8)  # 限制最大惩罚
                penalty = -0.5 * waste_severity  # 
                
                # 调试信息
                if getattr(self.config, 'ENABLE_REWARD_DEBUG', False):
                    print(f"⚠️ [DEBUG] 资源浪费检测: {len(idle_uavs)}个空闲UAV, "
                          f"{len(unmet_targets)}个未满足目标, "
                          f"{matchable_pairs}个可匹配对, 惩罚={penalty:.1f}")
        
        return penalty
    
    def _calculate_layer1_matching_reward(self, target, uav, actual_contribution):
        """
        第一层：资源匹配奖励 - 半稀疏奖励设计
        
        调整为半稀疏奖励，减小各项权重，避免压倒最终成功奖励
        
        Args:
            target: 目标对象
            uav: UAV对象
            actual_contribution: 实际贡献的资源
            
        Returns:
            tuple: (总奖励, 奖励分解字典)
        """
        contribution_amount = np.sum(actual_contribution)
        # [核心修正] 如果动作没有产生任何实际贡献，则直接返回一个明确的惩罚
        if contribution_amount <= 0:
            return -1.0, {'零贡献惩罚': -1.0}

        # 初始化奖励分解字典 - 调整权重，增加详细分解
        breakdown = {
            '基础匹配_固定奖励': 20.0,     # 提高基础匹配奖励，密集权重80
            '需求满足_贡献比': 0.0,        # 根据满足程度给予奖励
            '需求满足_满足率': 0.0,        # 满足率计算
            '类型匹配_效率': 0.0,          # 根据类型匹配给予奖励
            '类型匹配_匹配数': 0.0,        # 匹配的资源类型数量
            '紧急度_剩余比例': 0.0,        # 根据紧急程度给予奖励
            '探索奖励_固定': 5.0           # 提高探索奖励
        }
        

        
        # 2. 需求满足度奖励 - 提高到30.0，成为驱动模型完成任务的核心
        target_total_need = np.sum(target.resources)
        if target_total_need > 0:
            satisfaction_ratio = contribution_amount / target_total_need
            # 提高需求满足奖励权重
            breakdown['需求满足_贡献比'] = 20.0 * satisfaction_ratio
            breakdown['需求满足_满足率'] = satisfaction_ratio
        
        # 3. 资源类型匹配奖励 - 提高到10.0，鼓励更精准的资源分配
        match_count = 0
        total_efficiency = 0.0
        for i, contrib in enumerate(actual_contribution):
            if contrib > 0 and target.remaining_resources[i] > 0:
                # 精确匹配给予更高奖励
                match_efficiency = min(contrib, target.remaining_resources[i]) / contrib
                total_efficiency += match_efficiency
                match_count += 1
        
        if match_count > 0:
            breakdown['类型匹配_效率'] = 10.0 * (total_efficiency / match_count)
            breakdown['类型匹配_匹配数'] = match_count
        
        # 4. 紧急度奖励 - 提高到15.0
        target_urgency = np.sum(target.remaining_resources) / (np.sum(target.resources) + 1e-6)
        breakdown['紧急度_剩余比例'] = 15.0 * target_urgency
        
        # 将路径效率拆分为更详细的子项
        if '路径效率' in breakdown:
            path_efficiency_base = breakdown.get('路径效率_贡献比', 0.0)
            path_efficiency_penalty = breakdown.get('路径效率_长度惩罚', 0.0)
            breakdown['路径效率_贡献比'] = path_efficiency_base
            breakdown['路径效率_长度惩罚'] = path_efficiency_penalty
            # 移除原始的路径效率项
            if '路径效率' in breakdown:
                del breakdown['路径效率']
        
        # 检查breakdown字典中的异常值并打印警告
        for key, value in breakdown.items():
            if np.isnan(value) or np.isinf(value):
                print(f"[REWARD DEBUG] 警告: Layer1奖励项 '{key}' 出现异常值 {value}，已重置为0")
                breakdown[key] = 0.0
        
        layer1_total = sum(breakdown.values())
        
        return layer1_total, breakdown
    
    def _calculate_approach_reward(self, target, uav, previous_distance, current_distance):
        """
        计算接近奖励
        
        Args:
            target: 目标对象
            uav: 无人机对象
            previous_distance: 之前的距离
            current_distance: 当前距离
            
        Returns:
            float: 接近奖励值
        """
        try:
            # 计算距离缩短量
            distance_reduction = max(0.0, previous_distance - current_distance)
            
            # 应用奖励系数
            approach_reward = self.config.APPROACH_REWARD_COEFFICIENT * distance_reduction
            
            # 数值范围检查和截断
            if np.isnan(approach_reward) or np.isinf(approach_reward):
                approach_reward = 0.0
            else:
                # 限制奖励在合理范围内
                approach_reward = np.clip(approach_reward, 0.0, 1.0)
            
            return float(approach_reward)
            
        except Exception as e:
            print(f"警告: 接近奖励计算异常: {e}")
            return 0.0
    
    def _has_required_resources(self, target, uav):
        """
        检查无人机是否拥有目标所需的资源类型
        
        Args:
            target: 目标对象
            uav: 无人机对象
            
        Returns:
            bool: 是否拥有所需资源
        """
        try:
            # 检查无人机资源与目标需求资源的匹配性
            for i in range(len(target.remaining_resources)):
                if target.remaining_resources[i] > 0 and uav.resources[i] > 0:
                    return True
            return False
        except Exception as e:
            print(f"警告: 资源匹配检查异常: {e}")
            return False
    
    def _calculate_layer2_optimization_reward(self, target, uav, actual_contribution, 
                                            path_len, was_satisfied, travel_time, done,
                                            step_count: int, completion_rate: float): # <--- 新增接收参数
        """
        第二层：优化选择奖励 - 协同占比调整到20%左右
        
        在资源匹配的基础上，优化路径效率、协同作战等高级策略
        
        Args:
            target: 目标对象
            uav: UAV对象
            actual_contribution: 实际贡献的资源
            path_len: 路径长度
            was_satisfied: 目标之前是否已满足
            travel_time: 飞行时间
            done: 是否完成
            
        Returns:
            tuple: (总奖励, 奖励分解字典)
        """
        contribution_amount = np.sum(actual_contribution)
        
        # 初始化奖励分解字典
        breakdown = {
            # '任务完成': 0.0,
            '伤害贡献奖励': 0.0,
            '协同增效': 0.0,  # 调整到20%左右占比
            '路径效率': 0.0,
            '时间效率': 0.0,
            '资源浪费': 0.0,
            '进度奖励': 0.0,
            '接近奖励': 0.0,      # 新增接近奖励组件
            '战略损失': 0.0,
            '资源均衡': 0.0,  # <--- 新增奖励项
            '时间惩罚': -0.1  # [新增] 引入时间惩罚, 鼓励模型尽快完成任务
        }
        
        # # 1. 任务完成奖励 - 适度提高基础完成奖励
        # now_satisfied = np.all(target.remaining_resources <= 0)
        # new_satisfied = now_satisfied and not was_satisfied
        
        # if new_satisfied:
        #     # 任务完成奖励调整到30.0
        #     breakdown['任务完成'] = 80.0
            
        #     # 协同增效检测和奖励 - 与任务完成结合
        #     participating_uav_ids = {uav_info[0] for uav_info in target.allocated_uavs}
        #     if len(participating_uav_ids) > 1:
        #         # 协同增效奖励调整到30.0，与任务完成结合
        #         # 协同完成任务的总奖励约60.0分，远高于单机完成的30.0分
        #         base_collaboration_reward = 60.0
        #         synergy_bonus = base_collaboration_reward * min(1.0, len(participating_uav_ids) / 3.0)
        #         breakdown['协同增效'] = synergy_bonus
        #         # print(f"🎯 协同增效攻击成功！目标 {target.id} 被摧毁，参与UAV: {participating_uav_ids}，获得奖励 {synergy_bonus}")
                # [新增] 1. 伤害贡献奖励 (替代原有的任务完成奖励)
        # 1. 伤害贡献奖励 (替代原有的任务完成奖励)
        # 只要有实际贡献，就根据贡献占目标总需求的比例给予平滑奖励
        if contribution_amount > 0:
            target_total_need = np.sum(target.resources)
            if target_total_need > 0:
                damage_ratio = contribution_amount / target_total_need
                # 将原有的80分奖励平滑地分配到每一步的伤害贡献中
                damage_reward = 80.0 * damage_ratio
                breakdown['伤害贡献奖励'] = damage_reward

        # [修改] 2. 协同增效奖励 - 依然在目标完成时触发，但不再与任务完成奖励绑定
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = now_satisfied and not was_satisfied

        if new_satisfied:
            # 协同增效检测和奖励
            participating_uav_ids = {uav_info[0] for uav_info in target.allocated_uavs}
            if len(participating_uav_ids) > 1:
                base_collaboration_reward = 60.0
                synergy_bonus = base_collaboration_reward * min(1.0, len(participating_uav_ids) / 3.0)
                breakdown['协同增效'] = synergy_bonus

        # 2. 路径效率奖励 - 调整权重并拆分为详细子项
        if contribution_amount > 0:
            # 路径效率：贡献量与路径长度的比值
            efficiency_ratio = contribution_amount / (path_len + 1e-6)
            path_efficiency_base = 2.0 * min(efficiency_ratio, 1.0)  # 调整基础权重
            
            # 路径惩罚 - 适度惩罚过长路径
            path_penalty = -0.001 * path_len  # 调整惩罚权重
            
            # 拆分为详细子项
            breakdown['路径效率_贡献比'] = path_efficiency_base
            breakdown['路径效率_长度惩罚'] = path_penalty
        
        # 3. 时间效率奖励 - 调整权重
        if contribution_amount > 0 and travel_time > 0:
            # 奖励快速到达
            time_bonus = max(0, 5.0 - travel_time)  # 调整时间奖励
            breakdown['时间效率'] = time_bonus * 0.5  # 调整权重系数
        
        # 4. 资源浪费惩罚 - 调整权重        
        waste_penalty = self._calculate_resource_waste_penalty(breakdown, step_count, completion_rate) # <--- 传递参数
        breakdown['资源浪费'] = waste_penalty * 2.0  # 保持原有强度
        
        # 5. 进度奖励 - 调整权重
        if contribution_amount > 0:
            target_progress = 1.0 - (np.sum(target.remaining_resources) / np.sum(target.resources))
            breakdown['进度奖励'] = 3.0 * target_progress  # 提高进度奖励权重
        
        # 6. 接近奖励 - 新增组件
        if contribution_amount > 0 and self._has_required_resources(target, uav):
            # 计算无人机与目标之间的距离变化
            current_distance = np.linalg.norm(np.array(uav.current_position) - np.array(target.position))
            
            # 获取之前的位置（如果存在）
            if hasattr(uav, 'previous_position') and uav.previous_position is not None:
                previous_distance = np.linalg.norm(np.array(uav.previous_position) - np.array(target.position))
                approach_reward = self._calculate_approach_reward(target, uav, previous_distance, current_distance)
                breakdown['接近奖励'] = approach_reward
        
        # 7. 计算战略损失惩罚   
        strategic_loss_penalty = self._calculate_critical_uav_loss_penalty(uav, target)
        breakdown['战略损失'] = strategic_loss_penalty

        # 8. 资源均衡奖励 - 新增组件
        # [新增] 计算资源均衡奖励/惩罚
        if np.sum(actual_contribution) > 0:
            resources_before = uav.resources + actual_contribution
            resources_after = uav.resources
            
            # 使用标准差来衡量均衡性，标准差越小越均衡
            std_before = np.std(resources_before / (np.sum(resources_before) + 1e-6))
            std_after = np.std(resources_after / (np.sum(resources_after) + 1e-6))
            
            # 如果动作让资源分布更均衡（标准差变小），给予奖励
            balance_change = std_before - std_after
            # 将变化量缩放到一个小的奖励范围内
            RESOURCE_BALANCE_WEIGHT  = 5
            breakdown['资源均衡'] = np.clip(balance_change * RESOURCE_BALANCE_WEIGHT, -2.5, 2.5)        

        # 检查breakdown字典中的异常值并打印警告
        for key, value in breakdown.items():
            if np.isnan(value) or np.isinf(value):
                print(f"[REWARD DEBUG] 警告: Layer2奖励项 '{key}' 出现异常值 {value}，已重置为0")
                breakdown[key] = 0.0
        
        layer2_total = sum(breakdown.values())
        
        return layer2_total, breakdown
    def _calculate_critical_uav_loss_penalty(self, uav_taken_action, target_taken_action):
            """
            [修改后] 计算因本次动作导致“关键无人机”损失的惩罚。
            “关键无人机”是指能满足某个未来需求的无人机数量下降到危险阈值。
            """
            
            # 从Config中读取基础阈值和缩放因子，增加灵活性
            MIN_THRESHOLD = getattr(self.config, 'CRITICAL_UAV_MIN_THRESHOLD', 2)
            SCALING_FACTOR = getattr(self.config, 'CRITICAL_UAV_SCALING_FACTOR', 0.15)
            total_uavs = len(self.uavs)
            dynamic_threshold = max(MIN_THRESHOLD, round(total_uavs * SCALING_FACTOR))        

            CRITICAL_UAV_THRESHOLD = 2 
            penalty = 0.0
            
            for other_target in self.targets:
                if other_target.id != target_taken_action.id and np.any(other_target.remaining_resources > 0):
                    
                    # 统计动作前，有多少无人机能满足'other_target'
                    capable_uavs_before_action = [
                        u for u in self.uavs if self._has_actual_contribution(other_target, u)
                    ]
                    
                    # [核心修改] 当可用无人机数量处于或即将进入危险水平时
                    if len(capable_uavs_before_action) <= dynamic_threshold:
                        
                        # 检查当前动作涉及的无人机是否是这些关键无人机之一
                        is_critical_uav_involved = any(uav.id == uav_taken_action.id for uav in capable_uavs_before_action)
                        
                        
                        if is_critical_uav_involved:
                            # 模拟动作后的资源状态
                            contribution = np.minimum(uav_taken_action.resources, target_taken_action.remaining_resources)
                            resources_after = uav_taken_action.resources - contribution
                            
                            # 检查动作后，这架无人机是否还具备完成'other_target'的能力
                            from entities import UAV
                            temp_uav_state = UAV(id=uav_taken_action.id, position=[0,0], heading=0, resources=resources_after, max_distance=0, velocity_range=(0,0), economic_speed=0)
                            
                            # 如果动作导致这架关键无人机失去了能力
                            if not self._has_actual_contribution(other_target, temp_uav_state):
                                # 施加预警式惩罚
                                penalty -= 25.0  # 惩罚值可以调整
                                print(f"!!! 战略警告: 动作 UAV({uav_taken_action.id}) -> Tgt({target_taken_action.id}) 削弱了未来应对 Tgt({other_target.id}) 的能力 !!!")

            return penalty
    def _initialize_entities(self, scenario_name='medium'):
        """
        根据场景名称模板，从config中读取配置并随机生成实体。
        这是动态随机场景模式的核心。
        
        优化版本：添加位置去重机制，确保场景多样性，避免无人机和目标位置重复
        """
        # --- 第一部分：从config中获取模板，确定实体数量和参数 ---
        
        # [新增] 验证场景名称
        if scenario_name not in self.config.SCENARIO_TEMPLATES:
            print(f"警告: 场景名称 '{scenario_name}' 不存在，使用medium场景")
            scenario_name = 'medium'
        
        template = self.config.SCENARIO_TEMPLATES[scenario_name]
        if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
            print(f"[DEBUG] 使用场景模板: {scenario_name} -> {template}")
        
        # 生成唯一的随机种子，确保每次调用都产生不同的场景
        import time
        unique_seed = int(time.time() * 1000000) % 2147483647  # 使用微秒时间戳生成种子
        np.random.seed(unique_seed)
        if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
            print(f"[DEBUG] 随机种子: {unique_seed}")
        
        # 严格按照模板生成数量，确保不超出限制（修复：包含上限值）
        uav_num = np.random.randint(template['uav_num_range'][0], template['uav_num_range'][1] + 1)
        target_num = np.random.randint(template['target_num_range'][0], template['target_num_range'][1] + 1)
        obstacle_num = np.random.randint(template['obstacle_num_range'][0], template['obstacle_num_range'][1] + 1)
        resource_abundance = np.random.uniform(*template['resource_abundance_range'])
        
        # [新增] 严格验证生成的数量是否符合模板约束
        uav_range = template['uav_num_range']
        target_range = template['target_num_range']
        
        if not (uav_range[0] <= uav_num <= uav_range[1]):
            print(f"错误: 生成的UAV数量 {uav_num} 超出场景 {scenario_name} 的范围 {uav_range}")
            uav_num = np.clip(uav_num, uav_range[0], uav_range[1])
            print(f"已修正UAV数量为: {uav_num}")
            
        if not (target_range[0] <= target_num <= target_range[1]):
            print(f"错误: 生成的目标数量 {target_num} 超出场景 {scenario_name} 的范围 {target_range}")
            target_num = np.clip(target_num, target_range[0], target_range[1])
            print(f"已修正目标数量为: {target_num}")
        
        # if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
        #     print(f"[DEBUG] 最终生成数量: UAV={uav_num}, Target={target_num}, Obstacle={obstacle_num}")
        
        # 添加约束验证
        if self.config.VALIDATE_SCENARIO_CONSTRAINTS:
            # 确保不超过最大限制
            if uav_num > self.config.MAX_UAVS:
                print(f"警告: UAV数量 ({uav_num}) 超过MAX_UAVS ({self.config.MAX_UAVS})，已调整")
                uav_num = self.config.MAX_UAVS
            if target_num > self.config.MAX_TARGETS:
                print(f"警告: 目标数量 ({target_num}) 超过MAX_TARGETS ({self.config.MAX_TARGETS})，已调整")
                target_num = self.config.MAX_TARGETS
        
        # [新增] 最终验证：确保生成的数量完全符合场景约束
        final_uav_range = template['uav_num_range']
        final_target_range = template['target_num_range']
        
        if not (final_uav_range[0] <= uav_num <= final_uav_range[1]):
            print(f"严重错误: 最终UAV数量 {uav_num} 仍超出 {scenario_name} 场景约束 {final_uav_range}")
            uav_num = final_uav_range[0]  # 使用最小安全值
            
        if not (final_target_range[0] <= target_num <= final_target_range[1]):
            print(f"严重错误: 最终目标数量 {target_num} 仍超出 {scenario_name} 场景约束 {final_target_range}")
            target_num = final_target_range[0]  # 使用最小安全值
        
        # 场景生成信息已移除，不在控制台显示
        # print(f"生成{scenario_name}场景: {uav_num}架UAV, {target_num}个目标, {obstacle_num}个障碍物")
        
        # --- 第二部分：根据参数，随机生成实体 ---        
        
        self.uavs = []
        self.targets = []
        self.obstacles = []

        map_size = self.config.MAP_SIZE
        num_resource_types = self.config.RESOURCE_DIM

        # 1. 随机生成目标 - 优化版本，添加位置去重机制，确保空间分布合理
        total_demand = np.zeros(num_resource_types, dtype=int)
        occupied_positions = []  # 记录已占用的位置
        min_distance_between_targets = map_size * 0.1  # 目标间最小距离
        
        for i in range(target_num):
            target_id = i + 1
            # 生成不重复的目标位置
            position = self._generate_unique_position(
                occupied_positions, 
                min_distance=min_distance_between_targets,
                region_bounds=(map_size * 0.2, map_size * 0.8, map_size * 0.2, map_size * 0.8),
                max_attempts=50
            )
            occupied_positions.append(position)
            
            # 为每种资源类型生成整数需求，确保不为全零
            resources = np.random.randint(50, 151, size=num_resource_types)
            # 确保资源需求不为全零（虽然随机范围已经避免了这个问题）
            if np.all(resources == 0):
                resources = np.ones(num_resource_types, dtype=int) * 50
            
            value = np.random.randint(80, 121)
            self.targets.append(Target(id=target_id, position=position, resources=resources, value=value))
            total_demand += resources

        # 2. 根据总需求和资源富裕度，智能生成无人机整数资源
        # 修复: 确保每个资源维度都有相同的resource_abundance比率
        
        # 方法: 先计算统一的总供给，然后按需求比例分配到各维度
        total_demand_sum = np.sum(total_demand)
        total_supply_target = total_demand_sum * resource_abundance
        total_supply_int = int(np.round(total_supply_target))
        
        # 按需求比例分配到各维度
        demand_proportions = total_demand / total_demand_sum
        expected_supply_by_proportion = total_supply_int * demand_proportions
        
        # 优化的整数化策略：最小化比率差异
        # 方法1：直接四舍五入
        option1 = np.round(expected_supply_by_proportion).astype(int)
        
        # 方法2：优先保证总量，然后最小化比率差异
        option2 = np.floor(expected_supply_by_proportion).astype(int)
        remaining = total_supply_int - np.sum(option2)
        # 按剩余部分的大小排序，优先分配给剩余部分较大的维度
        fractional_parts = expected_supply_by_proportion - option2
        sorted_indices = np.argsort(-fractional_parts)  # 降序排列
        for i in range(remaining):
            if i < len(sorted_indices):
                option2[sorted_indices[i]] += 1
        
        # 选择比率差异最小的方案
        ratio1 = option1 / (total_demand.astype(float) + 1e-6)
        ratio2 = option2 / (total_demand.astype(float) + 1e-6)
        
        diff1 = np.abs(ratio1[0] - ratio1[1]) if len(ratio1) > 1 else 0
        diff2 = np.abs(ratio2[0] - ratio2[1]) if len(ratio2) > 1 else 0
        
        if diff1 <= diff2:
            total_supply_int_by_dim = option1
            chosen_method = "四舍五入"
        else:
            total_supply_int_by_dim = option2  
            chosen_method = "优化分配"
        
        # 处理四舍五入误差，确保总和一致
        supply_diff = total_supply_int - np.sum(total_supply_int_by_dim)
        if supply_diff != 0:
            # 调整最大需求维度来吸收四舍五入误差
            max_demand_idx = np.argmax(total_demand)
            total_supply_int_by_dim[max_demand_idx] += supply_diff
        
        # 验证统一比率
        actual_ratio = total_supply_int_by_dim / (total_demand.astype(float) + 1e-6)
        expected_range = template['resource_abundance_range']
        
        # 资源分配详细信息已移除，不在控制台显示
        # print(f"初始估算 - 总需求: {total_demand} (总和: {total_demand_sum})")
        # print(f"目标总供给: {total_supply_target:.1f} -> 整数: {total_supply_int}")
        # print(f"按比例分配 ({chosen_method}): {total_supply_int_by_dim}, 统一比率: {actual_ratio}")
        # print(f"比率差异: {np.abs(actual_ratio[0] - actual_ratio[1]):.6f}, 期望范围: {expected_range}")
        
        # 验证比率是否在期望范围内
        avg_ratio = np.mean(actual_ratio)
        if avg_ratio < expected_range[0] or avg_ratio > expected_range[1]:
            # 如果超出范围，调整总供给
            if avg_ratio < expected_range[0]:
                target_ratio = expected_range[0]
            else:
                target_ratio = expected_range[1]
            
            # 重新计算供给
            total_supply_adjusted = int(np.round(total_demand_sum * target_ratio))
            expected_supply_adjusted = total_supply_adjusted * demand_proportions
            total_supply_int_by_dim = np.round(expected_supply_adjusted).astype(int)
            
            # 再次处理四舍五入误差
            supply_diff = total_supply_adjusted - np.sum(total_supply_int_by_dim)
            if supply_diff != 0:
                max_demand_idx = np.argmax(total_demand)
                total_supply_int_by_dim[max_demand_idx] += supply_diff
            
            actual_ratio = total_supply_int_by_dim / (total_demand.astype(float) + 1e-6)
            # print(f"调整后 - 整数供给: {total_supply_int_by_dim}, 统一比率: {actual_ratio}")
        
        # 使用统一的供给量
        final_total_supply = total_supply_int_by_dim
        
        # 智能分配整数资源给每个UAV
        uav_resources = np.zeros((uav_num, num_resource_types), dtype=int)
        
        for r_type in range(num_resource_types):
            remaining_supply = final_total_supply[r_type]
            
            # 首先为每个UAV分配基础资源（确保每个UAV至少有一些资源）
            min_per_uav = max(2, remaining_supply // (uav_num * 2))  # 提高最小分配
            for uav_idx in range(uav_num):
                base_allocation = min(min_per_uav, remaining_supply)
                uav_resources[uav_idx, r_type] = base_allocation
                remaining_supply -= base_allocation
            
            # 随机分配剩余资源
            while remaining_supply > 0:
                # 随机选择一个UAV分配额外资源
                uav_idx = np.random.randint(0, uav_num)
                allocation = min(remaining_supply, np.random.randint(1, min(10, remaining_supply + 1)))
                uav_resources[uav_idx, r_type] += allocation
                remaining_supply -= allocation
        
        # 确保没有UAV的所有资源都为零，也确保没有单个资源类型为零
        zero_uav_count = 0
        zero_resource_fixes = 0
        
        for uav_idx in range(uav_num):
            # 检查是否所有资源都为零
            if np.all(uav_resources[uav_idx] == 0):
                zero_uav_count += 1
                # 如果某个UAV所有资源都为零，给它分配最小资源
                uav_resources[uav_idx] = np.array([2, 2], dtype=int)  # 直接赋值
                zero_resource_fixes += 1
            
            # 检查是否有单个资源类型为零
            for r_type in range(num_resource_types):
                if uav_resources[uav_idx, r_type] == 0:
                    uav_resources[uav_idx, r_type] = 1  # 给予最小值
                    zero_resource_fixes += 1
        
        # 记录分配结果
        actual_total_supply = np.sum(uav_resources, axis=0)
        actual_final_ratio = actual_total_supply / (total_demand.astype(float) + 1e-6)
        # print(f"最终分配 - UAV总资源: {actual_total_supply}, 实际比率: {actual_final_ratio}, 全零UAV数量: {zero_uav_count}, 零资源修复: {zero_resource_fixes}")

        # 生成无人机位置时也加入去重机制
        min_distance_between_uavs = map_size * 0.05  # 无人机间最小距离
        min_distance_uav_to_target = map_size * 0.08  # 无人机到目标的最小距离
        
        for i in range(uav_num):
            uav_id = i + 1
            # 生成不重复的无人机位置，在地图边缘区域
            position = self._generate_unique_uav_position(
                occupied_positions,
                min_distance_to_existing=min_distance_between_uavs,
                min_distance_to_targets=min_distance_uav_to_target,
                map_size=map_size,
                max_attempts=50
            )
            occupied_positions.append(position)
            
            heading = np.random.uniform(0, 2 * np.pi)
            max_distance = self.config.UAV_MAX_DISTANCE
            velocity_range = self.config.UAV_VELOCITY_RANGE
            economic_speed = self.config.UAV_ECONOMIC_SPEED
            
            self.uavs.append(UAV(
                id=uav_id,
                position=position,
                heading=heading,
                resources=uav_resources[i],
                max_distance=max_distance,
                velocity_range=velocity_range,
                economic_speed=economic_speed
            ))

        # 3. 智能生成障碍物 - 确保位置和大小合理，不与实体重叠
        generated_obstacles = 0
        max_retries = getattr(self.config, 'SCENARIO_GENERATION_MAX_RETRIES', 10)
        
        for i in range(obstacle_num):
            for retry in range(max_retries):
                # 障碍物位置：优先在目标和UAV之间的路径上，但要避免重叠
                if len(self.targets) > 0 and len(self.uavs) > 0 and np.random.random() < 0.7:
                    # 70%的概率在目标和UAV之间生成
                    target = np.random.choice(self.targets)
                    uav = np.random.choice(self.uavs)
                    
                    target_pos = target.position
                    uav_pos = uav.current_position
                    
                    # 在路径中点附近生成障碍物
                    mid_point = (target_pos + uav_pos) / 2
                    # 添加一些随机偏移
                    offset = np.random.uniform(-map_size * 0.15, map_size * 0.15, 2)
                    center = mid_point + offset
                else:
                    # 30%的概率随机生成
                    center = np.random.uniform(map_size * 0.15, map_size * 0.85, 2)
                
                # 确保障碍物在地图范围内
                center = np.clip(center, map_size * 0.1, map_size * 0.9)
                
                # 障碍物大小：根据地图大小和场景难度合理设置
                if scenario_name == 'easy':
                    min_radius = map_size * 0.01   # 简单场景：小障碍物
                    max_radius = map_size * 0.04
                elif scenario_name == 'medium':
                    min_radius = map_size * 0.02   # 中等场景：中等障碍物
                    max_radius = map_size * 0.06
                else:  # hard
                    min_radius = map_size * 0.03   # 困难场景：大障碍物
                    max_radius = map_size * 0.08
                
                radius = np.random.uniform(min_radius, max_radius)
                tolerance = getattr(self.config, 'OBSTACLE_TOLERANCE', 50.0)
                
                # 检查障碍物是否与现有实体重叠
                overlap = False
                
                # 根据场景难度调整安全距离
                if scenario_name == 'easy':
                    safety_distance = radius + 200  # 简单场景：大安全距离
                elif scenario_name == 'medium':
                    safety_distance = radius + 120  # 中等场景：中等安全距离
                else:  # hard
                    safety_distance = radius + 80   # 困难场景：小安全距离，允许更紧密放置
                
                # 检查与目标的重叠
                for target in self.targets:
                    if np.linalg.norm(center - target.position) < safety_distance:
                        overlap = True
                        break
                
                # 检查与UAV的重叠
                if not overlap:
                    for uav in self.uavs:
                        if np.linalg.norm(center - uav.current_position) < safety_distance:
                            overlap = True
                            break
                
                # 检查与其他障碍物的重叠
                if not overlap:
                    for obstacle in self.obstacles:
                        if hasattr(obstacle, 'center') and hasattr(obstacle, 'radius'):
                            dist = np.linalg.norm(center - obstacle.center)
                            # 根据场景难度调整障碍物间最小距离
                            if scenario_name == 'easy':
                                min_obstacle_distance = 80
                            elif scenario_name == 'medium':
                                min_obstacle_distance = 60
                            else:  # hard
                                min_obstacle_distance = 40  # 困难场景允许障碍物更紧密
                            
                            if dist < (radius + obstacle.radius + min_obstacle_distance):
                                overlap = True
                                break
                
                # 如果没有重叠，创建障碍物
                if not overlap:
                    self.obstacles.append(CircularObstacle(center=center, radius=radius, tolerance=tolerance))
                    generated_obstacles += 1
                    break
            
            # 如果重试次数超过限制，放宽约束或停止生成
            if retry == max_retries - 1:
                # 对于hard场景，放宽约束以确保生成足够的障碍物
                if scenario_name == 'hard' or generated_obstacles < obstacle_num * 0.6:
                    center = np.random.uniform(map_size * 0.2, map_size * 0.8, 2)
                    # 对于hard场景，使用较小的障碍物但数量更多
                    if scenario_name == 'hard':
                        radius = np.random.uniform(min_radius * 0.7, max_radius * 0.8)
                    else:
                        radius = np.random.uniform(min_radius, max_radius)
                    
                    self.obstacles.append(CircularObstacle(center=center, radius=radius, tolerance=tolerance))
                    generated_obstacles += 1
        
        # 记录障碍物生成结果
        # if generated_obstacles < obstacle_num:
        #     print(f"警告: 仅生成 {generated_obstacles}/{obstacle_num} 个障碍物（受重叠检查限制）")
        if getattr(self.config, 'ENABLE_SCENARIO_DEBUG', False):
            # 使用 len() 来获取实际生成的实体数量
            print(f"[DEBUG] 最终实际生成数量: UAV={len(self.uavs)}, Target={len(self.targets)}, Obstacle={len(self.obstacles)}")

        # 应用场景多样性特征，确保不同调用产生不同的场景布局
        self._add_scenario_diversity_features(scenario_name, uav_num, target_num)

    def _generate_unique_position(self, occupied_positions, min_distance, region_bounds, max_attempts=50):
        """
        生成不与已有位置重复的新位置
        
        Args:
            occupied_positions: 已占用位置列表
            min_distance: 最小距离要求
            region_bounds: 区域边界 (x_min, x_max, y_min, y_max)
            max_attempts: 最大尝试次数
            
        Returns:
            np.array: 新的位置坐标
        """
        x_min, x_max, y_min, y_max = region_bounds
        
        for attempt in range(max_attempts):
            # 生成候选位置
            position = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])
            
            # 检查与所有已占用位置的距离
            valid = True
            for occupied_pos in occupied_positions:
                distance = np.linalg.norm(position - occupied_pos)
                if distance < min_distance:
                    valid = False
                    break
            
            if valid:
                return position
        
        # 如果多次尝试都失败，返回一个随机位置（降级处理）
        print(f"警告: 无法生成满足距离要求的位置，使用随机位置（尝试{max_attempts}次）")
        return np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ])
    
    def _generate_unique_uav_position(self, occupied_positions, min_distance_to_existing, 
                                     min_distance_to_targets, map_size, max_attempts=50):
        """
        生成不与已有位置重复的无人机位置（在地图边缘区域）
        
        Args:
            occupied_positions: 已占用位置列表
            min_distance_to_existing: 与其他无人机的最小距离
            min_distance_to_targets: 与目标的最小距离
            map_size: 地图大小
            max_attempts: 最大尝试次数
            
        Returns:
            np.array: 新的无人机位置坐标
        """
        for attempt in range(max_attempts):
            # 在地图边缘区域生成候选位置
            if np.random.random() < 0.5:
                # 水平边缘
                x = np.random.uniform(0, map_size)
                y = np.random.choice([
                    np.random.uniform(0, map_size * 0.1),
                    np.random.uniform(map_size * 0.9, map_size)
                ])
            else:
                # 垂直边缘
                x = np.random.choice([
                    np.random.uniform(0, map_size * 0.1),
                    np.random.uniform(map_size * 0.9, map_size)
                ])
                y = np.random.uniform(0, map_size)
            
            position = np.array([x, y])
            
            # 检查与所有已占用位置的距离
            valid = True
            for i, occupied_pos in enumerate(occupied_positions):
                distance = np.linalg.norm(position - occupied_pos)
                # 对于目标位置，使用更大的最小距离
                if i < len(self.targets):
                    min_dist = min_distance_to_targets
                else:
                    min_dist = min_distance_to_existing
                
                if distance < min_dist:
                    valid = False
                    break
            
            if valid:
                return position
        
        # 如果多次尝试都失败，返回一个边缘随机位置（降级处理）
        print(f"警告: 无法生成满足距离要求的无人机位置，使用边缘随机位置（尝试{max_attempts}次）")
        if np.random.random() < 0.5:
            x = np.random.uniform(0, map_size)
            y = np.random.choice([
                np.random.uniform(0, map_size * 0.1),
                np.random.uniform(map_size * 0.9, map_size)
            ])
        else:
            x = np.random.choice([
                np.random.uniform(0, map_size * 0.1),
                np.random.uniform(map_size * 0.9, map_size)
            ])
            y = np.random.uniform(0, map_size)
        
        return np.array([x, y])

    def _add_scenario_diversity_features(self, scenario_name, uav_num, target_num):
        """
        为场景添加多样性特征，确保不同调用产生不同的场景布局
        
        Args:
            scenario_name: 场景名称
            uav_num: 无人机数量
            target_num: 目标数量
        """
        # 根据场景难度和实体数量调整空间分布模式
        if scenario_name == 'easy':
            # 简单场景：实体分布较为分散，便于规划
            self._apply_dispersed_layout()
        elif scenario_name == 'medium':
            # 中等场景：混合分布模式
            layout_mode = np.random.choice(['clustered', 'dispersed', 'mixed'])
            if layout_mode == 'clustered':
                self._apply_clustered_layout()
            elif layout_mode == 'dispersed':
                self._apply_dispersed_layout()
            else:
                self._apply_mixed_layout()
        else:  # hard
            # 困难场景：更倾向于聚集分布，增加协同难度
            layout_mode = np.random.choice(['clustered', 'mixed'], p=[0.7, 0.3])
            if layout_mode == 'clustered':
                self._apply_clustered_layout()
            else:
                self._apply_mixed_layout()
    
    def _apply_dispersed_layout(self):
        """应用分散式布局 - 实体尽可能分散分布"""
        # 为分散布局调整位置（如果需要进一步优化）
        pass
    
    def _apply_clustered_layout(self):
        """应用聚集式布局 - 目标形成若干聚集区域"""
        # 为聚集布局重新调整目标位置
        if len(self.targets) > 2:
            # 选择1-2个聚集中心
            num_clusters = min(2, len(self.targets) // 2)
            map_size = self.config.MAP_SIZE
            
            # 生成聚集中心
            cluster_centers = []
            for _ in range(num_clusters):
                center = np.array([
                    np.random.uniform(map_size * 0.3, map_size * 0.7),
                    np.random.uniform(map_size * 0.3, map_size * 0.7)
                ])
                cluster_centers.append(center)
            
            # 将目标重新分配到聚集中心附近
            cluster_radius = map_size * 0.15
            for i, target in enumerate(self.targets):
                cluster_idx = i % num_clusters
                center = cluster_centers[cluster_idx]
                
                # 在聚集中心附近生成新位置
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, cluster_radius)
                offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                new_position = center + offset
                
                # 确保位置在地图范围内
                new_position = np.clip(new_position, 
                                     [map_size * 0.1, map_size * 0.1], 
                                     [map_size * 0.9, map_size * 0.9])
                target.position = new_position
    
    def _apply_mixed_layout(self):
        """应用混合式布局 - 部分目标聚集，部分分散"""
        if len(self.targets) > 3:
            # 一半目标聚集，一半分散
            cluster_count = len(self.targets) // 2
            
            # 对前一半目标应用聚集
            clustered_targets = self.targets[:cluster_count]
            original_targets = self.targets.copy()
            self.targets = clustered_targets
            self._apply_clustered_layout()
            
            # 恢复完整目标列表
            self.targets = original_targets

    def decode_action(self, action):
        """解码动作索引为任务分配 - 公共接口"""
        return self._action_to_assignment(action)
    
    def _action_to_assignment(self, action):
        """将动作索引转换为任务分配 - 修复版本，添加边界检查"""
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        n_phi = self.graph.n_phi
        
        # 确保动作在有效范围内
        max_valid_action = n_targets * n_uavs * n_phi - 1
        if action > max_valid_action:
            print(f"警告: 动作 {action} 超出有效范围 [0, {max_valid_action}]，调整为模运算结果")
            action = action % (max_valid_action + 1)
        
        target_idx = action // (n_uavs * n_phi)
        remaining = action % (n_uavs * n_phi)
        uav_idx = remaining // n_phi
        phi_idx = remaining % n_phi
        
        # 再次验证索引边界
        target_idx = min(target_idx, n_targets - 1)
        uav_idx = min(uav_idx, n_uavs - 1)
        phi_idx = min(phi_idx, n_phi - 1)
        
        # 确保索引非负
        target_idx = max(0, target_idx)
        uav_idx = max(0, uav_idx)
        phi_idx = max(0, phi_idx)
        
        return target_idx, uav_idx, phi_idx
    
    def _is_valid_action(self, target, uav, phi_idx):
        """检查动作是否有效 - 【核心修改】实现智能动作屏蔽规则"""
        
        # 【修改后的代码】
        # 规则1：无人机必须拥有目标实际需要的资源
        if not self._has_actual_contribution(target, uav):
            return False

        # 规则2：不能向"即将满足"的目标重复派遣
        # 检查目标的剩余需求是否已经被"在途"的资源覆盖
        effective_demand = target.remaining_resources - target.in_flight_resources
        if np.all(effective_demand <= 0):
            return False
        
        # 规则3 (原有逻辑)：不能重复分配同一个无人机到同一个目标
        if (uav.id, phi_idx) in target.allocated_uavs:
            return False

        return True
    
    def _has_actual_contribution(self, target, uav):
        """检查UAV对目标是否有实际贡献"""
        actual_contribution = np.minimum(uav.resources, target.remaining_resources)
        return np.any(actual_contribution > 0)
    
    def _get_top_k_uavs_for_target(self, target, k=None):
            """
            [修正版] 为指定目标获取距离最近且资源匹配的K架无人机。
            核心逻辑: 1. 找出所有能贡献的UAV并计算距离; 2. 按距离排序; 3. 取前K个。
            """
            try:
                if k is None:
                    k = self.config.TOP_K_UAVS
                
                # 参数边界检查
                if not isinstance(k, int) or k <= 0:
                    k = self.config.TOP_K_UAVS

                # [核心修正] 步骤 1: 找出所有能做出实际贡献的无人机，并记录它们的索引和距离
                candidate_uavs_with_dist = []
                target_pos = np.array(target.position)
                
                for i, uav in enumerate(self.uavs):
                    # 筛选条件：无人机有资源，且能对当前目标做出实际贡献
                    if np.any(uav.resources > 0) and self._has_actual_contribution(target, uav):
                        uav_pos = np.array(uav.current_position)
                        distance = np.linalg.norm(target_pos - uav_pos)
                        
                        if not (np.isnan(distance) or np.isinf(distance)):
                            candidate_uavs_with_dist.append((i, distance))

                # [核心修正] 步骤 2 & 3: 计算综合效用分并排序，选择前K个
                candidate_uavs_with_scores = []
                for uav_idx, distance in candidate_uavs_with_dist:
                    uav = self.uavs[uav_idx]
                    
                    # 计算资源匹配度 (越小越好)
                    # 惩罚那些资源远多于需求的UAV，鼓励精确匹配
                    contribution = np.minimum(uav.resources, target.remaining_resources)
                    total_contribution = np.sum(contribution)
                    total_need = np.sum(target.remaining_resources)
                    
                    if total_need > 0:
                        # 资源匹配度：1 - (贡献 / 需求)，值越小说明越匹配
                        resource_match_score = 1.0 - (total_contribution / total_need)
                    else:
                        resource_match_score = 1.0 # 如果目标无需求，则匹配度最低

                    # 归一化距离 (0-1)
                    normalized_distance = distance / self.config.MAP_SIZE
                    
                    # 定义权重
                    w_dist = 0.6  # 距离权重占60%
                    w_res = 0.4   # 资源匹配权重占40%
                    
                    # 综合效用分 (分数越低越优)
                    utility_score = w_dist * normalized_distance + w_res * resource_match_score
                    candidate_uavs_with_scores.append((uav_idx, utility_score))

                # 按综合效用分排序 (升序)
                candidate_uavs_with_scores.sort(key=lambda x: x[1])
                
                top_k_count = min(k, len(candidate_uavs_with_scores))
                
                # 返回效用分最高的无人机索引列表
                return [uav_idx for uav_idx, _ in candidate_uavs_with_scores[:top_k_count]]
            except Exception as e:
                print(f"严重错误: Top-K 筛选方法出现异常: {e}，回退到返回所有无人机索引。")
                return list(range(len(self.uavs)))

    
    def get_action_mask(self):
        """
        生成基于距离剪枝的动作掩码，标识所有有效动作
        
        使用Top-K筛选算法减少动作空间，提高训练效率
        
        Returns:
            np.ndarray: 布尔型数组，形状为(n_actions,)，True表示有效动作，False表示无效动作
        """
        try:
            action_mask = np.zeros(self.n_actions, dtype=bool)
            
            # 为每个目标获取距离最近的K架无人机
            for target_idx, target in enumerate(self.targets):
                try:
                    # 获取该目标的Top-K无人机
                    top_k_uav_indices = self._get_top_k_uavs_for_target(target)
                    
                    # 只对筛选后的无人机-目标组合进行有效性检查
                    for uav_idx in top_k_uav_indices:
                        try:
                            # 索引边界检查
                            if uav_idx >= len(self.uavs):
                                # 不打印每个超出范围的无人机索引，避免大量输出
                                # 在step方法结束时统一汇报超出范围的索引数量
                                self._invalid_action_count = getattr(self, '_invalid_action_count', 0) + 1
                                continue
                                
                            uav = self.uavs[uav_idx]
                            
                            # 遍历所有可能的phi角度
                            for phi_idx in range(self.graph.n_phi):
                                try:
                                    # 计算对应的动作索引
                                    action_idx = target_idx * (len(self.uavs) * self.graph.n_phi) + uav_idx * self.graph.n_phi + phi_idx
                                    
                                    # 动作索引边界检查
                                    if action_idx >= self.n_actions:
                                        # 不打印每个超出范围的动作索引，避免大量输出
                                        # 在step方法结束时统一汇报超出范围的动作数量
                                        self._invalid_action_count = getattr(self, '_invalid_action_count', 0) + 1
                                        continue
                                    
                                    # 应用原有的有效性检查逻辑
                                    if self._is_valid_action(target, uav, phi_idx) and self._has_actual_contribution(target, uav):
                                        action_mask[action_idx] = True
                                        
                                except Exception as e:
                                    print(f"警告: 处理动作 (target={target_idx}, uav={uav_idx}, phi={phi_idx}) 时出错: {e}")
                                    continue
                                    
                        except Exception as e:
                            print(f"警告: 处理无人机 {uav_idx} 时出错: {e}")
                            continue
                            
                except Exception as e:
                    print(f"警告: 处理目标 {target_idx} 时出错: {e}")
                    continue
            
            # 如果没有有效动作，回退到原始方法
            if not np.any(action_mask):
                # print("警告: 剪枝后没有有效动作，回退到原始动作掩码生成")
                return self._get_original_action_mask()
            
            return action_mask
            
        except Exception as e:
            print(f"警告: 动作掩码生成异常: {e}，回退到原始方法")
            return self._get_original_action_mask()
    
    def _get_original_action_mask(self):
        """
        原始的动作掩码生成方法（作为回退机制）
        
        Returns:
            np.ndarray: 布尔型数组，形状为(n_actions,)
        """
        try:
            action_mask = np.zeros(self.n_actions, dtype=bool)
            
            # 遍历所有可能的动作
            for action_idx in range(self.n_actions):
                try:
                    target_idx, uav_idx, phi_idx = self._action_to_assignment(action_idx)
                    target = self.targets[target_idx]
                    uav = self.uavs[uav_idx]
                    
                    # 检查动作是否有效且有实际贡献
                    if self._is_valid_action(target, uav, phi_idx) and self._has_actual_contribution(target, uav):
                        action_mask[action_idx] = True
                except Exception as e:
                    print(f"警告: 处理动作 {action_idx} 时出错: {e}")
                    continue
            
            return action_mask
            
        except Exception as e:
            print(f"严重错误: 原始动作掩码生成也失败: {e}")
            # 最后的回退：返回全零掩码
            return np.zeros(self.n_actions, dtype=bool)
    
    def get_action_mask_from_state(self, state_dict=None):
        """
        从给定状态生成动作掩码（用于批处理）
        
        Args:
            state_dict: 状态字典，如果为None则使用当前状态
            
        Returns:
            np.ndarray: 布尔型数组，形状为(n_actions,)
        """
        # 如果没有提供状态字典，使用当前状态
        if state_dict is None:
            return self.get_action_mask()
        
        # 对于批处理，我们暂时使用当前状态的掩码
        # 这是一个简化实现，在实际应用中可能需要更复杂的状态重建逻辑
        return self.get_action_mask()

    def calculate_simplified_reward(self, target, uav, actual_contribution, path_len, 
                                was_satisfied, travel_time, done):
        """
        简化的奖励函数，重点关注目标资源满足和死锁避免
        
        Args:
            target: 目标对象
            uav: UAV对象
            actual_contribution: 实际资源贡献
            path_len: 路径长度
            was_satisfied: 之前是否已满足目标
            travel_time: 旅行时间
            done: 是否完成所有目标
            
        Returns:
            float: 归一化的奖励值
        """
        # 1. 任务完成奖励 (最高优先级)
        if done:
            return 10.0  # 归一化后的最高奖励
        
        # 2. 目标满足奖励
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = int(now_satisfied and not was_satisfied)
        target_completion_reward = 5.0 if new_satisfied else 0.0
        
        # 3. 资源贡献奖励 (核心奖励)
        # 计算贡献比例而不是绝对值
        target_initial_total = np.sum(target.resources)
        contribution_ratio = np.sum(actual_contribution) / target_initial_total if target_initial_total > 0 else 0
        contribution_reward = contribution_ratio * 3.0  # 最高3分
        
        # 4. 零贡献惩罚 (避免死锁)
        if np.all(actual_contribution <= 0):
            return -1.0  # 严重惩罚零贡献动作
        
        # 5. 距离惩罚 (简化版)
        # 使用相对距离而不是绝对距离
        max_distance = 1000.0  # 假设的最大距离
        distance_ratio = min(path_len / max_distance, 1.0)
        distance_penalty = -distance_ratio * 1.0  # 最多-1分
        
        # 总奖励 (归一化到[-5, 10]范围)
        total_reward = target_completion_reward + contribution_reward + distance_penalty
        
        return float(total_reward)
    
    def _calculate_reward_legacy(self, target, uav, actual_contribution, path_len, 
                         was_satisfied, travel_time, done):
        """
        Per-Agent归一化奖励函数 - 解决尺度漂移问题
        
        核心设计理念:
        1. 巨大的正向奖励作为核心激励
        2. 所有成本作为正奖励的动态百分比减项
        3. 塑形奖励引导探索
        4. **Per-Agent归一化**: 识别与无人机数量相关的奖励项，除以当前有效无人机数量
        5. 移除所有硬编码的巨大惩罚值
        
        奖励结构:
        - 任务完成奖励: 100.0 (核心正向激励)
        - 资源贡献奖励: 10.0-50.0 (基于贡献比例)
        - 塑形奖励: 0.1-2.0 (接近目标、协作等)
        - 动态成本: 正奖励的3-8%作为减项
        - **归一化处理**: 拥堵惩罚等与UAV数量相关的奖励项按N_active归一化
        """
        
        # ===== 第一部分: 计算当前有效无人机数量 (Per-Agent归一化基础) =====
        n_active_uavs = self._calculate_active_uav_count()
        
        # ===== 第二部分: 计算所有正向奖励 =====
        positive_rewards = 0.0
        reward_components = {
            'n_active_uavs': n_active_uavs,  # 记录当前有效无人机数量用于调试
            'normalization_applied': []      # 记录哪些奖励项应用了归一化
        }
        
        # 1. 任务完成的巨大正向奖励 (核心激励)
        now_satisfied = np.all(target.remaining_resources <= 0)
        new_satisfied = now_satisfied and not was_satisfied
        
        if new_satisfied:
            task_completion_reward = 100.0  # 巨大的任务完成奖励
            positive_rewards += task_completion_reward
            reward_components['task_completion'] = task_completion_reward
        
        # 2. 资源贡献奖励 (基于实际贡献的正向激励)
        contribution_reward = 0.0
        if np.sum(actual_contribution) > 0:
            target_initial_total = np.sum(target.resources)
            if target_initial_total > 0:
                # 计算贡献比例
                contribution_ratio = np.sum(actual_contribution) / target_initial_total
                
                # 基础贡献奖励: 10-50分
                base_contribution = 10.0 + 40.0 * contribution_ratio
                
                # 边际效用奖励: 对小贡献也给予鼓励（安全版本）
                def safe_marginal_utility(ratio):
                    if np.isnan(ratio) or np.isinf(ratio) or ratio < 0:
                        return 0.0
                    try:
                        result = 15.0 * np.sqrt(max(0.0, min(1.0, ratio)))
                        return result if np.isfinite(result) else 0.0
                    except:
                        return 0.0
                
                marginal_utility = safe_marginal_utility(contribution_ratio)
                
                # 高效贡献奖励: 对大比例贡献给予额外奖励
                efficiency_bonus = 0.0
                if contribution_ratio > 0.3:
                    efficiency_bonus = 10.0 * (contribution_ratio - 0.3)
                
                contribution_reward = base_contribution + marginal_utility + efficiency_bonus
                positive_rewards += contribution_reward
                reward_components['contribution'] = contribution_reward
        
        # 3. 塑形奖励 - 引导探索和协作
        shaping_rewards = 0.0
        
        # 3.1 接近目标的塑形奖励
        approach_reward = self._calculate_approach_reward(uav, target)
        shaping_rewards += approach_reward
        reward_components['approach_shaping'] = approach_reward
        
        # 3.2 首次接触目标奖励
        if len(target.allocated_uavs) == 1 and target.allocated_uavs[0][0] == uav.id:
            first_contact_reward = 5.0
            shaping_rewards += first_contact_reward
            reward_components['first_contact'] = first_contact_reward
        
        # 3.3 协作塑形奖励 (Per-Agent归一化)
        collaboration_reward_raw = self._calculate_collaboration_reward(target, uav)
        # 协作奖励与UAV数量相关，需要归一化
        collaboration_reward = collaboration_reward_raw / n_active_uavs
        shaping_rewards += collaboration_reward
        reward_components['collaboration_raw'] = collaboration_reward_raw
        reward_components['collaboration_normalized'] = collaboration_reward
        reward_components['normalization_applied'].append('collaboration')
        
        # 3.4 全局完成进度奖励
        global_progress_reward = self._calculate_global_progress_reward()
        shaping_rewards += global_progress_reward
        reward_components['global_progress'] = global_progress_reward
        
        positive_rewards += shaping_rewards
        
        # ===== 第三部分: 动态尺度成本计算 (包含Per-Agent归一化) =====
        total_costs = 0.0
        
        # 确保有最小正向奖励基数，避免除零
        reward_base = max(positive_rewards, 1.0)
        
        # 1. 距离成本 - 正向奖励的3-5%
        distance_cost_ratio = 0.03 + 0.02 * min(1.0, path_len / 3000.0)  # 3%-5%
        distance_cost_raw = reward_base * distance_cost_ratio
        total_costs += distance_cost_raw
        reward_components['distance_cost'] = -distance_cost_raw
        
        # 2. 时间成本 - 正向奖励的2-3%
        time_cost_ratio = 0.02 + 0.01 * min(1.0, travel_time / 60.0)  # 2%-3%
        time_cost_raw = reward_base * time_cost_ratio
        total_costs += time_cost_raw
        reward_components['time_cost'] = -time_cost_raw
        
        # 3. 拥堵惩罚 (新增 - 与UAV数量直接相关，需要Per-Agent归一化)
        congestion_penalty_raw = self._calculate_congestion_penalty(target, uav, n_active_uavs)
        congestion_penalty_normalized = congestion_penalty_raw / n_active_uavs
        total_costs += congestion_penalty_normalized
        reward_components['congestion_penalty_raw'] = -congestion_penalty_raw
        reward_components['congestion_penalty_normalized'] = -congestion_penalty_normalized
        if congestion_penalty_raw > 0:
            reward_components['normalization_applied'].append('congestion_penalty')
        
        # 4. 资源效率成本 - 如果贡献效率低
        efficiency_cost = 0.0
        if np.sum(actual_contribution) > 0:
            # 计算资源利用效率
            uav_capacity = np.sum(uav.resources)
            if uav_capacity > 0:
                utilization_ratio = np.sum(actual_contribution) / uav_capacity
                if utilization_ratio < 0.5:  # 利用率低于50%
                    efficiency_cost_ratio = 0.02 * (0.5 - utilization_ratio)  # 最多2%
                    efficiency_cost = reward_base * efficiency_cost_ratio
                    total_costs += efficiency_cost
                    reward_components['efficiency_cost'] = -efficiency_cost
        
        # ===== 第四部分: 特殊情况处理 =====
        
        # 零贡献的温和引导 (不再是硬编码的巨大惩罚)
        if np.sum(actual_contribution) <= 0:
            # 给予最小的基础奖励，但增加成本比例
            if positive_rewards == 0:
                positive_rewards = 0.5  # 最小基础奖励
                reward_components['base_reward'] = 0.5
            
            # 增加无效行动成本 (正向奖励的10%)
            ineffective_cost = positive_rewards * 0.1
            total_costs += ineffective_cost
            reward_components['ineffective_cost'] = -ineffective_cost
        
        # 全局任务完成的超级奖励
        if done:
            all_targets_satisfied = all(np.all(t.remaining_resources <= 0) for t in self.targets)
            if all_targets_satisfied:
                global_completion_reward = 200.0  # 超级完成奖励
                positive_rewards += global_completion_reward
                reward_components['global_completion'] = global_completion_reward
        
        # ===== 第五部分: 最终奖励计算与归一化总结 =====
        final_reward = positive_rewards - total_costs
        
        # 数值稳定性检查，但不进行裁剪（由step函数统一处理）
        if np.isnan(final_reward) or np.isinf(final_reward):
            final_reward = 0.0
        
        # 记录详细的奖励组成 (增强版 - 支持Per-Agent归一化监控)
        reward_components.update({
            'total_positive': positive_rewards,
            'total_costs': total_costs,
            'final_reward': final_reward,
            'target_id': target.id,
            'uav_id': uav.id,
            'contribution_amount': float(np.sum(actual_contribution)),
            'path_length': float(path_len),
            'travel_time': float(travel_time),
            'done': done,
            
            # Per-Agent归一化相关信息
            'per_agent_normalization': {
                'n_active_uavs': n_active_uavs,
                'total_uavs': len(self.uavs),
                'normalization_factor': 1.0 / n_active_uavs,
                'components_normalized': reward_components['normalization_applied'],
                'normalization_impact': self._calculate_normalization_impact(reward_components)
            },
            
            # 调试信息
            'debug_info': {
                'step_count': self.step_count,
                'allocated_uavs_to_target': len(target.allocated_uavs),
                'target_remaining_resources': float(np.sum(target.remaining_resources)),
                'uav_remaining_resources': float(np.sum(uav.resources))
            }
        })
        
        # 保存最新的奖励组成用于调试和监控
        self.last_reward_components = reward_components
        
        # 如果启用了详细日志记录，输出归一化信息
        if getattr(self.config, 'ENABLE_REWARD_LOGGING', False):
            self._log_reward_components(reward_components)
        # 暂时在这里进行奖励裁剪，进行压力测试，如果不再出现NaN，说明是奖励函数问题。
        # final_reward = np.clip(final_reward, -10.0, 10.0)

        return float(final_reward)
    
    # ===== PBRS相关方法 - 基于"协同战备"战略势函数 =====
    
    def _is_synergy_target(self, target):
        """
        判断一个目标是否必须通过协同才能完成
        
        Args:
            target: 目标对象
            
        Returns:
            bool: True表示需要协同，False表示单机可完成
        """
        try:
            # 计算单架无人机的平均载荷
            if not self.uavs:
                return False
                
            # 获取所有有作战能力的无人机的资源
            active_uavs = [uav for uav in self.uavs if np.sum(uav.resources) > 0]
            if not active_uavs:
                return False
            
            # 计算单架无人机的最大载荷
            max_single_uav_payload = max(np.sum(uav.resources) for uav in active_uavs)
            
            # 如果目标的总需求超过单架无人机的最大载荷，则需要协同
            target_total_demand = np.sum(target.remaining_resources)
            
            return target_total_demand > max_single_uav_payload
            
        except Exception as e:
            # 异常情况下保守返回False
            return False
    
    def _get_k_nearest_uavs(self, target, k):
        """
        为指定目标找出距离最近的k架有作战能力的无人机
        
        Args:
            target: 目标对象
            k: 需要的无人机数量
            
        Returns:
            list: 最近的k架有作战能力的无人机列表
        """
        try:
            # 筛选有作战能力的无人机（资源>0）
            active_uavs = [uav for uav in self.uavs if np.sum(uav.resources) > 0]
            
            if not active_uavs:
                return []
            
            # 计算所有有效无人机到目标的距离
            distances = []
            for uav in active_uavs:
                distance = np.linalg.norm(np.array(uav.position) - np.array(target.position))
                distances.append((distance, uav))
            
            # 按距离排序
            distances.sort(key=lambda x: x[0])
            
            # 返回最近的k架无人机
            k = min(k, len(distances))  # 确保k不超过可用无人机数量
            return [uav for _, uav in distances[:k]]
            
        except Exception as e:
            # 异常情况下返回空列表
            return []
    
    def _calculate_simple_potential(self):
        """
        最简单的势函数：Φ = 100 * (已完成目标数 / 总目标数)
        
        特点：
        - 单调递增：完成更多目标势能更高
        - 目标明确：直接对应最终目标
        - 最不可能出错：逻辑简单清晰
        - 理论安全：不改变最优策略
        
        Returns:
            float: 势能值 [0, 100]
        """
        completed_targets = sum(1 for t in self.targets if np.all(t.remaining_resources <= 0))
        total_targets = len(self.targets)
        
        if total_targets == 0:
            return 0.0
        
        completion_ratio = completed_targets / total_targets
        potential = 100.0 * completion_ratio
        
        # 应用缩放因子
        scale = getattr(self.config, 'PBRS_POTENTIAL_SCALE', 1.0)
        potential *= scale
        
        return potential
    
    def _calculate_progress_potential(self):
        """
        进度势函数：Φ = 100 * (总资源消耗进度)
        
        特点：
        - 连续变化：每次资源消耗都有反馈
        - 稠密信号：提供更多学习信息
        - 平滑过渡：避免奖励悬崖
        
        Returns:
            float: 势能值 [0, 100]
        """
        total_initial_demand = sum(np.sum(t.resources) for t in self.targets)
        total_remaining_demand = sum(np.sum(t.remaining_resources) for t in self.targets)
        
        if total_initial_demand <= 0:
            return 0.0
        
        progress_ratio = (total_initial_demand - total_remaining_demand) / total_initial_demand
        potential = 100.0 * progress_ratio
        
        # 应用缩放因子
        scale = getattr(self.config, 'PBRS_POTENTIAL_SCALE', 1.0)
        potential *= scale
        
        return potential

    def _calculate_standard_completion_rate(self):
        """
        计算标准完成率 - 统一的完成率计算方法
        
        基于实际资源贡献与需求总量的比值来统计完成率，确保与控制台和动作日志保持一致
        
        Returns:
            float: 标准完成率 [0.0, 1.0]
        """
        if not self.targets:
            return 1.0  # 没有目标时认为已完成
        
        # 计算总需求
        total_demand = np.sum([t.resources for t in self.targets], axis=0)
        total_demand_safe = np.maximum(total_demand, 1e-6)
        
        # 【修复】计算实际资源贡献 - 基于目标的剩余资源，确保与其他模块一致
        total_contribution = np.zeros_like(total_demand, dtype=np.float64)
        for target in self.targets:
            # 使用目标的实际贡献：初始需求 - 剩余需求
            target_contribution = target.resources - target.remaining_resources
            total_contribution += target_contribution.astype(np.float64)
        
        # 确保贡献值不会因为浮点误差出现负数
        total_contribution = np.maximum(total_contribution, 0)
        
        # 【修复】标准完成率计算：基于实际贡献资源与总需求的比值
        # 使用总和比值而不是平均比值，确保计算逻辑统一
        total_demand_sum = np.sum(total_demand)
        total_contribution_sum = np.sum(total_contribution)
        
        if total_demand_sum > 0:
                    # 使用“平均比例法”计算，与evaluate.py保持一致，更能反映任务成功度
            completion_rate_per_resource = np.minimum(total_contribution, total_demand) / total_demand_safe
            completion_rate = np.mean(completion_rate_per_resource)
        else:
            completion_rate = 1.0
        
        # 确保返回值在合理范围内
        return float(np.clip(completion_rate, 0.0, 1.0))
    
    def get_completion_rate(self):
        """
        对外提供的标准完成率获取接口
        
        Returns:
            float: 标准完成率 [0.0, 1.0]
        """
        return self._calculate_standard_completion_rate()
    
    def _calculate_potential(self):
        """
        简化势能函数 - 仅依赖任务总体完成进度
        
        使用非线性函数（平方函数）来计算势能，鼓励模型追求100%的完成率
        
        Returns:
            float: 总势能值，避免NaN情况
        """
        try:
            # 使用标准完成率计算方法
            completion_rate = self._calculate_standard_completion_rate()
            
            # 使用平方函数来计算势能，鼓励追求100%完成率
            # 平方函数使得接近完成时势能增长更快
            # [修改] 使用config中的PBRS_COMPLETION_WEIGHT替代硬编码的100.0
            completion_weight = getattr(self.config, 'PBRS_COMPLETION_WEIGHT', 50.0)
            completion_potential = (completion_rate ** 2) * completion_weight
            
            # 应用配置缩放因子
            scale = getattr(self.config, 'PBRS_POTENTIAL_SCALE', 0.01)
            total_potential = completion_potential * scale
            
            # 确保返回值不是NaN或无穷大
            if np.isnan(total_potential) or np.isinf(total_potential):
                return 0.0
            
            return float(total_potential)
            
        except Exception as e:
            # 异常情况下返回0，避免训练中断
            return 0.0

    # def _calculate_pbrs_base_reward(self, target, uav, actual_contribution, was_satisfied, all_targets_satisfied):
    #     """
    #     PBRS系统的基础奖励函数 - 已注释，恢复稳定基线
    #     """
    #     pass


    
    def _calculate_uav_alive_status(self, uav, uav_index):
        """
        计算无人机的存活状态（鲁棒性掩码）
        
        Args:
            uav: UAV对象
            uav_index: UAV索引
            
        Returns:
            float: 存活状态 (0.0 或 1.0)
        """
        # 基础存活检查：资源是否耗尽
        if np.all(uav.resources <= 0):
            return 0.0
        
        # 可以在这里添加更复杂的存活逻辑，如：
        # - 通信失效概率
        # - 传感器故障概率
        # - 距离过远导致的信号丢失
        
        return 1.0
    
    def _calculate_target_visibility_status(self, target, target_index):
        """
        计算目标的可见性状态（鲁棒性掩码）
        
        Args:
            target: 目标对象
            target_index: 目标索引
            
        Returns:
            float: 可见性状态 (0.0 或 1.0)
        """
        # 基础可见性检查：目标是否已完成
        if np.all(target.remaining_resources <= 0):
            return 0.0
        
        # 可以在这里添加更复杂的可见性逻辑，如：
        # - 天气条件影响
        # - 障碍物遮挡
        # - 传感器范围限制
        
        return 1.0
    
    def _calculate_robust_masks(self):
        """
        计算增强的掩码字典，支持鲁棒性场景
        
        Returns:
            dict: 包含各种掩码的字典
        """
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        # UAV有效性掩码
        uav_mask = np.ones(n_uavs, dtype=np.int32)
        for i, uav in enumerate(self.uavs):
            uav_mask[i] = int(self._calculate_uav_alive_status(uav, i))
        
        # 目标有效性掩码
        target_mask = np.ones(n_targets, dtype=np.int32)
        for i, target in enumerate(self.targets):
            target_mask[i] = int(self._calculate_target_visibility_status(target, i))
        
        return {
            "uav_mask": uav_mask,
            "target_mask": target_mask
        }
    
    def _calculate_active_uav_count(self):
        """
        计算当前有效无人机数量（用于Per-Agent归一化）
        
        Returns:
            int: 有效无人机数量
        """
        active_count = 0
        for uav in self.uavs:
            if np.any(uav.resources > 0):  # 至少有一种资源大于0
                active_count += 1
        return max(active_count, 1)  # 确保至少为1，避免除零错误
    
    def _calculate_congestion_penalty(self, target, uav, n_active_uavs):
        """
        计算拥堵惩罚（与UAV数量相关，需要归一化）
        
        Args:
            target: 目标对象
            uav: UAV对象
            n_active_uavs: 当前有效无人机数量
            
        Returns:
            float: 拥堵惩罚值
        """
        # 计算分配到同一目标的无人机数量
        uavs_on_target = len(target.allocated_uavs)
        
        # 如果多个无人机分配到同一目标，产生拥堵惩罚
        if uavs_on_target > 1:
            # 惩罚与分配的无人机数量成正比
            congestion_factor = (uavs_on_target - 1) / n_active_uavs
            base_penalty = 2.0  # 基础惩罚
            return base_penalty * congestion_factor
        
        return 0.0
    
    def _calculate_global_progress_reward(self):
        """
        计算全局完成进度奖励
        
        Returns:
            float: 全局进度奖励
        """
        if not self.targets:
            return 0.0
        
        # 计算总体完成进度
        total_initial_resources = sum(np.sum(target.resources) for target in self.targets)
        total_remaining_resources = sum(np.sum(target.remaining_resources) for target in self.targets)
        
        if total_initial_resources <= 0:
            return 0.0
        
        progress_ratio = (total_initial_resources - total_remaining_resources) / total_initial_resources
        
        # 给予渐进式奖励
        if progress_ratio > 0.8:
            return 2.0 * (progress_ratio - 0.8) / 0.2  # 80%-100%时给予最高2分
        elif progress_ratio > 0.5:
            return 1.0 * (progress_ratio - 0.5) / 0.3  # 50%-80%时给予最高1分
        else:
            return 0.5 * progress_ratio / 0.5  # 0%-50%时给予最高0.5分
    
    def _calculate_normalization_impact(self, reward_components):
        """
        计算归一化对奖励的影响
        
        Args:
            reward_components: 奖励组成字典
            
        Returns:
            dict: 归一化影响分析
        """
        impact = {}
        
        # 计算归一化前后的差异
        for component in reward_components.get('normalization_applied', []):
            raw_key = f"{component}_raw"
            normalized_key = f"{component}_normalized"
            
            if raw_key in reward_components and normalized_key in reward_components:
                raw_value = reward_components[raw_key]
                normalized_value = reward_components[normalized_key]
                impact[component] = {
                    'raw': raw_value,
                    'normalized': normalized_value,
                    'difference': raw_value - normalized_value,
                    'reduction_ratio': (raw_value - normalized_value) / raw_value if raw_value != 0 else 0
                }
        
        return impact
    
    def _log_reward_components(self, reward_components):
        """
        记录详细的奖励组成信息（用于调试）
        
        Args:
            reward_components: 奖励组成字典
        """
        print(f"[奖励详情] Step {self.step_count}")
        print(f"  有效UAV数量: {reward_components['n_active_uavs']}")
        print(f"  最终奖励: {reward_components['final_reward']:.3f}")
        print(f"  正向奖励总计: {reward_components['total_positive']:.3f}")
        print(f"  成本总计: {reward_components['total_costs']:.3f}")
        
        # 输出归一化信息
        if reward_components['normalization_applied']:
            print(f"  归一化组件: {reward_components['normalization_applied']}")
            for component, impact in reward_components['per_agent_normalization']['normalization_impact'].items():
                print(f"    {component}: {impact['raw']:.3f} -> {impact['normalized']:.3f} (减少 {impact['reduction_ratio']:.1%})")

    def _calculate_collaboration_reward(self, target, uav):
        """
        计算协作塑形奖励
        
        鼓励合理的协作，避免过度集中或过度分散
        """
        collaboration_reward = 0.0
        
        # 获取当前分配到该目标的UAV数量
        current_uav_count = len(target.allocated_uavs)
        
        if current_uav_count > 0:
            # 计算目标的资源需求量
            target_demand = np.sum(target.resources)
            
            # 估算理想的UAV数量 (基于资源需求)
            avg_uav_capacity = 50.0  # 假设平均UAV容量
            ideal_uav_count = max(1, min(4, int(np.ceil(target_demand / avg_uav_capacity))))
            
            # 协作效率奖励
            if current_uav_count <= ideal_uav_count:
                # 理想协作范围内
                efficiency_factor = 1.0 - abs(current_uav_count - ideal_uav_count) / ideal_uav_count
                collaboration_reward = 1.0 * efficiency_factor
            else:
                # 过度协作，递减奖励
                over_collaboration_penalty = (current_uav_count - ideal_uav_count) * 0.2
                collaboration_reward = max(0.2, 1.0 - over_collaboration_penalty)
            
            # 多样性奖励: 如果UAV来自不同起始位置
            if current_uav_count > 1:
                diversity_bonus = 0.3  # 基础多样性奖励
                collaboration_reward += diversity_bonus
        
        return collaboration_reward
    
    def _calculate_global_progress_reward(self):
        """
        计算全局进度塑形奖励
        
        基于整体任务完成进度给予奖励，鼓励系统性进展
        """
        if not self.targets:
            return 0.0
        
        # 计算全局完成率 - 使用标准计算方法
        completion_rate = self._calculate_standard_completion_rate()
        
        # 基于完成率的进度奖励
        progress_reward = 0.0
        
        # 里程碑奖励
        milestones = [0.25, 0.5, 0.75, 0.9]
        milestone_rewards = [0.5, 1.0, 1.5, 2.0]
        
        for milestone, reward in zip(milestones, milestone_rewards):
            if completion_rate >= milestone:
                # 检查是否刚达到这个里程碑
                if not hasattr(self, '_milestone_reached'):
                    self._milestone_reached = set()
                
                if milestone not in self._milestone_reached:
                    self._milestone_reached.add(milestone)
                    progress_reward += reward
        
        # 连续进度奖励 (平滑的进度激励)
        smooth_progress = 0.2 * completion_rate
        progress_reward += smooth_progress
        
        return progress_reward
    
    def _calculate_active_uav_count(self) -> int:
        """
        计算当前有效无人机数量，用于Per-Agent奖励归一化
        
        有效无人机定义：
        - 拥有剩余资源 (resources > 0)
        - 通信/感知系统正常 (is_alive = 1.0)
        
        Returns:
            int: 当前有效无人机数量 N_active
        """
        active_count = 0
        
        for i, uav in enumerate(self.uavs):
            # 检查是否有剩余资源
            has_resources = np.any(uav.resources > 0)
            
            # 检查通信/感知状态
            is_alive = self._calculate_uav_alive_status(uav, i)
            
            # 只有同时满足资源和通信条件的UAV才算有效
            if has_resources and is_alive >= 0.5:  # is_alive >= 0.5 表示至少部分功能正常
                active_count += 1
        
        # 确保至少有1个有效UAV，避免除零错误
        return max(active_count, 1)
    
    def _calculate_congestion_penalty(self, target, uav, n_active_uavs: int) -> float:
        """
        计算拥堵惩罚 - 与无人机数量相关的惩罚项
        
        拥堵惩罚的核心思想：
        1. 当多个UAV同时分配到同一目标时，产生拥堵
        2. 拥堵程度与分配到该目标的UAV数量成正比
        3. 该惩罚项会随着总UAV数量增加而增长，因此需要归一化
        
        Args:
            target: 目标对象
            uav: 当前UAV对象
            n_active_uavs: 当前有效无人机数量
            
        Returns:
            float: 拥堵惩罚值 (原始值，调用方负责归一化)
        """
        congestion_penalty = 0.0
        
        # 1. 目标拥堵惩罚：分配到同一目标的UAV过多
        allocated_uav_count = len(target.allocated_uavs)
        if allocated_uav_count > 1:
            # 计算理想分配数量
            target_demand = np.sum(target.resources)
            avg_uav_capacity = 50.0  # 假设平均UAV容量
            ideal_allocation = max(1, min(3, int(np.ceil(target_demand / avg_uav_capacity))))
            
            if allocated_uav_count > ideal_allocation:
                # 过度分配惩罚，随UAV数量线性增长
                over_allocation = allocated_uav_count - ideal_allocation
                congestion_penalty += over_allocation * 2.0  # 每个多余UAV惩罚2分
        
        # 2. 全局拥堵惩罚：系统整体UAV密度过高
        if n_active_uavs > len(self.targets) * 2:  # 如果UAV数量超过目标数量的2倍
            density_factor = n_active_uavs / (len(self.targets) * 2)
            global_congestion = (density_factor - 1.0) * 1.5  # 密度超标惩罚
            congestion_penalty += global_congestion
        
        # 3. 局部拥堵惩罚：计算当前UAV周围的拥堵情况
        local_congestion = self._calculate_local_congestion(uav, target)
        congestion_penalty += local_congestion
        
        return max(congestion_penalty, 0.0)  # 确保惩罚值非负
    
    def _calculate_local_congestion(self, uav, target) -> float:
        """
        计算局部拥堵情况
        
        Args:
            uav: 当前UAV对象
            target: 目标对象
            
        Returns:
            float: 局部拥堵惩罚值
        """
        local_congestion = 0.0
        congestion_radius = 200.0  # 拥堵检测半径
        
        # 统计在拥堵半径内的其他UAV数量
        nearby_uavs = 0
        for other_uav in self.uavs:
            if other_uav.id != uav.id:
                distance = np.linalg.norm(
                    np.array(other_uav.current_position) - np.array(uav.current_position)
                )
                if distance < congestion_radius:
                    nearby_uavs += 1
        
        # 如果附近UAV过多，产生拥堵惩罚
        if nearby_uavs > 2:  # 超过2个邻近UAV就算拥堵
            local_congestion = (nearby_uavs - 2) * 0.5  # 每个多余邻近UAV惩罚0.5分
        
        return local_congestion
    
    def _calculate_normalization_impact(self, reward_components: dict) -> dict:
        """
        计算归一化对奖励的影响程度
        
        Args:
            reward_components: 奖励组成字典
            
        Returns:
            dict: 归一化影响分析
        """
        impact = {
            'total_raw_normalized_rewards': 0.0,
            'total_normalized_rewards': 0.0,
            'normalization_savings': 0.0,
            'components_impact': {}
        }
        
        # 计算协作奖励的归一化影响
        if 'collaboration_raw' in reward_components and 'collaboration_normalized' in reward_components:
            raw_collab = reward_components['collaboration_raw']
            norm_collab = reward_components['collaboration_normalized']
            impact['total_raw_normalized_rewards'] += raw_collab
            impact['total_normalized_rewards'] += norm_collab
            impact['components_impact']['collaboration'] = {
                'raw': raw_collab,
                'normalized': norm_collab,
                'reduction': raw_collab - norm_collab
            }
        
        # 计算拥堵惩罚的归一化影响
        if 'congestion_penalty_raw' in reward_components and 'congestion_penalty_normalized' in reward_components:
            raw_congestion = abs(reward_components['congestion_penalty_raw'])
            norm_congestion = abs(reward_components['congestion_penalty_normalized'])
            impact['total_raw_normalized_rewards'] += raw_congestion
            impact['total_normalized_rewards'] += norm_congestion
            impact['components_impact']['congestion_penalty'] = {
                'raw': raw_congestion,
                'normalized': norm_congestion,
                'reduction': raw_congestion - norm_congestion
            }
        
        # 计算总的归一化节省
        impact['normalization_savings'] = impact['total_raw_normalized_rewards'] - impact['total_normalized_rewards']
        
        return impact
    
    def _log_reward_components(self, reward_components: dict):
        """
        记录奖励组成的详细日志，用于调试和监控
        
        Args:
            reward_components: 奖励组成字典
        """
        normalization_info = reward_components['per_agent_normalization']
        
        print(f"[Step {self.step_count}] Per-Agent奖励归一化详情:")
        print(f"  有效UAV数量: {normalization_info['n_active_uavs']}/{normalization_info['total_uavs']}")
        print(f"  归一化因子: {normalization_info['normalization_factor']:.4f}")
        print(f"  应用归一化的组件: {normalization_info['components_normalized']}")
        
        impact = normalization_info['normalization_impact']
        if impact['normalization_savings'] > 0:
            print(f"  归一化节省: {impact['normalization_savings']:.4f}")
            for component, details in impact['components_impact'].items():
                print(f"    {component}: {details['raw']:.4f} -> {details['normalized']:.4f} "
                      f"(减少 {details['reduction']:.4f})")
        
        print(f"  最终奖励: {reward_components['final_reward']:.4f}")
        print()
    
    def _calculate_uav_alive_status(self, uav, uav_idx: int) -> float:
        """
        计算UAV的存活状态，考虑通信/感知失效情况
        
        鲁棒性掩码机制的核心组件，用于标识无人机的通信/感知状态。
        在实际部署中，这可以基于：
        - 通信链路质量
        - 传感器状态
        - 电池电量
        - 系统健康状态
        
        Args:
            uav: UAV实体对象
            uav_idx: UAV索引
            
        Returns:
            float: 存活状态 (0.0=失效, 1.0=正常)
        """
        # 基础存活检查：是否有剩余资源
        has_resources = np.any(uav.resources > 0)
        if not has_resources:
            return 0.0
        
        # 模拟通信失效场景（可配置的失效概率）
        communication_failure_rate = getattr(self.config, 'UAV_COMM_FAILURE_RATE', 0.0)
        if communication_failure_rate > 0:
            # 使用确定性的伪随机数，基于step_count和uav_idx确保可复现性
            failure_seed = (self.step_count * 31 + uav_idx * 17) % 1000
            failure_prob = failure_seed / 1000.0
            if failure_prob < communication_failure_rate:
                return 0.0
        
        # 模拟感知系统失效（基于距离和环境复杂度）
        sensing_failure_rate = getattr(self.config, 'UAV_SENSING_FAILURE_RATE', 0.0)
        if sensing_failure_rate > 0:
            # 计算环境复杂度因子（障碍物密度、目标密度等）
            complexity_factor = self._calculate_environment_complexity(uav)
            adjusted_failure_rate = sensing_failure_rate * complexity_factor
            
            sensing_seed = (self.step_count * 23 + uav_idx * 19) % 1000
            sensing_prob = sensing_seed / 1000.0
            if sensing_prob < adjusted_failure_rate:
                return 0.0
        
        # 模拟电池电量影响的通信能力
        battery_threshold = getattr(self.config, 'UAV_LOW_BATTERY_THRESHOLD', 0.1)
        if hasattr(uav, 'battery_level'):
            if uav.battery_level < battery_threshold:
                # 低电量时通信能力下降，但不完全失效
                return 0.3
        
        # 模拟系统过载导致的响应延迟
        system_load = len([u for u in self.uavs if np.any(u.resources > 0)])
        max_concurrent_uavs = getattr(self.config, 'MAX_CONCURRENT_UAVS', 20)
        if system_load > max_concurrent_uavs:
            # 系统过载时，部分UAV可能响应延迟
            overload_factor = (system_load - max_concurrent_uavs) / max_concurrent_uavs
            if (uav_idx + self.step_count) % system_load < overload_factor * system_load:
                return 0.5  # 部分功能受限
        
        return 1.0  # 正常状态
    
    def _calculate_uav_idle_status(self, uav):
        """
        计算UAV的空闲状态
        
        空闲定义：UAV有资源但未分配任何任务
        为网络提供更直接的信号来理解哪些无人机可以被分配
        
        Args:
            uav: UAV对象
            
        Returns:
            float: 1.0表示空闲，0.0表示非空闲
        """
        # 检查UAV是否有资源
        if not np.any(uav.resources > 0):
            return 0.0  # 没有资源，不算空闲
        
        # 检查UAV是否有分配的任务
        has_tasks = any(uav.id in [uav_info[0] for uav_info in target.allocated_uavs] 
                       for target in self.targets)
        
        # 有资源但没有任务 = 空闲
        return 0.0 if has_tasks else 1.0
    
    def _calculate_target_visibility_status(self, target, target_idx: int) -> float:
        """
        计算目标的可见性状态，考虑感知范围和环境遮挡
        
        鲁棒性掩码机制的核心组件，用于标识目标的可见性状态。
        在实际部署中，这可以基于：
        - 传感器感知范围
        - 环境遮挡（建筑物、地形）
        - 天气条件
        - 目标特性（大小、反射率等）
        
        Args:
            target: 目标实体对象
            target_idx: 目标索引
            
        Returns:
            float: 可见性状态 (0.0=不可见, 1.0=完全可见)
        """
        # 基础可见性检查：目标是否还有剩余资源
        has_remaining_resources = np.any(target.remaining_resources > 0)
        if not has_remaining_resources:
            return 0.0
        
        # 计算最近UAV到目标的距离
        min_distance = float('inf')
        closest_uav_alive = False
        
        for i, uav in enumerate(self.uavs):
            if np.any(uav.resources > 0):  # UAV仍然活跃
                dist = np.linalg.norm(
                    np.array(target.position) - np.array(uav.current_position)
                )
                if dist < min_distance:
                    min_distance = dist
                    # 检查最近的UAV是否处于正常通信状态
                    closest_uav_alive = self._calculate_uav_alive_status(uav, i) > 0.5
        
        # 如果没有活跃的UAV，目标不可见
        if min_distance == float('inf') or not closest_uav_alive:
            return 0.0
        
        # 基于距离的可见性衰减
        max_sensing_range = getattr(self.config, 'MAX_SENSING_RANGE', 1000.0)
        if min_distance > max_sensing_range:
            return 0.0
        
        # 距离衰减函数：近距离完全可见，远距离逐渐衰减
        distance_visibility = max(0.0, 1.0 - (min_distance / max_sensing_range) ** 2)
        
        # 模拟环境遮挡影响
        occlusion_rate = getattr(self.config, 'TARGET_OCCLUSION_RATE', 0.0)
        if occlusion_rate > 0:
            # 基于目标位置和环境复杂度计算遮挡概率
            occlusion_seed = (self.step_count * 37 + target_idx * 41) % 1000
            occlusion_prob = occlusion_seed / 1000.0
            
            # 环境复杂度影响遮挡概率
            env_complexity = self._calculate_target_environment_complexity(target)
            adjusted_occlusion_rate = occlusion_rate * env_complexity
            
            if occlusion_prob < adjusted_occlusion_rate:
                distance_visibility *= 0.2  # 遮挡时可见性大幅下降
        
        # 模拟天气条件影响
        weather_visibility = getattr(self.config, 'WEATHER_VISIBILITY_FACTOR', 1.0)
        distance_visibility *= weather_visibility
        
        # 模拟目标特性影响（大小、反射率等）
        target_detectability = getattr(target, 'detectability_factor', 1.0)
        distance_visibility *= target_detectability
        
        # 确保返回值在[0, 1]范围内
        return float(np.clip(distance_visibility, 0.0, 1.0))
    
    def _calculate_environment_complexity(self, uav) -> float:
        """
        计算UAV周围环境的复杂度因子
        
        用于调整通信/感知失效概率。环境越复杂，失效概率越高。
        
        Args:
            uav: UAV实体对象
            
        Returns:
            float: 环境复杂度因子 [0.5, 2.0]
        """
        complexity = 1.0
        
        # 障碍物密度影响
        if hasattr(self, 'obstacles') and self.obstacles:
            nearby_obstacles = 0
            search_radius = 200.0  # 搜索半径
            
            for obstacle in self.obstacles:
                if hasattr(obstacle, 'position'):
                    dist = np.linalg.norm(
                        np.array(uav.current_position) - np.array(obstacle.position)
                    )
                    if dist < search_radius:
                        nearby_obstacles += 1
            
            # 障碍物密度因子
            obstacle_density = nearby_obstacles / max(1, len(self.obstacles))
            complexity += obstacle_density * 0.5
        
        # UAV密度影响（通信干扰）
        nearby_uavs = 0
        interference_radius = 150.0
        
        for other_uav in self.uavs:
            if other_uav.id != uav.id and np.any(other_uav.resources > 0):
                dist = np.linalg.norm(
                    np.array(uav.current_position) - np.array(other_uav.current_position)
                )
                if dist < interference_radius:
                    nearby_uavs += 1
        
        # UAV密度因子
        uav_density = nearby_uavs / max(1, len(self.uavs) - 1)
        complexity += uav_density * 0.3
        
        # 目标密度影响（感知负载）
        nearby_targets = 0
        sensing_radius = 300.0
        
        for target in self.targets:
            if np.any(target.remaining_resources > 0):
                dist = np.linalg.norm(
                    np.array(uav.current_position) - np.array(target.position)
                )
                if dist < sensing_radius:
                    nearby_targets += 1
        
        # 目标密度因子
        target_density = nearby_targets / max(1, len(self.targets))
        complexity += target_density * 0.2
        
        # 限制复杂度因子范围
        return float(np.clip(complexity, 0.5, 2.0))
    
    def _calculate_target_environment_complexity(self, target) -> float:
        """
        计算目标周围环境的复杂度因子
        
        用于调整目标遮挡概率。环境越复杂，遮挡概率越高。
        
        Args:
            target: 目标实体对象
            
        Returns:
            float: 环境复杂度因子 [0.5, 2.0]
        """
        complexity = 1.0
        
        # 障碍物遮挡影响
        if hasattr(self, 'obstacles') and self.obstacles:
            nearby_obstacles = 0
            occlusion_radius = 100.0  # 遮挡影响半径
            
            for obstacle in self.obstacles:
                if hasattr(obstacle, 'position'):
                    dist = np.linalg.norm(
                        np.array(target.position) - np.array(obstacle.position)
                    )
                    if dist < occlusion_radius:
                        nearby_obstacles += 1
            
            # 障碍物遮挡因子
            occlusion_density = nearby_obstacles / max(1, len(self.obstacles))
            complexity += occlusion_density * 0.8
        
        # 其他目标的干扰影响
        nearby_targets = 0
        interference_radius = 80.0
        
        for other_target in self.targets:
            if (other_target.id != target.id and 
                np.any(other_target.remaining_resources > 0)):
                dist = np.linalg.norm(
                    np.array(target.position) - np.array(other_target.position)
                )
                if dist < interference_radius:
                    nearby_targets += 1
        
        # 目标干扰因子
        target_interference = nearby_targets / max(1, len(self.targets) - 1)
        complexity += target_interference * 0.3
        
        # 限制复杂度因子范围
        return float(np.clip(complexity, 0.5, 2.0))
    
    def _calculate_robust_masks(self) -> Dict[str, np.ndarray]:
        """
        计算增强的鲁棒性掩码，结合is_alive和is_visible位
        
        掩码机制的核心功能：
        1. 基础有效性掩码：基于资源状态
        2. 通信/感知掩码：基于is_alive和is_visible位
        3. 组合掩码：为TransformerGNN提供失效节点屏蔽能力
        4. 使用固定维度确保批处理兼容性
        
        Returns:
            Dict[str, np.ndarray]: 包含多层掩码的字典
        """
        # 使用固定的最大数量，确保维度一致性
        max_uavs = getattr(self.config, 'MAX_UAVS', 10)
        max_targets = getattr(self.config, 'MAX_TARGETS', 15)
        
        n_uavs = len(self.uavs)
        n_targets = len(self.targets)
        
        # === 基础有效性掩码 ===
        # UAV基础掩码：基于资源状态，使用固定维度
        uav_resource_mask = np.zeros(max_uavs, dtype=np.int32)
        for i, uav in enumerate(self.uavs):
            uav_resource_mask[i] = 1 if np.any(uav.resources > 0) else 0
        
        # 目标基础掩码：基于剩余资源状态，使用固定维度
        target_resource_mask = np.zeros(max_targets, dtype=np.int32)
        for i, target in enumerate(self.targets):
            target_resource_mask[i] = 1 if np.any(target.remaining_resources > 0) else 0
        
        # === 通信/感知掩码 ===
        # UAV通信掩码：基于is_alive位，使用固定维度
        uav_communication_mask = np.zeros(max_uavs, dtype=np.int32)
        for i, uav in enumerate(self.uavs):
            uav_communication_mask[i] = 1 if self._calculate_uav_alive_status(uav, i) > 0.5 else 0
        
        # 目标可见性掩码：基于is_visible位，使用固定维度
        target_visibility_mask = np.zeros(max_targets, dtype=np.int32)
        for i, target in enumerate(self.targets):
            target_visibility_mask[i] = 1 if self._calculate_target_visibility_status(target, i) > 0.5 else 0
        
        # === 组合掩码（用于TransformerGNN） ===
        # UAV有效掩码：同时满足资源和通信条件
        uav_effective_mask = uav_resource_mask & uav_communication_mask
        
        # 目标有效掩码：同时满足资源和可见性条件
        target_effective_mask = target_resource_mask & target_visibility_mask
        
        # === 交互掩码 ===
        # UAV-目标交互掩码 [max_uavs, max_targets]：标识哪些UAV-目标对可以进行有效交互
        interaction_mask = np.zeros((max_uavs, max_targets), dtype=np.int32)
        
        for i in range(n_uavs):
            for j in range(n_targets):
                # 只有当UAV有效且目标有效时，才能进行交互
                if uav_effective_mask[i] == 1 and target_effective_mask[j] == 1:
                    # 额外检查距离约束
                    uav = self.uavs[i]
                    target = self.targets[j]
                    dist = np.linalg.norm(
                        np.array(target.position) - np.array(uav.current_position)
                    )
                    max_interaction_range = getattr(self.config, 'MAX_INTERACTION_RANGE', 2000.0)
                    
                    if dist <= max_interaction_range:
                        interaction_mask[i, j] = 1
        
        # 构建完整的掩码字典
        masks = {
            # 基础掩码（向后兼容）
            "uav_mask": uav_effective_mask,
            "target_mask": target_effective_mask,
            
            # 详细掩码（用于调试和分析）
            "uav_resource_mask": uav_resource_mask,
            "uav_communication_mask": uav_communication_mask,
            "target_resource_mask": target_resource_mask,
            "target_visibility_mask": target_visibility_mask,
            
            # 交互掩码（用于TransformerGNN的注意力计算）
            "interaction_mask": interaction_mask,
            
            # 统计信息（用于监控和调试）
            "active_uav_count": np.sum(uav_effective_mask),
            "visible_target_count": np.sum(target_effective_mask),
            "total_interactions": np.sum(interaction_mask)
        }
        
        return masks