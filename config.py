# -*- coding: utf-8 -*-
# 文件名: config.py
# 描述: 统一管理项目的所有配置参数，包括训练配置和PBRS配置
#
# 日志输出控制说明:
# ===================
# 
# 本文件新增了分级日志输出控制功能，包括：
# 
# 1. 日志级别控制:
#    - LOG_LEVEL: 日志输出级别 ('minimal', 'simple', 'detailed', 'debug')
#    - LOG_EPISODE_DETAIL: 是否输出轮次内步的详细信息
#    - LOG_REWARD_DETAIL: 是否输出奖励分解详细信息
# 
# 2. 日志级别说明:
#    - 'minimal': 只输出关键信息，包括训练模式、训练参数、每个轮次的基本数据
#    - 'simple': 输出简洁模式信息，包含轮次数据和简单分解
#    - 'detailed': 输出详细信息，包含完整的奖励分解和调试信息
#    - 'debug': 输出所有调试信息，包括步级别的详细信息
# 
# 3. 向后兼容:
#    - ENABLE_DEBUG: 保持向后兼容，映射到LOG_LEVEL
#    - ENABLE_SCENARIO_DEBUG: 场景调试信息控制
# 
# 使用示例:
# --------
# config = Config()
# config.set_log_level('simple')  # 设置为简洁模式
# config.set_log_level('detailed')  # 设置为详细模式
# config.set_log_level('minimal')  # 设置为最小输出模式

import os
import pickle
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class Hyperparameters:
    """模型网络结构参数类 - 统一管理所有网络架构相关参数"""
    
    # ===== 网络结构参数 =====
    hidden_dim: int = 256                   # GNN隐藏层维度
    num_layers: int = 3                     # 网络层数
    num_heads: int = 8                      # 注意力头数（用于Transformer/GAT）
    dropout_rate: float = 0.1               # Dropout比例
    
    # ===== 图网络特定参数 =====
    node_feature_dim: int = 64              # 节点特征维度
    edge_feature_dim: int = 32              # 边特征维度
    graph_pooling: str = "mean"             # 图池化方式 ("mean", "max", "sum")
    
    # ===== 激活函数和正则化 =====
    activation: str = "relu"                # 激活函数类型
    use_batch_norm: bool = True             # 是否使用批归一化
    use_layer_norm: bool = False            # 是否使用层归一化
    
    # ===== 输出层参数 =====
    output_activation: str = "linear"       # 输出层激活函数
    use_dueling: bool = True                # 是否使用Dueling架构

@dataclass
class TrainingConfig:
    """训练配置类 - 统一管理所有训练过程参数"""
    
    # ===== 基础训练参数 =====
    episodes: int = 3000                    # 训练轮次 - 观察奖励收敛性
    learning_rate: float = 0.00005         # 降低学习率，提高数值稳定性
    gamma: float = 0.99                    # 提高折扣因子，更重视长期奖励
    batch_size: int = 256                   # 较大的批次大小，提高训练效率 256-32%显存  
    memory_size: int = 15000               # 适当减小记忆库，避免过旧经验
    
    # ===== 探索策略参数 =====
    epsilon_start: float = 0.9             # 降低初始探索率
    epsilon_end: float = 0.01               # 提高最终探索率，保持适度探索
    epsilon_decay: float = 0.99995          # 放缓探索率衰减
    epsilon_min: float = 0.01               # 提高最小探索率
    
    # ===== 网络更新参数 =====
    patience: int = 300                    # 大幅增加早停耐心值，支持更长时间训练
    target_update_freq: int = 20           # 降低目标网络更新频率，增加稳定性    
    log_interval: int = 20                 # 减少日志输出频率
    max_best_models: int = 5               # 保存最优模型的数量，默认为5
    
    # ===== 梯度裁剪参数 =====
    use_gradient_clipping: bool = True     # 是否使用梯度裁剪
    max_grad_norm: float = 1.0             # 最大梯度范数
    
    # ===== 优先经验回放参数 =====
    use_prioritized_replay: bool = True    # 是否使用优先经验回放
    per_alpha: float = 0.6                 # 优先级指数 (0=均匀采样, 1=完全优先级采样)
    per_beta_start: float = 0.4            # 重要性采样权重初始值
    per_beta_frames: int = 100000          # β从初始值增长到1.0的帧数
    per_epsilon: float = 1e-6              # 防止优先级为0的小值
    
    # ===== 调试参数 =====
    verbose: bool = True                   # 详细输出
    debug_mode: bool = False               # 调试模式
    save_training_history: bool = True     # 保存训练历史

class Config:
    """统一管理所有算法和模拟的参数"""
    
    def __init__(self):
        # 不检查Nan
        # self.debug_mode = False  
        # ----- 训练系统控制参数 -----
        # 训练模式选择：
        # - 'training': 训练模式，从头开始训练或继续训练
        # - 'inference': 推理模式，仅加载已训练模型进行推理
        # - 'zero_shot_train': 零样本训练模式，专用于ZeroShotGNN
        self.TRAINING_MODE = 'zero_shot_train'
        
        # --- 智能化训练参数 (Intelligent Training Parameters) --- 
        self.TOP_K_UAVS = 5  # 动作空间剪枝的K值 
        self.APPROACH_REWARD_COEFFICIENT = 0.001  # 接近激励奖励的系数 
        self.STAGNATION_THRESHOLD = 10  # 训练停滞提前终止的阈值
        
        # 强制重新训练标志：
        # - True: 忽略已有模型，强制重新训练
        # - False: 优先加载已有模型，不存在时才训练
        self.FORCE_RETRAIN = True
        
        # 路径规划精度控制：
        # - True: 使用高精度PH-RRT算法，计算准确但耗时
        # - False: 使用快速近似算法，计算快速但精度较低
        self.USE_PHRRT_DURING_TRAINING = False          # 训练时是否使用高精度PH-RRT
        self.USE_PHRRT_DURING_PLANNING = True          # 规划时是否使用高精度PH-RRT
        
        # 距离计算服务配置
        self.ENABLE_DISTANCE_CACHE = True              # 启用距离计算缓存
        self.DISTANCE_CACHE_SIZE = 10000               # 缓存大小限制
        self.DISTANCE_PRECISION = 2                    # 距离计算精度（小数位数）
        
        # 模型保存/加载路径配置
        self.SAVED_MODEL_PATH = 'output/models/saved_model_final.pth'
        
        # ----- 分级日志输出控制配置 -----
        # 日志输出级别控制 ('minimal', 'simple', 'detailed', 'debug')
        self.LOG_LEVEL = 'detailed'#'simple'#                       # 默认简洁模式
        self.LOG_EPISODE_DETAIL = False                # 是否输出轮次内步的详细信息
        self.LOG_REWARD_DETAIL = False                  # 是否输出奖励分解详细信息
        
        # 向后兼容的调试参数（自动映射到LOG_LEVEL）
        self.ENABLE_DEBUG = False                       # 启用通用调试信息（向后兼容）
        self.ENABLE_SCENARIO_DEBUG = False              # 启用场景生成调试信息
        
        # ----- 场景记录配置 -----
        # 场景数据记录控制
        self.SAVE_SCENARIO_DATA = True              # 是否保存场景数据
        self.SCENARIO_DATA_FORMAT = 'txt'           # 场景数据格式: 'pkl', 'txt', 'both'
        self.SCENARIO_DATA_DIR = 'output/scenario_logs'  # 场景数据保存目录
        
        # 场景生成验证配置
        self.VALIDATE_SCENARIO_CONSTRAINTS = True   # 是否验证场景约束
        self.SCENARIO_GENERATION_MAX_RETRIES = 10   # 场景生成最大重试次数
        
        # ----- 训练场景数据记录配置 -----
        # 训练中场景数据记录
        self.SAVE_TRAINING_SCENARIO_DATA = True         # 是否保存训练中的每个场景数据
        self.TRAINING_SCENARIO_LOG_FORMAT = 'detailed'  # 场景记录格式: 'simple', 'detailed', 'both'
        self.TRAINING_SCENARIO_LOG_DIR = 'output/training_scenario_logs'  # 训练场景数据保存目录
        
        # 训练后推理结果记录
        self.SAVE_INFERENCE_RESULTS = True              # 是否保存训练后的推理结果
        self.INFERENCE_RESULTS_FORMAT = 'json'          # 推理结果格式: 'json', 'txt', 'both'
        self.INFERENCE_RESULTS_DIR = 'output/inference_results'  # 推理结果保存目录
        self.INFERENCE_RESULTS_INCLUDE_SCENARIO = True  # 推理结果是否包含对应的场景数据
        
        # ----- 训练轮次推理结果记录配置 -----
        # 每轮次推理结果记录控制
        self.SAVE_EPISODE_INFERENCE_RESULTS = True      # 是否保存每轮次的推理结果
        self.EPISODE_INFERENCE_LOG_FORMAT = 'detailed'  # 轮次推理日志记录格式: 'simple', 'detailed', 'both'
        self.EPISODE_INFERENCE_INCLUDE_TASK_ALLOCATION = True  # 是否包含详细的任务分配方案
        self.EPISODE_INFERENCE_INCLUDE_PERFORMANCE_METRICS = True  # 是否包含性能指标
        self.EPISODE_INFERENCE_INCLUDE_COMPLETE_VISUALIZATION = True  # 是否包含完整可视化结果
        self.EPISODE_INFERENCE_LOG_INTERVAL = 1         # 推理结果记录间隔（每N轮记录一次）
        
        # ----- 统一的日志文件命名配置 -----
        # 日志文件命名规则统一
        self.LOG_FILE_NAMING_PATTERN = '{timestamp}_{network_type}_{scenario}'  # 日志文件命名模式
        self.LOG_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'     # 时间戳格式
        self.UNIFIED_LOG_NAMING = True                   # 是否启用统一的日志命名规则
        
        # action_log文件命名与reward_log保持一致
        self.ACTION_LOG_USE_UNIFIED_NAMING = True        # action_log是否使用统一命名
        self.REWARD_LOG_USE_UNIFIED_NAMING = True        # reward_log是否使用统一命名
        
        # ----- 训练曲线配置 -----
        # 训练曲线标题配置
        self.TRAINING_CURVE_INCLUDE_EARLY_STOP_INFO = True  # 训练曲线标题是否包含早停信息
        self.TRAINING_CURVE_TITLE_FORMAT = '{network_type} 训练曲线 - {scenario}场景 (早停轮数: {early_stop_episode})'
        self.TRAINING_CURVE_SAVE_FORMAT = ['png', 'pdf']    # 训练曲线保存格式
        self.TRAINING_CURVE_DPI = 300                       # 训练曲线图片分辨率
        
        # 训练进度记录配置
        self.SAVE_TRAINING_PROGRESS = True               # 是否保存详细的训练进度
        self.TRAINING_PROGRESS_LOG_INTERVAL = 10         # 训练进度记录间隔
        self.TRAINING_PROGRESS_INCLUDE_METRICS = True    # 是否在进度中包含详细指标
        
        # ----- 网络结构选择参数 -----
        # 网络结构类型选择，支持以下候选项：
        # - 'SimpleNetwork': 基础全连接网络，适合简单场景，训练快速
        # - 'DeepFCN': 深度全连接网络，具有更强的表达能力
        # - 'DeepFCNResidual': 带残差连接的深度网络，缓解梯度消失问题
        # - 'ZeroShotGNN': 零样本图神经网络，具有泛化能力，适合不同规模场景
        # - 'GAT': 图注意力网络，专注于图结构数据处理
        self.NETWORK_TYPE = 'ZeroShotGNN'    # 切换到ZeroShotGNN进行稳定性调试

        # ----- 改进 ZeroShotGNN奖励函数 -----
        self.USE_IMPROVED_REWARD = True  # 启用改进版奖励函数
        
        # ----- 路径规划参数 -----
        # RRT算法核心参数：
        self.RRT_ITERATIONS = 1000          # RRT最大迭代次数，影响路径质量和计算时间
        self.RRT_STEP_SIZE = 50.0           # RRT单步扩展距离，影响路径平滑度
        self.RRT_GOAL_BIAS = 0.1            # 目标偏向概率(0-1)，越大越快收敛但可能陷入局部最优
        self.RRT_ADAPTIVE_STEP = True       # 自适应步长：True=根据环境调整，False=固定步长
        self.RRT_OBSTACLE_AWARE = True      # 障碍物感知采样：True=避开障碍物，False=随机采样
        self.RRT_MAX_ATTEMPTS = 3           # 路径规划失败时的最大重试次数
        
        # ===== PH曲线平滑参数 =====
        self.MAX_REFINEMENT_ATTEMPTS = 5    # 最大细化尝试次数
        self.BEZIER_SAMPLES = 50            # 贝塞尔曲线采样点数
        self.OBSTACLE_TOLERANCE = 50.0      # 障碍物的安全容忍距离

        # ----- 图构建参数 -----
        # 图结构离散化参数：
        self.GRAPH_N_PHI = 1                # 每个目标节点的离散化接近角度数量，影响动作空间大小（默认值设为1，降低动作空间复杂度）

        # ----- 环境维度参数 -----
        # 环境规模限制（用于张量维度统一）：
        self.MAX_UAVS = 25                  # 支持的最大UAV数量，超出会截断**
        self.MAX_TARGETS = 15               # 支持的最大目标数量，超出会截断**
        self.MAP_SIZE = 1000.0              # 地图边长(米)，用于坐标归一化
        self.MAX_INTERACTION_RANGE = 2000.0 # UAV最大交互距离(米)，超出视为无效
        self.RESOURCE_DIM = 2                 # 资源维度，例如：货物和燃料
        self.UAV_MAX_DISTANCE = 1000.0        # 无人机最大续航距离
        self.UAV_VELOCITY_RANGE = [10.0, 50.0] # 无人机速度范围 [min, max]
        self.UAV_ECONOMIC_SPEED = 20.0        # 无人机经济巡航速度

        self.CRITICAL_UAV_MIN_THRESHOLD = 2
        self.CRITICAL_UAV_SCALING_FACTOR = 0.15

        # 统一的、分级的场景生成模板
        self.SCENARIO_TEMPLATES = {
            'easy': {
                'uav_num_range': (3, 7),
                'target_num_range': (2, 4),
                'obstacle_num_range': (1, 5),
                'resource_abundance_range': (1.0, 1.15)
            },
            'medium': {
                'uav_num_range': (7, 15),
                'target_num_range': (4, 7),
                'obstacle_num_range': (6, 12),
                'resource_abundance_range': (1.1, 1.3)
            },
            'hard': {
                'uav_num_range': (16, 25),      # 使用最大上限
                'target_num_range': (7, 15), # 使用最大上限
                'obstacle_num_range': (13, 20),
                'resource_abundance_range': (1.0, 1.4)
            }
        }
        
        # ----- 自适应课程训练参数 -----
        # 自适应课程训练功能开关和核心参数
        self.CURRICULUM_ENABLE_ADAPTIVE = True          # 启用自适应课程训练机制   
        
        # 高级自适应参数
        self.CURRICULUM_REGRESSION_THRESHOLD = 0.40     # 课程退步阈值，低于此值考虑降级
        self.CURRICULUM_ENABLE_REGRESSION = False       # 启用降级机制（可选高级功能）
        self.CURRICULUM_PROMOTION_COOLDOWN = 5          # 晋级冷却期，防止频繁晋级
        self.CURRICULUM_MAX_REGRESSION_COUNT = 2        # 最大降级次数限制
        
        # 自适应训练监控参数
        self.CURRICULUM_LOG_DETAILED_PERFORMANCE = True # 记录详细的性能数据
        self.CURRICULUM_SAVE_LEVEL_CHECKPOINTS = True   # 保存每个等级的检查点
        self.CURRICULUM_PERFORMANCE_SMOOTHING = 0.1     # 性能指标平滑系数
        # ----- 渐进式自适应课程学习 (Granular Adaptive Curriculum) -----
        # 精细化课程训练参数
        self.CURRICULUM_USE_GRANULAR_PROGRESSION = True  # True: 启用渐进式课程, False: 使用原有模板
        self.GRANULAR_CURRICULUM_LEVELS = 15  #3#              # 课程的总等级数量
        self.CURRICULUM_MASTERY_THRESHOLD = 0.85 #0.50 #        # 课程掌握度阈值，达到此完成率视为掌握
        self.CURRICULUM_PERFORMANCE_WINDOW = 20  #5 #          # 性能评估滑动窗口大小
        self.CURRICULUM_MAX_EPISODES_PER_LEVEL = 500 # 20#      # 单个难度等级最大训练轮次
        self.CURRICULUM_MIN_EPISODES_PER_LEVEL = 30 # 3#       # 单个难度等级最小训练轮次
        
        
        # --- 课程起点参数 ---
        self.GRANULAR_START_UAVS = 3                     # 起始UAV数量
        self.GRANULAR_START_TARGETS = 2                  # 起始目标数量
        self.GRANULAR_START_OBSTACLES = 1                # 起始障碍物数量
        self.GRANULAR_START_ABUNDANCE = 1.5              # 起始资源充裕度 (1.5倍需求)

        # --- 课程终点参数 (将直接从全局配置读取 MAX_UAVS 和 MAX_TARGETS) ---
        self.GRANULAR_END_OBSTACLES = 20                 # 最终障碍物数量
        self.GRANULAR_END_ABUNDANCE = 1.0                # 最终资源充裕度 (1.0倍需求)  

        # ----- 掌握度阈值自适应调整机制 -----
        self.ADAPTIVE_THRESHOLD_ENABLED = True           # True: 启用阈值自适应调整
        self.ADAPTIVE_THRESHOLD_INITIAL = 0.90           # 初始掌握度阈值
        self.ADAPTIVE_THRESHOLD_MIN = 0.85               # 阈值可降低到的最小值，防止标准过低
        self.ADAPTIVE_THRESHOLD_MAX = 0.95               # 阈值可提升到的最大值 (为未来功能预留)

        # --- 停滞判断参数 (用于自动降低阈值) ---
        self.ADAPTIVE_THRESHOLD_STAGNATION_TRIGGER = 0.7 # 当训练超过等级最大轮次的70%时，开始检查停滞
        self.ADAPTIVE_THRESHOLD_PLATEAU_STD_DEV = 0.03   # 当性能窗口内完成率的标准差低于此值，视为进入平台期
        self.ADAPTIVE_THRESHOLD_DOWN_STEP = 0.02         # 每次自动降低的步长 (例如从0.85降至0.83)
        
        # ----- 阈值自适应调整高级策略 -----
        # 1. 晋级后不稳定处理 (用于自动升高阈值)
        self.ADAPTIVE_THRESHOLD_INSTABILITY_CHECK = True # True: 启用晋级后表现检查
        self.ADAPTIVE_THRESHOLD_INSTABILITY_DROP = 0.30  # 晋级后，若完成率下跌超过30%，则认为基础不牢
        self.ADAPTIVE_THRESHOLD_UP_STEP = 0.02           # 每次自动升高的步长 (例如从0.85升至0.87)

        # 2. 更精确的平台期判断
        self.ADAPTIVE_THRESHOLD_USE_SLOPE = True         # True: 启用趋势斜率辅助判断平台期
        self.ADAPTIVE_THRESHOLD_PLATEAU_SLOPE = 0.005    # 当性能窗口的趋势斜率低于此值，视为无明显增长

        # ----- 渐进式自适应课程学习 (Granular Adaptive Curriculum) -----
              
        # ----- 模拟与评估参数 -----
        # 可视化控制：
        self.SHOW_VISUALIZATION = False     # 是否显示matplotlib可视化图表
        
        # 负载均衡参数：
        self.LOAD_BALANCE_PENALTY = 0.1     # 负载不均衡惩罚系数(0-1)，越大越重视均衡

        # ----- 奖励函数参数 -----
        self.TARGET_COMPLETION_REWARD = 1500    # 目标完成奖励
        self.MARGINAL_UTILITY_FACTOR = 1000    # 边际效用因子
        self.EFFICIENCY_REWARD_FACTOR = 500     # 效率奖励因子
        self.DISTANCE_PENALTY_FACTOR = 0.1     # 距离惩罚因子
        self.TIME_PENALTY_FACTOR = 10          # 时间惩罚因子
        self.COMPLETION_REWARD = 1000          # 完成奖励
        self.INVALID_ACTION_PENALTY = -100     # 无效动作惩罚
        self.ZERO_CONTRIBUTION_PENALTY = -50   # 零贡献惩罚
        self.DEADLOCK_PENALTY = -200           # 死锁惩罚
        self.COLLABORATION_BONUS = 200         # 协作奖励
        

        self.DEADLOCK_PENALTY = -200           # <--- [新增] 死局惩罚

        # ----- PBRS (Potential-Based Reward Shaping) 参数 -----
        # PBRS功能开关 (暂时禁用，回到稳定基线)
        self.ENABLE_PBRS = True                        # 启用PBRS，使用协同战备模式
        self.PBRS_TYPE = 'synergy'                       # PBRS类型: 'simple'(完成目标数), 'progress'(资源进度), 或 'synergy'(协同战备)
        self.ENABLE_REWARD_LOGGING = True               # 是否保存最新的奖励组成用于调试和监控
        
        # 势函数权重参数
        self.PBRS_COMPLETION_WEIGHT = 50.0              # 完成度势能权重
        self.PBRS_DISTANCE_WEIGHT = 0.01                # 距离势能权重
        self.PBRS_COLLABORATION_WEIGHT = 5.0            # 协作势能权重
        
        # 奖励裁剪参数 (极保守版本)
        self.PBRS_REWARD_CLIP_MIN = -30.0                # 塑形奖励最小值 
        self.PBRS_REWARD_CLIP_MAX = 30.0                 # 塑形奖励最大值
        self.PBRS_POTENTIAL_SCALE = 0.5                # 势函数缩放因子 (极小影响)
        self.PBRS_WARMUP_EPISODES = 100                 # PBRS预热期 (前100轮不使用)
        
        # 调试参数
        self.PBRS_DEBUG_MODE = True                    # PBRS调试模式
        self.PBRS_LOG_POTENTIAL_VALUES = True          # 是否记录势函数值
        self.PBRS_LOG_REWARD_BREAKDOWN = True          # 是否记录奖励组成详情
        
        # 数值稳定性参数
        self.PBRS_POTENTIAL_CLIP_MIN = -1000.0          # 势函数值最小值
        self.PBRS_POTENTIAL_CLIP_MAX = 1000.0           # 势函数值最大值
        self.PBRS_ENABLE_GRADIENT_CLIPPING = True       # 是否启用梯度裁剪
        self.PBRS_MAX_POTENTIAL_CHANGE = 100.0          # 单步最大势函数变化量
        
        # 缓存和性能参数
        self.PBRS_ENABLE_DISTANCE_CACHE = True          # 是否启用距离缓存
        self.PBRS_CACHE_UPDATE_THRESHOLD = 0.1          # 缓存更新阈值
        
        # ----- 紧急稳定性修复参数 -----
        # 奖励归一化优化
        self.REWARD_NORMALIZATION = True           # 启用奖励归一化
        self.REWARD_SCALE = 0.3                    # 从0.1提升到0.3 (3倍)
        
        # 数值稳定性检查
        self.ENABLE_NUMERICAL_STABILITY_CHECKS = True  # 启用数值稳定性检查
        
        # 调试相关配置
        self.ENABLE_REWARD_DEBUG = False  # 启用奖励计算调试
        self.ENABLE_GRADIENT_DEBUG = True  # 启用梯度调试
        self.ENABLE_LOSS_DEBUG = True  # 启用损失调试
        
        # ----- 配置对象 -----
        self.hyperparameters = Hyperparameters()
        self.training_config = TrainingConfig()
        
        # 根据网络类型设置优化的参数配置
        self._setup_network_specific_params()
        
        # 设置统一的训练参数访问接口
        self._setup_unified_training_params()
        
        # 验证PBRS配置
        self._validate_pbrs_on_init()
        
        # 验证自适应课程训练配置
        self._validate_adaptive_curriculum_on_init()
        
        # 初始化日志级别映射
        self._setup_log_level_mapping()
       
    def _setup_log_level_mapping(self):
        """设置日志级别映射，确保向后兼容"""
        # 根据LOG_LEVEL自动设置相关参数
        if self.LOG_LEVEL == 'minimal':
            self.LOG_EPISODE_DETAIL = True
            self.LOG_REWARD_DETAIL = False
            self.ENABLE_DEBUG = False
            self.ENABLE_SCENARIO_DEBUG = False
        elif self.LOG_LEVEL == 'simple':
            self.LOG_EPISODE_DETAIL = True
            self.LOG_REWARD_DETAIL = True
            self.ENABLE_DEBUG = False
            self.ENABLE_SCENARIO_DEBUG = True
        elif self.LOG_LEVEL == 'detailed':
            self.LOG_EPISODE_DETAIL = True
            self.LOG_REWARD_DETAIL = True
            self.ENABLE_DEBUG = True
            self.ENABLE_SCENARIO_DEBUG = True
        elif self.LOG_LEVEL == 'debug':
            self.LOG_EPISODE_DETAIL = True
            self.LOG_REWARD_DETAIL = True
            self.ENABLE_DEBUG = True
            self.ENABLE_SCENARIO_DEBUG = True
    
    def set_log_level(self, level: str):
        """
        设置日志输出级别
        
        Args:
            level: 日志级别 ('minimal', 'simple', 'detailed', 'debug')
        """
        valid_levels = ['minimal', 'simple', 'detailed', 'debug']
        if level not in valid_levels:
            raise ValueError(f"无效的日志级别: {level}。有效级别: {valid_levels}")
        
        self.LOG_LEVEL = level
        self._setup_log_level_mapping()
        print(f"✓ 日志级别已设置为: {level}")
    
    def get_log_level_info(self) -> Dict[str, Any]:
        """获取当前日志级别配置信息"""
        return {
            "当前级别": self.LOG_LEVEL,
            "轮次内步详情": self.LOG_EPISODE_DETAIL,
            "奖励分解详情": self.LOG_REWARD_DETAIL,
            "调试模式": self.ENABLE_DEBUG,
            "场景调试": self.ENABLE_SCENARIO_DEBUG
        }
    
    def print_log_config(self):
        """打印日志配置信息"""
        print("=" * 60)
        print("日志输出配置")
        print("=" * 60)
        
        info = self.get_log_level_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n级别说明:")
        print("  minimal  - 只输出关键信息")
        print("  simple   - 输出简洁模式信息")
        print("  detailed - 输出详细信息")
        print("  debug    - 输出所有调试信息")
        
        print("\n" + "=" * 60)
    
    def _setup_network_specific_params(self):
        """根据网络类型设置优化的参数配置"""
        
        if self.NETWORK_TYPE == 'DeepFCN':
            # DeepFCN稳定训练参数 (经过测试验证的最佳配置)
            print(f"🎯 应用DeepFCN稳定训练参数配置")
            self.training_config.learning_rate = 1e-05              # 极低学习率，避免训练震荡
            self.training_config.gradient_clip_norm = 0.5           # 严格梯度裁剪，防止梯度爆炸
            self.training_config.weight_decay = 2e-05               # 高正则化，防止过拟合和数值不稳定
            self.training_config.target_update_frequency = 1500     # 稳定的目标网络更新
            # self.training_config.batch_size = 64                   # 中等批次大小，平衡稳定性和效率
            self.training_config.epsilon_decay = 0.995             # 平滑探索衰减
            self.training_config.epsilon_min = 0.05                # 保持最小探索
            
        elif self.NETWORK_TYPE == 'ZeroShotGNN':
            # ZeroShotGNN优化配置 (基于问题分析)
            print(f"🚀 应用ZeroShotGNN优化配置")
            self.training_config.learning_rate = 1e-04              # 提高学习率，加快学习速度（从1e-05提高到1e-04）
            self.training_config.gradient_clip_norm = 0.5           # 严格梯度裁剪，防止图网络梯度不稳定
            self.training_config.weight_decay = 2e-05               # 高正则化，增强稳定性
            self.training_config.target_update_frequency = 2000     # 更稳定的目标网络更新
            # self.training_config.batch_size = 16                   # 小批次，减少计算开销
            self.training_config.epsilon_decay = 0.998             # 更慢的探索衰减，适合图网络
            self.training_config.epsilon_min = 0.1                 # 保持较高最小探索
            
            # 经验回放优化检查
            if self.training_config.batch_size <= 32:
                print(f"   ⚠️  批次大小({self.training_config.batch_size})较小，建议启用优先经验回放以提高样本利用效率")
                if not self.training_config.use_prioritized_replay:
                    self.training_config.use_prioritized_replay = True
                    print(f"   ✅ 已自动启用优先经验回放(PER)优化")
            else:
                print(f"   ✅ 批次大小({self.training_config.batch_size})适中，经验回放优化正常")
        elif self.NETWORK_TYPE == 'SimpleNetwork':
            # SimpleNetwork基础配置
            print(f"⚡ 应用SimpleNetwork基础参数配置")
            self.training_config.learning_rate = 1e-04              # 标准学习率
            self.training_config.gradient_clip_norm = 1.0           # 标准梯度裁剪
            self.training_config.weight_decay = 1e-06               # 低正则化
            self.training_config.target_update_frequency = 500      # 频繁更新
            self.training_config.batch_size = 128                  # 大批次
            self.training_config.epsilon_decay = 0.995             # 标准衰减
            self.training_config.epsilon_min = 0.01                # 低最小探索
            
        elif self.NETWORK_TYPE == 'DeepFCNResidual':
            # DeepFCNResidual配置 (基于DeepFCN优化)
            print(f"🚀 应用DeepFCNResidual参数配置")
            self.training_config.learning_rate = 2e-05              # 略高于DeepFCN
            self.training_config.gradient_clip_norm = 0.8           # 适中梯度裁剪
            self.training_config.weight_decay = 1e-05               # 中等正则化
            self.training_config.target_update_frequency = 1200     # 适中更新频率
            # self.training_config.batch_size = 64                   # 与DeepFCN相同
            self.training_config.epsilon_decay = 0.996             # 略快衰减
            self.training_config.epsilon_min = 0.05                # 标准最小探索
            
        else:
            # 默认配置
            print(f"⚠️ 使用默认参数配置 (网络类型: {self.NETWORK_TYPE})")
            self.training_config.learning_rate = 1e-04
            self.training_config.gradient_clip_norm = 1.0
            self.training_config.weight_decay = 1e-05
            self.training_config.target_update_frequency = 1000
            # self.training_config.batch_size = 64
            self.training_config.epsilon_decay = 0.995
            self.training_config.epsilon_min = 0.05
        
        # 简化输出，不显示详细参数
    
    def _setup_unified_training_params(self):
        """
        设置统一的训练参数访问接口
        所有训练相关参数都通过training_config统一管理，避免重复定义
        """
        # 为了向后兼容，提供属性访问接口
        pass
    
    def _validate_pbrs_on_init(self):
        """在初始化时验证PBRS配置"""
        # 静默验证PBRS配置
        if not self.validate_pbrs_config():
            self.reset_pbrs_to_defaults()
    
    def _validate_adaptive_curriculum_on_init(self):
        """在初始化时验证自适应课程训练配置"""
        # 验证自适应课程训练配置
        if not self.validate_adaptive_curriculum_config():
            print("⚠️ 自适应课程训练配置验证失败，重置为默认值")
            self.reset_adaptive_curriculum_to_defaults()
    
    # ===== 统一的训练参数访问属性 =====
    @property
    def EPISODES(self):
        return self.training_config.episodes
    
    @EPISODES.setter
    def EPISODES(self, value):
        self.training_config.episodes = value
    
    @property
    def LEARNING_RATE(self):
        return self.training_config.learning_rate
    
    @LEARNING_RATE.setter
    def LEARNING_RATE(self, value):
        self.training_config.learning_rate = 1e-05
    
    @property
    def GAMMA(self):
        return self.training_config.gamma
    
    @GAMMA.setter
    def GAMMA(self, value):
        self.training_config.gamma = value
    
    @property
    def BATCH_SIZE(self):
        return self.training_config.batch_size
    
    @BATCH_SIZE.setter
    def BATCH_SIZE(self, value):
        self.training_config.batch_size = value
    
    @property
    def MEMORY_SIZE(self):
        return self.training_config.memory_size
    
    @MEMORY_SIZE.setter
    def MEMORY_SIZE(self, value):
        self.training_config.memory_size = value
    
    @property
    def MEMORY_CAPACITY(self):
        return self.training_config.memory_size
    
    @MEMORY_CAPACITY.setter
    def MEMORY_CAPACITY(self, value):
        self.training_config.memory_size = value
    
    @property
    def EPSILON_START(self):
        return self.training_config.epsilon_start
    
    @EPSILON_START.setter
    def EPSILON_START(self, value):
        self.training_config.epsilon_start = value
    
    @property
    def EPSILON_END(self):
        return self.training_config.epsilon_end
    
    @EPSILON_END.setter
    def EPSILON_END(self, value):
        self.training_config.epsilon_end = value
    
    @property
    def EPSILON_DECAY(self):
        return self.training_config.epsilon_decay
    
    @EPSILON_DECAY.setter
    def EPSILON_DECAY(self, value):
        self.training_config.epsilon_decay = value
    
    @property
    def EPSILON_MIN(self):
        return self.training_config.epsilon_min
    
    @EPSILON_MIN.setter
    def EPSILON_MIN(self, value):
        self.training_config.epsilon_min = value
    
    @property
    def TARGET_UPDATE_FREQ(self):
        return self.training_config.target_update_freq
    
    @TARGET_UPDATE_FREQ.setter
    def TARGET_UPDATE_FREQ(self, value):
        self.training_config.target_update_freq = value
    
    @property
    def PATIENCE(self):
        return self.training_config.patience
    
    @PATIENCE.setter
    def PATIENCE(self, value):
        self.training_config.patience = value
    
    @property
    def LOG_INTERVAL(self):
        return self.training_config.log_interval
    
    @LOG_INTERVAL.setter
    def LOG_INTERVAL(self, value):
        self.training_config.log_interval = value
    
    @property
    def MAX_BEST_MODELS(self):
        return self.training_config.max_best_models
    
    @MAX_BEST_MODELS.setter
    def MAX_BEST_MODELS(self, value):
        self.training_config.max_best_models = value
    
    # ===== PH-RRT路径规划参数访问属性 =====
    @property
    def USE_PHRRT_TRAINING(self):
        """训练时是否使用PH-RRT*算法"""
        return self.USE_PHRRT_DURING_TRAINING
    
    @USE_PHRRT_TRAINING.setter
    def USE_PHRRT_TRAINING(self, value):
        self.USE_PHRRT_DURING_TRAINING = value
    
    @property
    def USE_PHRRT_PLANNING(self):
        """规划时是否使用PH-RRT*算法"""
        return self.USE_PHRRT_DURING_PLANNING
    
    @USE_PHRRT_PLANNING.setter
    def USE_PHRRT_PLANNING(self, value):
        self.USE_PHRRT_DURING_PLANNING = value
    
    # ===== 便捷的参数修改方法 =====
    def update_training_params(self, **kwargs):
        """
        便捷的训练参数批量更新方法
        
        使用示例:
        config.update_training_params(
            episodes=1000,
            learning_rate=0.001,
            batch_size=128
        )
        """
        for key, value in kwargs.items():
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)
                print(f"✓ 更新训练参数: {key} = {value}")
            else:
                print(f"✗ 警告: 未知的训练参数 '{key}'")
    
    def get_training_summary(self):
        """获取当前训练参数摘要"""
        summary = {
            "基础参数": {
                "episodes": self.training_config.episodes,
                "learning_rate": self.training_config.learning_rate,
                "gamma": self.training_config.gamma,
                "batch_size": self.training_config.batch_size,
                "memory_size": self.training_config.memory_size,
            },
            "探索策略": {
                "epsilon_start": self.training_config.epsilon_start,
                "epsilon_end": self.training_config.epsilon_end,
                "epsilon_decay": self.training_config.epsilon_decay,
                "epsilon_min": self.training_config.epsilon_min,
            },
            "网络更新": {
                "target_update_freq": self.training_config.target_update_freq,
                "patience": self.training_config.patience,
                "log_interval": self.training_config.log_interval,
            },
            "优先经验回放": {
                "use_prioritized_replay": self.training_config.use_prioritized_replay,
                "per_alpha": self.training_config.per_alpha,
                "per_beta_start": self.training_config.per_beta_start,
                "per_beta_frames": self.training_config.per_beta_frames,
            }
        }
        return summary
    
    def print_training_config(self):
        """打印当前训练配置"""
        print("=" * 60)
        print("当前训练配置参数")
        print("=" * 60)
        
        summary = self.get_training_summary()
        for category, params in summary.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        
        # 新增的训练参数
        self.use_gradient_clipping = self.training_config.use_gradient_clipping
        self.max_grad_norm = self.training_config.max_grad_norm
    
    def update_training_config(self, new_config: TrainingConfig):
        """更新训练配置"""
        self.training_config = new_config
        self._setup_backward_compatibility()
    
    def get_training_config(self) -> TrainingConfig:
        """获取当前训练配置"""
        return self.training_config
    
    def load_existing_model(self, model_path: str = None) -> bool:
        """尝试加载已存在的模型"""
        if model_path is None:
            model_path = self.SAVED_MODEL_PATH
        
        if os.path.exists(model_path):
            print(f"发现已存在的模型: {model_path}")
            return True
        return False
    
    # ===== 训练模式便捷方法 =====
    def set_training_mode(self, mode: str):
        """设置训练模式"""
        valid_modes = ['training', 'inference', 'zero_shot_train']
        if mode not in valid_modes:
            raise ValueError(f"无效的训练模式: {mode}。有效模式: {valid_modes}")
        self.TRAINING_MODE = mode
    
    def is_training_mode(self) -> bool:
        """检查是否为训练模式"""
        return self.TRAINING_MODE == 'training'
    
    def is_inference_mode(self) -> bool:
        """检查是否为推理模式"""
        return self.TRAINING_MODE == 'inference'
    
    # ===== PBRS配置管理方法 =====
    def validate_pbrs_config(self) -> bool:
        """
        验证PBRS配置参数的有效性
        
        Returns:
            bool: 配置是否有效
        """
        validation_errors = []
        
        # 验证权重参数
        if self.PBRS_COMPLETION_WEIGHT < 0:
            validation_errors.append("PBRS_COMPLETION_WEIGHT必须为非负数")
        
        if self.PBRS_DISTANCE_WEIGHT < 0:
            validation_errors.append("PBRS_DISTANCE_WEIGHT必须为非负数")
        
        if self.PBRS_COLLABORATION_WEIGHT < 0:
            validation_errors.append("PBRS_COLLABORATION_WEIGHT必须为非负数")
        
        # 验证裁剪参数
        if self.PBRS_REWARD_CLIP_MIN >= self.PBRS_REWARD_CLIP_MAX:
            validation_errors.append("PBRS_REWARD_CLIP_MIN必须小于PBRS_REWARD_CLIP_MAX")
        
        if self.PBRS_POTENTIAL_CLIP_MIN >= self.PBRS_POTENTIAL_CLIP_MAX:
            validation_errors.append("PBRS_POTENTIAL_CLIP_MIN必须小于PBRS_POTENTIAL_CLIP_MAX")
        
        # 验证数值稳定性参数
        if self.PBRS_MAX_POTENTIAL_CHANGE <= 0:
            validation_errors.append("PBRS_MAX_POTENTIAL_CHANGE必须为正数")
        
        if self.PBRS_CACHE_UPDATE_THRESHOLD <= 0 or self.PBRS_CACHE_UPDATE_THRESHOLD >= 1:
            validation_errors.append("PBRS_CACHE_UPDATE_THRESHOLD必须在(0,1)范围内")
        
        # 输出验证结果
        # 静默验证，不输出详细信息
        return len(validation_errors) == 0
    
    def get_pbrs_config_summary(self) -> Dict[str, Any]:
        """获取PBRS配置摘要"""
        return {
            "功能开关": {
                "ENABLE_PBRS": self.ENABLE_PBRS,
                "PBRS_DEBUG_MODE": self.PBRS_DEBUG_MODE,
                "PBRS_LOG_POTENTIAL_VALUES": self.PBRS_LOG_POTENTIAL_VALUES,
                "PBRS_LOG_REWARD_BREAKDOWN": self.PBRS_LOG_REWARD_BREAKDOWN,
            },
            "势函数权重": {
                "PBRS_COMPLETION_WEIGHT": self.PBRS_COMPLETION_WEIGHT,
                "PBRS_DISTANCE_WEIGHT": self.PBRS_DISTANCE_WEIGHT,
                "PBRS_COLLABORATION_WEIGHT": self.PBRS_COLLABORATION_WEIGHT,
            },
            "数值稳定性": {
                "PBRS_REWARD_CLIP_MIN": self.PBRS_REWARD_CLIP_MIN,
                "PBRS_REWARD_CLIP_MAX": self.PBRS_REWARD_CLIP_MAX,
                "PBRS_POTENTIAL_CLIP_MIN": self.PBRS_POTENTIAL_CLIP_MIN,
                "PBRS_POTENTIAL_CLIP_MAX": self.PBRS_POTENTIAL_CLIP_MAX,
                "PBRS_MAX_POTENTIAL_CHANGE": self.PBRS_MAX_POTENTIAL_CHANGE,
            },
            "性能优化": {
                "PBRS_ENABLE_DISTANCE_CACHE": self.PBRS_ENABLE_DISTANCE_CACHE,
                "PBRS_CACHE_UPDATE_THRESHOLD": self.PBRS_CACHE_UPDATE_THRESHOLD,
                "PBRS_ENABLE_GRADIENT_CLIPPING": self.PBRS_ENABLE_GRADIENT_CLIPPING,
            }
        }
    
    def print_pbrs_config(self):
        """打印PBRS配置参数"""
        print("=" * 60)
        print("PBRS (Potential-Based Reward Shaping) 配置参数")
        print("=" * 60)
        
        summary = self.get_pbrs_config_summary()
        for category, params in summary.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
    
    def update_pbrs_params(self, **kwargs):
        """
        便捷的PBRS参数批量更新方法
        
        使用示例:
        config.update_pbrs_params(
            PBRS_COMPLETION_WEIGHT=15.0,
            PBRS_DEBUG_MODE=True,
            ENABLE_PBRS=False
        )
        """
        pbrs_params = [attr for attr in dir(self) if attr.startswith('PBRS_') or attr == 'ENABLE_PBRS']
        
        for key, value in kwargs.items():
            if key in pbrs_params:
                setattr(self, key, value)
                print(f"✓ 更新PBRS参数: {key} = {value}")
            else:
                print(f"✗ 警告: 未知的PBRS参数 '{key}'")
        
        # 更新后重新验证配置
        if not self.validate_pbrs_config():
            print("⚠️  警告: PBRS配置验证失败，请检查参数设置")
    
    def reset_pbrs_to_defaults(self):
        """重置PBRS参数为默认值"""
        self.ENABLE_PBRS = True
        self.PBRS_COMPLETION_WEIGHT = 10.0
        self.PBRS_DISTANCE_WEIGHT = 0.01
        self.PBRS_COLLABORATION_WEIGHT = 5.0
        self.PBRS_REWARD_CLIP_MIN = -30.0
        self.PBRS_REWARD_CLIP_MAX = 30.0
        self.PBRS_DEBUG_MODE = False
        self.PBRS_LOG_POTENTIAL_VALUES = False
        self.PBRS_LOG_REWARD_BREAKDOWN = False
        self.PBRS_POTENTIAL_CLIP_MIN = -1000.0
        self.PBRS_POTENTIAL_CLIP_MAX = 1000.0
        self.PBRS_ENABLE_GRADIENT_CLIPPING = True
        self.PBRS_MAX_POTENTIAL_CHANGE = 100.0
        self.PBRS_ENABLE_DISTANCE_CACHE = True
        self.PBRS_CACHE_UPDATE_THRESHOLD = 0.1
        
        print("✓ PBRS参数已重置为默认值")
        self.validate_pbrs_config()
    
    def is_pbrs_enabled(self) -> bool:
        """检查PBRS功能是否启用"""
        return self.ENABLE_PBRS and self.validate_pbrs_config()
    
    def save_pbrs_config(self, filepath: str = "pbrs_config.pkl"):
        """
        保存PBRS配置到文件
        
        Args:
            filepath: 保存路径，默认为pbrs_config.pkl
        """
        pbrs_config = self.get_pbrs_config_summary()
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(pbrs_config, f)
            print(f"✓ PBRS配置已保存到: {filepath}")
        except Exception as e:
            print(f"✗ 保存PBRS配置失败: {e}")
    
    def load_pbrs_config(self, filepath: str = "pbrs_config.pkl"):
        """
        从文件加载PBRS配置
        
        Args:
            filepath: 配置文件路径
        """
        try:
            with open(filepath, 'rb') as f:
                pbrs_config = pickle.load(f)
            
            # 展平配置字典并更新参数
            flat_config = {}
            for category, params in pbrs_config.items():
                flat_config.update(params)
            
            self.update_pbrs_params(**flat_config)
            print(f"✓ PBRS配置已从 {filepath} 加载")
            
        except FileNotFoundError:
            print(f"✗ 配置文件不存在: {filepath}")
        except Exception as e:
            print(f"✗ 加载PBRS配置失败: {e}")
    
    def load_params_for_scenario(self, uavs, targets): 
        """ 
        根据场景规模（无人机数量）动态调整关键超参数。 
        """ 
        num_uavs = len(uavs) 
        print(f"[Config] 检测到无人机数量: {num_uavs}。正在应用基于规则的参数优化...") 

        # 根据无人机数量应用不同的配置规则 
        if num_uavs < 5: 
            print("[Config] 应用 -> 简单场景配置") 
            self.TOP_K_UAVS = 3 
            self.STAGNATION_THRESHOLD = 20 # 简单场景容忍度更高 
        elif 5 <= num_uavs <= 10: 
            print("[Config] 应用 -> 中等场景配置") 
            self.TOP_K_UAVS = 5 
            self.APPROACH_REWARD_COEFFICIENT = 0.002 
            self.STAGNATION_THRESHOLD = 15 
        else: # num_uavs > 10 
            print("[Config] 应用 -> 复杂场景配置") 
            self.TOP_K_UAVS = 8 
            self.APPROACH_REWARD_COEFFICIENT = 0.005 
            self.STAGNATION_THRESHOLD = 10 # 复杂场景应更快放弃无效轮次 

        # 打印最终应用的动态参数 
        print(f"  [Dynamic Param] TOP_K_UAVS: {self.TOP_K_UAVS}") 
        print(f"  [Dynamic Param] APPROACH_REWARD_COEFFICIENT: {self.APPROACH_REWARD_COEFFICIENT}") 
        print(f"  [Dynamic Param] STAGNATION_THRESHOLD: {self.STAGNATION_THRESHOLD}") 
        print("-" * 30) 
    
    def get_log_filename(self, log_type: str, network_type: str = None, scenario: str = None, timestamp: str = None) -> str:
        """
        生成统一格式的日志文件名
        
        Args:
            log_type: 日志类型 ('action', 'reward', 'training', 'inference')
            network_type: 网络类型，默认使用当前配置
            scenario: 场景名称，默认使用'default'
            timestamp: 时间戳，默认使用当前时间
            
        Returns:
            str: 格式化的文件名
        """
        from datetime import datetime
        
        if not self.UNIFIED_LOG_NAMING:
            # 如果未启用统一命名，返回简单格式
            return f"{log_type}_log.txt"
            
        # 使用默认值
        network_type = network_type or self.NETWORK_TYPE or 'default'
        scenario = scenario or 'default'
        
        if timestamp is None:
            timestamp = datetime.now().strftime(self.LOG_TIMESTAMP_FORMAT)
            
        # 格式化文件名
        filename_base = self.LOG_FILE_NAMING_PATTERN.format(
            timestamp=timestamp,
            network_type=network_type,
            scenario=scenario
        )
        
        return f"{log_type}_log_{filename_base}.txt"
        
    def get_training_curve_title(self, network_type: str = None, scenario: str = None, early_stop_episode: int = None) -> str:
        """
        生成训练曲线标题，包含早停轮数信息
        
        Args:
            network_type: 网络类型
            scenario: 场景名称
            early_stop_episode: 早停轮数
            
        Returns:
            str: 格式化的标题
        """
        if not self.TRAINING_CURVE_INCLUDE_EARLY_STOP_INFO:
            # 如果不包含早停信息，返回简单标题
            network = network_type or self.NETWORK_TYPE or 'Network'
            scene = scenario or 'Default'
            return f"{network} 训练曲线 - {scene}场景"
            
        network_type = network_type or self.NETWORK_TYPE or 'Network'
        scenario = scenario or 'Default'
        early_stop_episode = early_stop_episode or 'N/A'
        
        return self.TRAINING_CURVE_TITLE_FORMAT.format(
            network_type=network_type,
            scenario=scenario,
            early_stop_episode=early_stop_episode
        )
        
    def get_scenario_log_config_summary(self) -> Dict[str, Any]:
        """获取场景记录配置摘要"""
        return {
            "基本配置": {
                "SAVE_SCENARIO_DATA": self.SAVE_SCENARIO_DATA,
                "SCENARIO_DATA_FORMAT": self.SCENARIO_DATA_FORMAT,
                "SCENARIO_DATA_DIR": self.SCENARIO_DATA_DIR,
            },
            "训练场景记录": {
                "SAVE_TRAINING_SCENARIO_DATA": self.SAVE_TRAINING_SCENARIO_DATA,
                "TRAINING_SCENARIO_LOG_FORMAT": self.TRAINING_SCENARIO_LOG_FORMAT,
                "TRAINING_SCENARIO_LOG_DIR": self.TRAINING_SCENARIO_LOG_DIR,
            },
            "推理结果记录": {
                "SAVE_INFERENCE_RESULTS": self.SAVE_INFERENCE_RESULTS,
                "INFERENCE_RESULTS_FORMAT": self.INFERENCE_RESULTS_FORMAT,
                "INFERENCE_RESULTS_DIR": self.INFERENCE_RESULTS_DIR,
                "INFERENCE_RESULTS_INCLUDE_SCENARIO": self.INFERENCE_RESULTS_INCLUDE_SCENARIO,
            },
            "日志命名配置": {
                "UNIFIED_LOG_NAMING": self.UNIFIED_LOG_NAMING,
                "LOG_FILE_NAMING_PATTERN": self.LOG_FILE_NAMING_PATTERN,
                "ACTION_LOG_USE_UNIFIED_NAMING": self.ACTION_LOG_USE_UNIFIED_NAMING,
                "REWARD_LOG_USE_UNIFIED_NAMING": self.REWARD_LOG_USE_UNIFIED_NAMING,
            },
            "训练曲线配置": {
                "TRAINING_CURVE_INCLUDE_EARLY_STOP_INFO": self.TRAINING_CURVE_INCLUDE_EARLY_STOP_INFO,
                "TRAINING_CURVE_TITLE_FORMAT": self.TRAINING_CURVE_TITLE_FORMAT,
                "TRAINING_CURVE_SAVE_FORMAT": self.TRAINING_CURVE_SAVE_FORMAT,
            }
        }
        
    def print_scenario_log_config(self):
        """打印场景记录配置参数"""
        print("=" * 60)
        print("场景记录与日志管理配置参数")
        print("=" * 60)
        
        summary = self.get_scenario_log_config_summary()
        for category, params in summary.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        
    def update_scenario_log_params(self, **kwargs):
        """
        便捷的场景记录参数批量更新方法
        
        使用示例:
        config.update_scenario_log_params(
            SAVE_TRAINING_SCENARIO_DATA=True,
            UNIFIED_LOG_NAMING=True,
            TRAINING_CURVE_INCLUDE_EARLY_STOP_INFO=True
        )
        """
        # 定义场景记录相关的参数名称
        scenario_log_params = [
            'SAVE_SCENARIO_DATA', 'SCENARIO_DATA_FORMAT', 'SCENARIO_DATA_DIR',
            'SAVE_TRAINING_SCENARIO_DATA', 'TRAINING_SCENARIO_LOG_FORMAT', 'TRAINING_SCENARIO_LOG_DIR',
            'SAVE_INFERENCE_RESULTS', 'INFERENCE_RESULTS_FORMAT', 'INFERENCE_RESULTS_DIR', 'INFERENCE_RESULTS_INCLUDE_SCENARIO',
            'SAVE_EPISODE_INFERENCE_RESULTS', 'EPISODE_INFERENCE_LOG_FORMAT', 'EPISODE_INFERENCE_INCLUDE_TASK_ALLOCATION', 
            'EPISODE_INFERENCE_INCLUDE_PERFORMANCE_METRICS', 'EPISODE_INFERENCE_INCLUDE_COMPLETE_VISUALIZATION', 'EPISODE_INFERENCE_LOG_INTERVAL',
            'UNIFIED_LOG_NAMING', 'LOG_FILE_NAMING_PATTERN', 'ACTION_LOG_USE_UNIFIED_NAMING', 'REWARD_LOG_USE_UNIFIED_NAMING',
            'TRAINING_CURVE_INCLUDE_EARLY_STOP_INFO', 'TRAINING_CURVE_TITLE_FORMAT', 'TRAINING_CURVE_SAVE_FORMAT',
            'SAVE_TRAINING_PROGRESS', 'TRAINING_PROGRESS_LOG_INTERVAL', 'TRAINING_PROGRESS_INCLUDE_METRICS'
        ]
        
        for key, value in kwargs.items():
            if key in scenario_log_params:
                setattr(self, key, value)
                print(f"✓ 更新场景记录参数: {key} = {value}")
            else:
                print(f"✗ 警告: 未知的场景记录参数 '{key}'") 
        
    def ensure_log_directories(self):
        """确保所有日志目录存在"""
        directories_to_create = [
            self.SCENARIO_DATA_DIR,
            self.TRAINING_SCENARIO_LOG_DIR,
            self.INFERENCE_RESULTS_DIR,
            'output/logs',  # 主要日志目录
            'output/training_curves'  # 训练曲线目录
        ]
        
        for directory in directories_to_create:
            os.makedirs(directory, exist_ok=True)
            
        print("✓ 所有日志目录已创建")
        
    # 向后兼容的方法
    @property
    def RUN_TRAINING(self) -> bool:
        """向后兼容的RUN_TRAINING属性"""
        return self.is_training_mode()
    
    @RUN_TRAINING.setter
    def RUN_TRAINING(self, value: bool):
        """向后兼容的RUN_TRAINING设置器"""
        self.TRAINING_MODE = 'training' if value else 'inference'
    
    # ===== 自适应课程训练配置管理方法 =====
    def validate_adaptive_curriculum_config(self) -> bool:
        """
        验证自适应课程训练配置参数的有效性
        
        Returns:
            bool: 配置是否有效
        """
        validation_errors = []
        
        # 验证阈值参数范围
        if not (0.0 <= self.CURRICULUM_MASTERY_THRESHOLD <= 1.0):
            validation_errors.append("CURRICULUM_MASTERY_THRESHOLD必须在[0.0, 1.0]范围内")
        
        if not (0.0 <= self.CURRICULUM_REGRESSION_THRESHOLD <= 1.0):
            validation_errors.append("CURRICULUM_REGRESSION_THRESHOLD必须在[0.0, 1.0]范围内")
        
        # 验证窗口大小
        if self.CURRICULUM_PERFORMANCE_WINDOW <= 0:
            validation_errors.append("CURRICULUM_PERFORMANCE_WINDOW必须为正整数")
        
        # 验证轮次参数
        if self.CURRICULUM_MAX_EPISODES_PER_LEVEL <= 0:
            validation_errors.append("CURRICULUM_MAX_EPISODES_PER_LEVEL必须为正整数")
        
        if self.CURRICULUM_MIN_EPISODES_PER_LEVEL <= 0:
            validation_errors.append("CURRICULUM_MIN_EPISODES_PER_LEVEL必须为正整数")
        
        if self.CURRICULUM_MIN_EPISODES_PER_LEVEL >= self.CURRICULUM_MAX_EPISODES_PER_LEVEL:
            validation_errors.append("CURRICULUM_MIN_EPISODES_PER_LEVEL必须小于CURRICULUM_MAX_EPISODES_PER_LEVEL")
        
        # 验证冷却期和降级次数
        if self.CURRICULUM_PROMOTION_COOLDOWN < 0:
            validation_errors.append("CURRICULUM_PROMOTION_COOLDOWN必须为非负整数")
        
        if self.CURRICULUM_MAX_REGRESSION_COUNT < 0:
            validation_errors.append("CURRICULUM_MAX_REGRESSION_COUNT必须为非负整数")
        
        # 验证平滑系数
        if not (0.0 <= self.CURRICULUM_PERFORMANCE_SMOOTHING <= 1.0):
            validation_errors.append("CURRICULUM_PERFORMANCE_SMOOTHING必须在[0.0, 1.0]范围内")
        
        # 输出验证结果
        if validation_errors:
            print("⚠️ 自适应课程训练配置验证失败:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_adaptive_curriculum_config_summary(self) -> Dict[str, Any]:
        """获取自适应课程训练配置摘要"""
        return {
            "核心参数": {
                "CURRICULUM_ENABLE_ADAPTIVE": self.CURRICULUM_ENABLE_ADAPTIVE,
                "CURRICULUM_MASTERY_THRESHOLD": self.CURRICULUM_MASTERY_THRESHOLD,
                "CURRICULUM_PERFORMANCE_WINDOW": self.CURRICULUM_PERFORMANCE_WINDOW,
                "CURRICULUM_MAX_EPISODES_PER_LEVEL": self.CURRICULUM_MAX_EPISODES_PER_LEVEL,
                "CURRICULUM_MIN_EPISODES_PER_LEVEL": self.CURRICULUM_MIN_EPISODES_PER_LEVEL,
            },
            "高级功能": {
                "CURRICULUM_ENABLE_REGRESSION": self.CURRICULUM_ENABLE_REGRESSION,
                "CURRICULUM_REGRESSION_THRESHOLD": self.CURRICULUM_REGRESSION_THRESHOLD,
                "CURRICULUM_PROMOTION_COOLDOWN": self.CURRICULUM_PROMOTION_COOLDOWN,
                "CURRICULUM_MAX_REGRESSION_COUNT": self.CURRICULUM_MAX_REGRESSION_COUNT,
            },
            "监控参数": {
                "CURRICULUM_LOG_DETAILED_PERFORMANCE": self.CURRICULUM_LOG_DETAILED_PERFORMANCE,
                "CURRICULUM_SAVE_LEVEL_CHECKPOINTS": self.CURRICULUM_SAVE_LEVEL_CHECKPOINTS,
                "CURRICULUM_PERFORMANCE_SMOOTHING": self.CURRICULUM_PERFORMANCE_SMOOTHING,
            }
        }
    
    def print_adaptive_curriculum_config(self):
        """打印自适应课程训练配置参数"""
        print("=" * 60)
        print("自适应课程训练配置参数")
        print("=" * 60)
        
        summary = self.get_adaptive_curriculum_config_summary()
        for category, params in summary.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
    
    def update_adaptive_curriculum_params(self, **kwargs):
        """
        便捷的自适应课程训练参数批量更新方法
        
        使用示例:
        config.update_adaptive_curriculum_params(
            CURRICULUM_MASTERY_THRESHOLD=0.85,
            CURRICULUM_ENABLE_REGRESSION=True,
            CURRICULUM_PERFORMANCE_WINDOW=25
        )
        """
        adaptive_params = [attr for attr in dir(self) if attr.startswith('CURRICULUM_')]
        
        for key, value in kwargs.items():
            if key in adaptive_params:
                setattr(self, key, value)
                print(f"✓ 更新自适应课程参数: {key} = {value}")
            else:
                print(f"✗ 警告: 未知的自适应课程参数 '{key}'")
        
        # 更新后重新验证配置
        if not self.validate_adaptive_curriculum_config():
            print("⚠️ 警告: 自适应课程配置验证失败，请检查参数设置")
    
    def reset_adaptive_curriculum_to_defaults(self):
        """重置自适应课程训练参数为默认值"""
        self.CURRICULUM_ENABLE_ADAPTIVE = True
        self.CURRICULUM_MASTERY_THRESHOLD = 0.80
        self.CURRICULUM_PERFORMANCE_WINDOW = 20
        self.CURRICULUM_MAX_EPISODES_PER_LEVEL = 500
        self.CURRICULUM_MIN_EPISODES_PER_LEVEL = 10
        self.CURRICULUM_REGRESSION_THRESHOLD = 0.40
        self.CURRICULUM_ENABLE_REGRESSION = False
        self.CURRICULUM_PROMOTION_COOLDOWN = 5
        self.CURRICULUM_MAX_REGRESSION_COUNT = 2
        self.CURRICULUM_LOG_DETAILED_PERFORMANCE = True
        self.CURRICULUM_SAVE_LEVEL_CHECKPOINTS = True
        self.CURRICULUM_PERFORMANCE_SMOOTHING = 0.1
        
        print("✓ 自适应课程训练参数已重置为默认值")
    
    # 便捷的属性访问接口
    @property
    def is_adaptive_curriculum_enabled(self) -> bool:
        """检查是否启用自适应课程训练"""
        return self.CURRICULUM_ENABLE_ADAPTIVE
    
    @property
    def is_regression_enabled(self) -> bool:
        """检查是否启用降级机制"""
        return self.CURRICULUM_ENABLE_REGRESSION