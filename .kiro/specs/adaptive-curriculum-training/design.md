# 自适应课程训练设计文档

## 概述

本设计文档描述了如何在现有的课程训练系统基础上实现基于智能体表现的自适应晋级机制。该机制将替代当前固定轮次的线性训练流程，根据智能体的实际学习表现动态决定晋级时机，确保每个难度等级都得到充分掌握。

## 架构

### 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                    自适应课程训练系统                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   配置管理器     │  │   性能监控器     │  │   晋级判断器     │ │
│  │ ConfigManager   │  │ PerformanceMonitor│ │ PromotionDecider│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                     │                     │        │
│           └─────────────────────┼─────────────────────┘        │
│                                 │                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              自适应训练循环控制器                            │ │
│  │            AdaptiveTrainingController                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                 │                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                现有训练基础设施                              │ │
│  │         (ModelTrainer, GraphRLSolver, etc.)               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

```
训练开始 → 初始化性能监控 → 执行训练轮次 → 记录表现数据 → 
评估掌握度 → 判断晋级条件 → [晋级/继续/超时处理] → 下一轮次/下一等级
```

## 组件设计

### 1. 配置管理器 (Config扩展)

**职责：** 管理自适应课程训练的所有配置参数

**新增配置参数：**
```python
# 自适应课程训练配置
CURRICULUM_MASTERY_THRESHOLD: float = 0.80      # 掌握度阈值
CURRICULUM_PERFORMANCE_WINDOW: int = 20         # 性能评估窗口大小  
CURRICULUM_MAX_EPISODES_PER_LEVEL: int = 500    # 单级最大训练轮次
CURRICULUM_REGRESSION_THRESHOLD: float = 0.40   # 退步阈值
CURRICULUM_ENABLE_ADAPTIVE: bool = True         # 启用自适应机制
CURRICULUM_ENABLE_REGRESSION: bool = False      # 启用降级机制
CURRICULUM_MIN_EPISODES_PER_LEVEL: int = 10     # 单级最小训练轮次
CURRICULUM_PROMOTION_COOLDOWN: int = 5          # 晋级冷却期
```

**验证机制：**
- 阈值参数范围检查 (0.0 ≤ threshold ≤ 1.0)
- 窗口大小正整数检查
- 最大轮次合理性检查

### 2. 性能监控器 (PerformanceMonitor)

**职责：** 跟踪和分析智能体在各个难度等级的表现

**核心数据结构：**
```python
@dataclass
class LevelPerformance:
    level_index: int
    level_name: str
    completion_rates: deque  # 滑动窗口
    episode_count: int
    total_episodes: int
    start_time: float
    best_completion_rate: float
    average_completion_rate: float
    is_mastered: bool
    promotion_episode: Optional[int]
```

**关键方法：**
- `record_episode_performance(completion_rate: float)`: 记录单轮表现
- `calculate_mastery_score() -> float`: 计算掌握度分数
- `is_level_mastered() -> bool`: 判断是否掌握当前等级
- `should_promote() -> bool`: 判断是否应该晋级
- `get_performance_summary() -> Dict`: 获取性能摘要

### 3. 晋级判断器 (PromotionDecider)

**职责：** 基于性能数据做出晋级、继续或降级决策

**决策逻辑：**
```python
def make_decision(performance: LevelPerformance, config: Config) -> Decision:
    # 1. 检查最小训练轮次要求
    if performance.episode_count < config.CURRICULUM_MIN_EPISODES_PER_LEVEL:
        return Decision.CONTINUE
    
    # 2. 检查掌握度
    if performance.is_level_mastered():
        return Decision.PROMOTE
    
    # 3. 检查超时
    if performance.episode_count >= config.CURRICULUM_MAX_EPISODES_PER_LEVEL:
        return Decision.TIMEOUT_PROMOTE  # 或 TIMEOUT_CONTINUE
    
    # 4. 检查是否需要降级（可选）
    if config.CURRICULUM_ENABLE_REGRESSION and should_regress(performance):
        return Decision.REGRESS
    
    return Decision.CONTINUE
```

**决策类型：**
```python
class Decision(Enum):
    CONTINUE = "continue"           # 继续当前等级
    PROMOTE = "promote"             # 晋级到下一等级
    REGRESS = "regress"             # 降级到上一等级
    TIMEOUT_PROMOTE = "timeout_promote"  # 超时强制晋级
    TIMEOUT_CONTINUE = "timeout_continue" # 超时继续训练
```

### 4. 自适应训练循环控制器 (AdaptiveTrainingController)

**职责：** 协调各组件，控制整个自适应训练流程

**核心算法：**
```python
def run_adaptive_curriculum_training(self):
    current_level_index = 0
    curriculum_scenarios = get_curriculum_scenarios()
    
    while current_level_index < len(curriculum_scenarios):
        # 初始化当前等级
        level_performance = self._initialize_level(current_level_index)
        
        # 内部训练循环
        while True:
            # 执行单轮训练
            completion_rate = self._train_single_episode(
                curriculum_scenarios[current_level_index]
            )
            
            # 记录表现
            level_performance.record_episode_performance(completion_rate)
            
            # 做出决策
            decision = self.promotion_decider.make_decision(
                level_performance, self.config
            )
            
            # 处理决策
            if decision == Decision.PROMOTE:
                self._handle_promotion(level_performance)
                current_level_index += 1
                break
            elif decision == Decision.REGRESS:
                self._handle_regression(level_performance)
                current_level_index = max(0, current_level_index - 1)
                break
            elif decision in [Decision.TIMEOUT_PROMOTE, Decision.TIMEOUT_CONTINUE]:
                self._handle_timeout(decision, level_performance)
                if decision == Decision.TIMEOUT_PROMOTE:
                    current_level_index += 1
                break
            # Decision.CONTINUE: 继续当前等级训练
```

## 数据模型

### 训练状态数据
```python
@dataclass
class AdaptiveCurriculumState:
    current_level_index: int
    total_levels: int
    level_performances: List[LevelPerformance]
    global_episode_counter: int
    training_start_time: float
    is_adaptive_enabled: bool
    regression_count: int
    total_promotions: int
```

### 性能指标数据
```python
@dataclass
class PerformanceMetrics:
    completion_rate: float
    episode_reward: float
    step_count: int
    training_time: float
    memory_usage: Optional[float]
    convergence_indicator: float
```

## 接口设计

### 1. 配置接口
```python
class AdaptiveCurriculumConfig:
    def validate_config(self) -> bool
    def get_config_summary(self) -> Dict
    def update_adaptive_params(self, **kwargs)
    def reset_to_defaults(self)
```

### 2. 监控接口
```python
class IPerformanceMonitor:
    def record_performance(self, metrics: PerformanceMetrics)
    def get_current_mastery_score(self) -> float
    def is_ready_for_promotion(self) -> bool
    def get_level_summary(self) -> Dict
```

### 3. 决策接口
```python
class IPromotionDecider:
    def evaluate_promotion_readiness(self, performance: LevelPerformance) -> bool
    def make_promotion_decision(self) -> Decision
    def get_decision_rationale(self) -> str
```

## 错误处理

### 异常类型
```python
class AdaptiveCurriculumError(Exception):
    """自适应课程训练基础异常"""
    pass

class InvalidConfigurationError(AdaptiveCurriculumError):
    """无效配置异常"""
    pass

class PerformanceMonitoringError(AdaptiveCurriculumError):
    """性能监控异常"""
    pass

class PromotionDecisionError(AdaptiveCurriculumError):
    """晋级决策异常"""
    pass
```

### 错误恢复策略
1. **配置错误：** 自动回退到默认配置并记录警告
2. **性能监控错误：** 使用备用指标或跳过当前轮次
3. **决策错误：** 默认继续当前等级训练
4. **系统错误：** 保存当前状态并优雅退出

## 测试策略

### 单元测试
- 配置参数验证测试
- 性能监控器功能测试
- 晋级判断逻辑测试
- 边界条件测试

### 集成测试
- 完整自适应训练流程测试
- 与现有训练系统兼容性测试
- 多等级晋级场景测试
- 降级机制测试

### 性能测试
- 大规模场景下的性能表现
- 内存使用效率测试
- 训练时间对比测试

### 回归测试
- 确保不影响现有动态随机训练
- 确保原有API兼容性
- 确保模型加载/保存功能正常

## 实现优先级

### 第一阶段：核心功能
1. 配置参数扩展
2. 性能监控器实现
3. 基础晋级判断逻辑
4. 训练循环重构

### 第二阶段：增强功能
1. 详细日志记录
2. 性能可视化
3. 配置验证和错误处理
4. 单元测试覆盖

### 第三阶段：高级功能
1. 降级机制实现
2. 高级决策算法
3. 性能优化
4. 完整测试套件

## 兼容性考虑

### 向后兼容
- 保持现有训练接口不变
- 通过配置开关控制新功能
- 默认禁用自适应机制，需要显式启用

### 配置迁移
- 提供配置迁移工具
- 自动检测和转换旧配置格式
- 提供配置验证和建议

### API稳定性
- 保持现有公共API不变
- 新增功能通过扩展接口提供
- 废弃功能提供过渡期和警告