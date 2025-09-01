# 需求文档

## 介绍

本项目旨在构建一个支持零样本迁移的、具备高级稳定与工程化机制的局部注意力Transformer网络，并实施课程学习训练。该系统将解决现有RL集群任务分配方法在无人机数量增加、目标数量变化等情况下泛化能力不足的问题，实现强化学习智能体在任意数量无人机/目标（N/M）场景下的鲁棒零样本迁移。

核心改进思想是围绕全新的`TransformerGNN`架构，引入一系列高级机制来解决维度爆炸、排列不变性、尺度漂移、灾难性遗忘、分布式数据一致性等关键挑战，并通过精心设计的课程学习范式系统性地训练这个强大的模型。

## 需求

### 需求 1：环境层升级 - 双模式图观测支持

**用户故事：** 作为一个RL研究者，我希望环境能够支持传统扁平观测和新的图结构观测两种模式，以便在保持向后兼容性的同时支持新的TransformerGNN架构。

#### 验收标准

1. WHEN 初始化UAVTaskEnv时 THEN 系统SHALL支持obs_mode参数选择"flat"或"graph"模式
2. WHEN obs_mode为"flat"时 THEN 系统SHALL维持现有扁平向量观测空间以确保FCN向后兼容性
3. WHEN obs_mode为"graph"时 THEN 系统SHALL定义gym.spaces.Dict观测空间以支持可变数量实体
4. WHEN 使用图模式时 THEN 系统SHALL输出包含uav_features、target_features、relative_positions、distances、masks的归一化状态字典

### 需求 2：尺度不变的状态表示

**用户故事：** 作为一个系统设计者，我希望状态表示能够解决尺度漂移问题，使模型能够在不同规模的场景间进行零样本迁移。

#### 验收标准

1. WHEN 生成图模式状态时 THEN 系统SHALL移除绝对坐标，仅包含实体自身属性如资源比例res/max_res
2. WHEN 计算relative_positions时 THEN 系统SHALL存储归一化相对位置向量(pos_j - pos_i) / MAP_SIZE
3. WHEN 计算distances时 THEN 系统SHALL存储无人机与目标间的归一化距离
4. WHEN 生成masks时 THEN 系统SHALL包含uav_mask和target_mask用于标识有效实体

### 需求 3：鲁棒性输入掩码机制

**用户故事：** 作为一个系统工程师，我希望系统能够处理通信/感知失效问题，在部分可观测情况下仍能稳健运行。

#### 验收标准

1. WHEN 生成uav_features时 THEN 系统SHALL增加is_alive位（0/1）标识无人机状态
2. WHEN 生成target_features时 THEN 系统SHALL增加is_visible位（0/1）标识目标可见性
3. WHEN TransformerGNN前向传播时 THEN 系统SHALL利用掩码位结合masks屏蔽失效节点
4. WHEN 存在失效节点时 THEN 系统SHALL确保在部分可观测情况下稳健输出

### 需求 4：Per-Agent奖励归一化

**用户故事：** 作为一个算法设计者，我希望奖励函数能够解决奖励归一化不一致问题，使奖励不随无人机数量变化而产生偏差。

#### 验收标准

1. WHEN 计算拥堵惩罚等与无人机数量相关的奖励项时 THEN 系统SHALL除以当前有效无人机数量N_active
2. WHEN 无人机数量变化时 THEN 系统SHALL确保单个智能体的平均奖励保持一致
3. WHEN 评估不同规模场景时 THEN 系统SHALL提供可比较的奖励指标
4. WHEN 训练过程中 THEN 系统SHALL记录归一化后的奖励用于监控

### 需求 5：TransformerGNN网络架构

**用户故事：** 作为一个深度学习工程师，我希望构建一个高效、鲁棒的TransformerGNN网络，能够处理可变数量的实体并支持零样本迁移。

#### 验收标准

1. WHEN 网络前向传播时 THEN 系统SHALL引入相对位置编码解决排列不变性问题
2. WHEN 处理大规模场景时 THEN 系统SHALL实现k-近邻局部注意力机制避免维度爆炸
3. WHEN 确定k值时 THEN 系统SHALL基于当前场景有效无人机数量N动态调整k值
4. WHEN 训练期间 THEN 系统SHALL对k值进行随机抖动增强模型鲁棒性

### 需求 6：参数空间噪声探索

**用户故事：** 作为一个强化学习研究者，我希望网络能够使用参数空间噪声进行探索，解决探索策略耦合问题。

#### 验收标准

1. WHEN 构建TransformerGNN时 THEN 系统SHALL将所有nn.Linear层替换为NoisyLinear层
2. WHEN 网络处于训练模式时 THEN 系统SHALL启用参数噪声进行探索
3. WHEN 网络处于eval模式时 THEN 系统SHALL关闭NoisyLinear层噪声确保推理可复现性
4. WHEN 继承Ray RLlib的TorchModelV2时 THEN 系统SHALL正确实现自定义模型接口

### 需求 7：课程学习训练策略

**用户故事：** 作为一个训练工程师，我希望实施课程学习范式，让智能体从简单场景逐步过渡到复杂场景，并具备回退机制防止灾难性发散。

#### 验收标准

1. WHEN 开始训练时 THEN 系统SHALL创建run_curriculum_training.py作为最高级别训练协调器
2. WHEN 定义课程时 THEN 系统SHALL设计从少实体场景到多实体场景的渐进式训练阶段
3. WHEN 性能下降时 THEN 系统SHALL实现回退门限机制，连续3个评估周期性能低于上阶段60%时自动回退
4. WHEN 进入第二阶段及以后时 THEN 系统SHALL实现混合经验回放，包含70%当前阶段和30%旧阶段经验

### 需求 8：分布式训练数据一致性

**用户故事：** 作为一个分布式系统工程师，我希望解决GNN稀疏张量跨进程错误问题，确保分布式训练的数据一致性。

#### 验收标准

1. WHEN 使用Ray RLlib分布式训练时 THEN RolloutWorker中的图数据张量SHALL在发送前调用.cpu().share_memory_()
2. WHEN 在Learner中处理数据时 THEN 系统SHALL配置数据加载器使用pin_memory=True
3. WHEN num_rollout_workers > 0时 THEN 系统SHALL确保稀疏张量正确跨进程传输
4. WHEN 分布式训练时 THEN 系统SHALL维持数据一致性和训练稳定性

### 需求 9：尺度不变评价指标

**用户故事：** 作为一个评估专家，我希望引入尺度不变的评价指标，解决评价指标误导问题，提供更准确的性能评估。

#### 验收标准

1. WHEN 记录训练指标时 THEN 系统SHALL计算Per-Agent Reward = total_reward / N_active
2. WHEN 评估完成情况时 THEN 系统SHALL计算Normalized Completion Score = satisfied_targets_rate * (1 - average_congestion_metric)
3. WHEN 衡量效率时 THEN 系统SHALL计算Efficiency Metric = total_completed_targets / total_flight_distance
4. WHEN 生成对比图表时 THEN 系统SHALL以尺度不变指标作为主要Y轴显示

### 需求 10：Ray RLlib集成优先级

**用户故事：** 作为一个系统架构师，我希望优先使用Ray RLlib库中经过验证的功能，仅在必要时自编代码，确保系统的可靠性和可维护性。

#### 验收标准

1. WHEN 实现经验回放时 THEN 系统SHALL优先使用Ray RLlib的Replay Buffer API
2. WHEN 实现算法逻辑时 THEN 系统SHALL优先使用Ray RLlib的算法组件
3. WHEN 需要自定义功能时 THEN 系统SHALL仅在RLlib无法满足需求时才自编代码
4. WHEN 编写自定义代码时 THEN 系统SHALL包含完整清晰的注释并保持向后兼容性