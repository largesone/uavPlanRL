# TransformerGNN性能调优指南

## 概述
本指南提供TransformerGNN系统的性能优化建议和最佳实践，帮助用户在不同规模场景下获得最佳性能。

## 内存优化策略

### 1. 批次大小调优
根据场景规模调整批次大小以平衡内存使用和训练效率：

- **小规模场景(≤5 UAVs)**: batch_size = 64-128
- **中等规模场景(6-10 UAVs)**: batch_size = 32-64  
- **大规模场景(>10 UAVs)**: batch_size = 16-32

```python
# 动态批次大小配置示例
def get_optimal_batch_size(n_uavs, n_targets):
    total_entities = n_uavs + n_targets
    if total_entities <= 8:
        return 128
    elif total_entities <= 16:
        return 64
    elif total_entities <= 24:
        return 32
    else:
        return 16
```

### 2. 局部注意力k值优化
k值直接影响计算复杂度和内存占用：

- **自适应k值公式**: `k = min(max(4, ceil(N/4)), 16)`
- **训练期随机化**: `k ± 2` 增强鲁棒性
- **推理期固定k值**确保可复现性

```python
# k值自适应计算
def compute_adaptive_k(n_uavs, training=True):
    base_k = min(max(4, math.ceil(n_uavs / 4)), 16)
    if training:
        # 训练期间添加随机化
        noise = random.randint(-2, 2)
        return max(1, base_k + noise)
    return base_k
```

### 3. 梯度检查点
在大规模场景下启用梯度检查点减少内存占用：

```python
# 启用梯度检查点
import torch.utils.checkpoint as checkpoint

class TransformerGNN(nn.Module):
    def forward(self, x):
        # 使用检查点包装计算密集的层
        x = checkpoint.checkpoint(self.attention_layer, x)
        return x
```

### 4. 内存池管理
优化GPU内存分配策略：

```python
# 设置内存池
torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
torch.cuda.empty_cache()  # 定期清理缓存
```

## 训练速度优化

### 1. 分布式训练配置
针对不同硬件配置的推荐设置：

```python
# 单机多GPU配置
config = {
    "num_rollout_workers": 4,
    "num_envs_per_worker": 2,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
}

# 多机分布式配置
config = {
    "num_rollout_workers": 8,
    "num_envs_per_worker": 1,
    "train_batch_size": 8000,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 5,
}
```

### 2. GPU优化设置
启用现代GPU优化特性：

```python
# 混合精度训练
config["mixed_precision"] = True

# 编译优化（PyTorch 2.0+）
config["torch_compile"] = True

# 优化数据加载
config["num_data_loader_workers"] = 4
config["pin_memory"] = True
```

### 3. 网络架构优化
针对不同场景规模的架构调整：

```python
# 小规模场景配置
small_config = {
    "hidden_dim": 128,
    "num_attention_heads": 4,
    "num_layers": 2,
}

# 大规模场景配置
large_config = {
    "hidden_dim": 256,
    "num_attention_heads": 8,
    "num_layers": 4,
}
```

## 课程学习调优

### 1. 阶段推进策略
根据训练稳定性选择推进策略：

```python
# 保守推进策略（推荐用于复杂场景）
conservative_config = {
    "advance_threshold": 0.8,  # 性能达到80%才推进
    "consecutive_evaluations": 5,  # 连续5次评估
    "patience": 10,  # 等待10个评估周期
}

# 激进推进策略（适用于简单场景）
aggressive_config = {
    "advance_threshold": 0.7,
    "consecutive_evaluations": 3,
    "patience": 5,
}
```

### 2. 回退门限设置
防止训练发散的回退机制：

```python
# 回退配置
fallback_config = {
    "performance_threshold": 0.6,  # 性能低于上阶段60%时回退
    "consecutive_failures": 3,     # 连续3次失败
    "max_fallbacks": 2,           # 最大回退次数
    "learning_rate_decay": 0.5,   # 回退时学习率衰减
}
```

### 3. 混合经验回放比例
不同阶段的经验混合策略：

```python
# 经验回放比例配置
replay_ratios = {
    1: {"current": 1.0, "historical": 0.0},      # 第1阶段：纯当前经验
    2: {"current": 0.7, "historical": 0.3},      # 第2阶段：70%当前 + 30%历史
    3: {"current": 0.6, "historical": 0.4},      # 第3阶段：60%当前 + 40%历史
    4: {"current": 0.5, "historical": 0.5},      # 第4阶段：50%当前 + 50%历史
}
```

## 监控和调试

### 1. 关键监控指标
实时监控以下指标确保训练健康：

- **Per-Agent Reward**: 单智能体平均奖励
- **Normalized Completion Score**: 归一化完成分数
- **Efficiency Metric**: 效率指标
- **内存使用率**: GPU/CPU内存占用
- **GPU利用率**: 计算资源利用情况

```python
# 监控指标记录
def log_training_metrics(metrics, step):
    logger.log_scalar("per_agent_reward", metrics["reward"] / metrics["n_agents"], step)
    logger.log_scalar("completion_score", metrics["completion_rate"] * (1 - metrics["congestion"]), step)
    logger.log_scalar("efficiency", metrics["completed_targets"] / metrics["total_distance"], step)
    logger.log_scalar("memory_usage", torch.cuda.memory_allocated() / 1024**3, step)
```

### 2. 性能瓶颈识别
使用性能分析工具定位瓶颈：

```python
# 使用PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(training_steps):
        train_step()
        prof.step()
```

### 3. 自动化性能调优
实现自动化的超参数调优：

```python
# 自动批次大小调优
def auto_tune_batch_size(model, env, max_memory_gb=8):
    batch_size = 32
    while batch_size <= 256:
        try:
            # 测试当前批次大小
            test_batch(model, env, batch_size)
            memory_used = torch.cuda.memory_allocated() / 1024**3
            
            if memory_used > max_memory_gb * 0.9:  # 90%内存使用率
                return batch_size // 2
            
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                return batch_size // 2
            raise e
    
    return batch_size
```

## 故障排除

### 常见问题和解决方案

#### 1. 内存不足(OOM)
**症状**: CUDA out of memory错误
**解决方案**:
- 减少批次大小
- 启用梯度检查点
- 降低k值上限
- 使用CPU offloading

```python
# OOM恢复策略
def handle_oom_error():
    torch.cuda.empty_cache()
    # 减少批次大小
    config["train_batch_size"] //= 2
    config["sgd_minibatch_size"] //= 2
    print(f"OOM detected, reducing batch size to {config['train_batch_size']}")
```

#### 2. 训练发散
**症状**: 奖励急剧下降，损失爆炸
**解决方案**:
- 降低学习率
- 增加梯度裁剪
- 检查奖励归一化
- 启用回退机制

```python
# 训练稳定性检查
def check_training_stability(rewards, threshold=0.1):
    if len(rewards) < 10:
        return True
    
    recent_std = np.std(rewards[-10:])
    overall_std = np.std(rewards)
    
    if recent_std > overall_std * 2:
        print("Training instability detected!")
        return False
    return True
```

#### 3. 收敛缓慢
**症状**: 长时间无性能提升
**解决方案**:
- 调整学习率调度
- 增加探索噪声
- 检查课程难度设置
- 优化网络架构

```python
# 自适应学习率调整
def adaptive_lr_schedule(optimizer, performance_history, patience=5):
    if len(performance_history) < patience:
        return
    
    recent_performance = performance_history[-patience:]
    if max(recent_performance) - min(recent_performance) < 0.01:
        # 性能停滞，降低学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8
        print(f"Learning rate reduced to {param_group['lr']}")
```

#### 4. 零样本迁移失效
**症状**: 大规模场景性能急剧下降
**解决方案**:
- 检查状态归一化
- 增加训练场景多样性
- 调整k值自适应策略
- 强化位置编码

```python
# 零样本迁移验证
def validate_zero_shot_transfer(model, small_env, large_env):
    small_performance = evaluate_model(model, small_env)
    large_performance = evaluate_model(model, large_env)
    
    transfer_ratio = large_performance / small_performance
    if transfer_ratio < 0.5:  # 性能下降超过50%
        print("Zero-shot transfer failed!")
        return False
    return True
```

## 最佳实践总结

### 1. 渐进式调优策略
- 从小规模场景开始调优
- 逐步增加复杂度
- 记录每次调优的效果
- 建立性能基线

### 2. 充分的性能监控
- 实时监控关键指标
- 设置性能告警阈值
- 定期生成性能报告
- 保存训练检查点

### 3. 定期性能基准测试
- 建立标准测试集
- 对比不同配置性能
- 记录硬件配置信息
- 追踪性能变化趋势

### 4. 文档化调优过程
- 记录调优参数变化
- 保存最佳配置
- 分享调优经验
- 建立知识库

## 配置模板

### 开发环境配置
```python
dev_config = {
    "train_batch_size": 1000,
    "sgd_minibatch_size": 64,
    "num_rollout_workers": 2,
    "num_envs_per_worker": 1,
    "evaluation_interval": 10,
    "checkpoint_freq": 50,
}
```

### 生产环境配置
```python
prod_config = {
    "train_batch_size": 8000,
    "sgd_minibatch_size": 256,
    "num_rollout_workers": 8,
    "num_envs_per_worker": 2,
    "evaluation_interval": 100,
    "checkpoint_freq": 500,
    "mixed_precision": True,
    "torch_compile": True,
}
```

### 调试配置
```python
debug_config = {
    "train_batch_size": 100,
    "sgd_minibatch_size": 32,
    "num_rollout_workers": 1,
    "num_envs_per_worker": 1,
    "evaluation_interval": 1,
    "checkpoint_freq": 10,
    "log_level": "DEBUG",
}
```

通过遵循这些指南和最佳实践，您可以在不同场景下获得TransformerGNN系统的最佳性能。记住，性能调优是一个迭代过程，需要根据具体的硬件配置和应用场景进行调整。