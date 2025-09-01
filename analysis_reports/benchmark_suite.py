"""
性能基准测试套件 - 对比TransformerGNN与现有方法的性能
包括训练速度、推理延迟、内存占用等关键指标
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    method_name: str
    scenario_size: Tuple[int, int]  # (n_uavs, n_targets)
    training_time_per_episode: float
    inference_time_ms: float
    memory_usage_mb: float
    convergence_episodes: int
    final_performance: float

class BenchmarkSuite:
    """性能基准测试套件"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    def benchmark_training_speed(self, model, env, episodes: int = 100) -> float:
        """基准测试训练速度"""
        start_time = time.time()
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 200:  # 最大步数限制
                if hasattr(model, 'compute_single_action'):
                    # Ray RLlib模型
                    action = model.compute_single_action(state)
                else:
                    # 自定义模型
                    with torch.no_grad():
                        if isinstance(state, dict):
                            # 图模式状态
                            action = model.forward({"obs": state}, [], [])
                        else:
                            # 扁平状态
                            action = model(torch.FloatTensor(state).unsqueeze(0))
                        action = action.cpu().numpy()
                
                state, reward, done, info = env.step(action)
                step_count += 1
        
        total_time = time.time() - start_time
        avg_time_per_episode = total_time / episodes
        
        return avg_time_per_episode
        
    def benchmark_inference_latency(self, model, test_states: List, runs: int = 1000) -> float:
        """基准测试推理延迟"""
        if not test_states:
            return 0.0
            
        # 预热
        for _ in range(10):
            state = test_states[0]
            if hasattr(model, 'compute_single_action'):
                model.compute_single_action(state)
            else:
                with torch.no_grad():
                    if isinstance(state, dict):
                        model.forward({"obs": state}, [], [])
                    else:
                        model(torch.FloatTensor(state).unsqueeze(0))
        
        # 实际测试
        start_time = time.time()
        
        for i in range(runs):
            state = test_states[i % len(test_states)]
            
            if hasattr(model, 'compute_single_action'):
                model.compute_single_action(state)
            else:
                with torch.no_grad():
                    if isinstance(state, dict):
                        model.forward({"obs": state}, [], [])
                    else:
                        model(torch.FloatTensor(state).unsqueeze(0))
        
        total_time = time.time() - start_time
        avg_latency_ms = (total_time / runs) * 1000
        
        return avg_latency_ms
        
    def benchmark_memory_usage(self, model, scenario_sizes: List[Tuple[int, int]]) -> Dict:
        """基准测试内存使用"""
        memory_results = {}
        
        for n_uavs, n_targets in scenario_sizes:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                initial_memory = 0
            
            # 创建测试状态
            if hasattr(model, 'observation_space'):
                # 检查是否为图模式
                if hasattr(model.observation_space, 'spaces'):
                    # 图模式状态
                    test_state = {
                        'uav_features': np.random.randn(n_uavs, 8),
                        'target_features': np.random.randn(n_targets, 4),
                        'relative_positions': np.random.randn(n_uavs, n_targets, 2),
                        'distances': np.random.randn(n_uavs, n_targets),
                        'masks': {
                            'uav_mask': np.ones(n_uavs),
                            'target_mask': np.ones(n_targets)
                        }
                    }
                else:
                    # 扁平模式状态
                    state_dim = model.observation_space.shape[0]
                    test_state = np.random.randn(state_dim)
            else:
                # 默认扁平状态
                test_state = np.random.randn(100)  # 假设状态维度
            
            # 执行前向传播
            try:
                if hasattr(model, 'compute_single_action'):
                    model.compute_single_action(test_state)
                else:
                    with torch.no_grad():
                        if isinstance(test_state, dict):
                            model.forward({"obs": test_state}, [], [])
                        else:
                            model(torch.FloatTensor(test_state).unsqueeze(0))
                
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                    memory_used = peak_memory - initial_memory
                else:
                    memory_used = 0
                    
                memory_results[f"{n_uavs}_{n_targets}"] = memory_used
                
            except Exception as e:
                print(f"内存测试失败 ({n_uavs}, {n_targets}): {e}")
                memory_results[f"{n_uavs}_{n_targets}"] = -1
        
        return memory_results
        
    def compare_methods(self, methods: Dict[str, Any], scenarios: List[Tuple[int, int]]):
        """对比不同方法的性能"""
        self.results.clear()
        
        for method_name, model in methods.items():
            print(f"测试方法: {method_name}")
            
            for n_uavs, n_targets in scenarios:
                print(f"  场景规模: {n_uavs} UAVs, {n_targets} 目标")
                
                try:
                    # 创建测试环境（简化版）
                    from environment import UAVTaskEnv
                    env = UAVTaskEnv(
                        n_uavs=n_uavs,
                        n_targets=n_targets,
                        obs_mode="graph" if "transformer" in method_name.lower() else "flat"
                    )
                    
                    # 训练速度测试
                    training_time = self.benchmark_training_speed(model, env, episodes=10)
                    
                    # 推理延迟测试
                    test_states = [env.reset() for _ in range(5)]
                    inference_time = self.benchmark_inference_latency(model, test_states, runs=100)
                    
                    # 内存使用测试
                    memory_usage = self.benchmark_memory_usage(model, [(n_uavs, n_targets)])
                    memory_mb = memory_usage.get(f"{n_uavs}_{n_targets}", 0)
                    
                    result = BenchmarkResult(
                        method_name=method_name,
                        scenario_size=(n_uavs, n_targets),
                        training_time_per_episode=training_time,
                        inference_time_ms=inference_time,
                        memory_usage_mb=memory_mb,
                        convergence_episodes=0,  # 需要长期训练才能获得
                        final_performance=0.0    # 需要长期训练才能获得
                    )
                    
                    self.results.append(result)
                    
                except Exception as e:
                    print(f"    测试失败: {e}")
                    continue
        
    def generate_performance_report(self) -> str:
        """生成性能对比报告"""
        if not self.results:
            return "无测试结果"
        
        report = ["# 性能基准测试报告\n"]
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"总测试结果数: {len(self.results)}\n")
        
        # 按方法分组
        methods = {}
        for result in self.results:
            if result.method_name not in methods:
                methods[result.method_name] = []
            methods[result.method_name].append(result)
        
        for method_name, results in methods.items():
            report.append(f"\n## {method_name}\n")
            report.append("| 场景规模 | 训练时间(s/episode) | 推理延迟(ms) | 内存使用(MB) |")
            report.append("|----------|-------------------|-------------|-------------|")
            
            for result in results:
                scenario = f"{result.scenario_size[0]}×{result.scenario_size[1]}"
                report.append(f"| {scenario} | {result.training_time_per_episode:.3f} | {result.inference_time_ms:.2f} | {result.memory_usage_mb:.1f} |")
        
        # 性能对比总结
        report.append("\n## 性能对比总结\n")
        
        # 找出最快的方法
        avg_training_times = {}
        avg_inference_times = {}
        avg_memory_usage = {}
        
        for method_name, results in methods.items():
            avg_training_times[method_name] = np.mean([r.training_time_per_episode for r in results])
            avg_inference_times[method_name] = np.mean([r.inference_time_ms for r in results])
            avg_memory_usage[method_name] = np.mean([r.memory_usage_mb for r in results if r.memory_usage_mb > 0])
        
        fastest_training = min(avg_training_times.items(), key=lambda x: x[1])
        fastest_inference = min(avg_inference_times.items(), key=lambda x: x[1])
        lowest_memory = min(avg_memory_usage.items(), key=lambda x: x[1]) if avg_memory_usage else ("N/A", 0)
        
        report.append(f"- **训练速度最快**: {fastest_training[0]} ({fastest_training[1]:.3f}s/episode)")
        report.append(f"- **推理延迟最低**: {fastest_inference[0]} ({fastest_inference[1]:.2f}ms)")
        report.append(f"- **内存使用最少**: {lowest_memory[0]} ({lowest_memory[1]:.1f}MB)")
        
        return "\n".join(report)
        
    def plot_performance_comparison(self):
        """绘制性能对比图表"""
        if not self.results:
            print("无测试结果可绘制")
            return
        
        # 准备数据
        methods = list(set(r.method_name for r in self.results))
        scenarios = list(set(r.scenario_size for r in self.results))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('性能基准测试对比', fontsize=16)
        
        # 训练时间对比
        ax1 = axes[0, 0]
        training_data = {}
        for method in methods:
            training_data[method] = []
            for scenario in scenarios:
                result = next((r for r in self.results 
                             if r.method_name == method and r.scenario_size == scenario), None)
                training_data[method].append(result.training_time_per_episode if result else 0)
        
        x = np.arange(len(scenarios))
        width = 0.35
        for i, (method, times) in enumerate(training_data.items()):
            ax1.bar(x + i * width, times, width, label=method)
        
        ax1.set_xlabel('场景规模')
        ax1.set_ylabel('训练时间 (s/episode)')
        ax1.set_title('训练速度对比')
        ax1.set_xticks(x + width / 2)
        ax1.set_xticklabels([f"{s[0]}×{s[1]}" for s in scenarios])
        ax1.legend()
        
        # 推理延迟对比
        ax2 = axes[0, 1]
        inference_data = {}
        for method in methods:
            inference_data[method] = []
            for scenario in scenarios:
                result = next((r for r in self.results 
                             if r.method_name == method and r.scenario_size == scenario), None)
                inference_data[method].append(result.inference_time_ms if result else 0)
        
        for i, (method, times) in enumerate(inference_data.items()):
            ax2.bar(x + i * width, times, width, label=method)
        
        ax2.set_xlabel('场景规模')
        ax2.set_ylabel('推理延迟 (ms)')
        ax2.set_title('推理速度对比')
        ax2.set_xticks(x + width / 2)
        ax2.set_xticklabels([f"{s[0]}×{s[1]}" for s in scenarios])
        ax2.legend()
        
        # 内存使用对比
        ax3 = axes[1, 0]
        memory_data = {}
        for method in methods:
            memory_data[method] = []
            for scenario in scenarios:
                result = next((r for r in self.results 
                             if r.method_name == method and r.scenario_size == scenario), None)
                memory_data[method].append(result.memory_usage_mb if result and result.memory_usage_mb > 0 else 0)
        
        for i, (method, memory) in enumerate(memory_data.items()):
            ax3.bar(x + i * width, memory, width, label=method)
        
        ax3.set_xlabel('场景规模')
        ax3.set_ylabel('内存使用 (MB)')
        ax3.set_title('内存使用对比')
        ax3.set_xticks(x + width / 2)
        ax3.set_xticklabels([f"{s[0]}×{s[1]}" for s in scenarios])
        ax3.legend()
        
        # 综合性能雷达图
        ax4 = axes[1, 1]
        ax4.remove()  # 移除子图
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')
        
        # 计算归一化性能指标
        metrics = ['训练速度', '推理速度', '内存效率']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for method in methods:
            method_results = [r for r in self.results if r.method_name == method]
            if not method_results:
                continue
                
            # 归一化指标 (越小越好的指标需要取倒数)
            avg_training = np.mean([r.training_time_per_episode for r in method_results])
            avg_inference = np.mean([r.inference_time_ms for r in method_results])
            avg_memory = np.mean([r.memory_usage_mb for r in method_results if r.memory_usage_mb > 0])
            
            # 归一化到0-1范围 (1表示最好)
            max_training = max([np.mean([r.training_time_per_episode for r in self.results if r.method_name == m]) 
                               for m in methods])
            max_inference = max([np.mean([r.inference_time_ms for r in self.results if r.method_name == m]) 
                                for m in methods])
            max_memory = max([np.mean([r.memory_usage_mb for r in self.results if r.method_name == m and r.memory_usage_mb > 0]) 
                             for m in methods])
            
            values = [
                1 - (avg_training / max_training) if max_training > 0 else 0,
                1 - (avg_inference / max_inference) if max_inference > 0 else 0,
                1 - (avg_memory / max_memory) if max_memory > 0 else 0
            ]
            values += values[:1]  # 闭合图形
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=method)
            ax4.fill(angles, values, alpha=0.25)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('综合性能对比', y=1.08)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"性能对比图表已保存到: {self.output_dir / 'performance_comparison.png'}")
