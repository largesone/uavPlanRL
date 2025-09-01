"""
性能监控系统 - 实现内存使用监控和性能指标收集
支持大规模场景下的内存占用优化和实时监控
"""

import psutil
import torch
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging
from collections import deque
import json

@dataclass
class MemorySnapshot:
    """内存快照数据结构"""
    timestamp: float
    cpu_memory_mb: float
    gpu_memory_mb: float
    gpu_memory_cached_mb: float
    process_memory_mb: float
    n_uavs: int
    n_targets: int
    stage: str

class PerformanceMonitor:
    """性能监控器 - 监控内存使用、GPU占用和训练性能"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.memory_history: deque = deque(maxlen=max_history)
        self.performance_history: deque = deque(maxlen=max_history)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self, interval: float = 1.0):
        """开始性能监控"""
        if self.monitoring:
            self.logger.warning("监控已在运行中")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"开始性能监控，间隔: {interval}秒")
        
    def stop_monitoring(self):
        """停止性能监控"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("性能监控已停止")
        
    def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                snapshot = self._capture_memory_snapshot()
                self.memory_history.append(snapshot)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"监控过程中出错: {e}")
                
    def _capture_memory_snapshot(self) -> MemorySnapshot:
        """捕获内存快照"""
        # CPU内存使用
        process = psutil.Process()
        cpu_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU内存使用
        gpu_memory = 0.0
        gpu_cached = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            
        return MemorySnapshot(
            timestamp=time.time(),
            cpu_memory_mb=cpu_memory,
            gpu_memory_mb=gpu_memory,
            gpu_memory_cached_mb=gpu_cached,
            process_memory_mb=process_memory,
            n_uavs=0,  # 需要外部设置
            n_targets=0,  # 需要外部设置
            stage="unknown"  # 需要外部设置
        )
        
    def record_memory_snapshot(self, n_uavs: int, n_targets: int, stage: str):
        """记录内存快照"""
        snapshot = self._capture_memory_snapshot()
        snapshot.n_uavs = n_uavs
        snapshot.n_targets = n_targets
        snapshot.stage = stage
        self.memory_history.append(snapshot)
        
    def get_memory_usage_summary(self) -> Dict:
        """获取内存使用摘要"""
        if not self.memory_history:
            return {"error": "无内存数据"}
            
        snapshots = list(self.memory_history)
        cpu_memories = [s.cpu_memory_mb for s in snapshots]
        gpu_memories = [s.gpu_memory_mb for s in snapshots]
        process_memories = [s.process_memory_mb for s in snapshots]
        
        return {
            "cpu_memory": {
                "current": cpu_memories[-1],
                "max": max(cpu_memories),
                "avg": sum(cpu_memories) / len(cpu_memories)
            },
            "gpu_memory": {
                "current": gpu_memories[-1],
                "max": max(gpu_memories),
                "avg": sum(gpu_memories) / len(gpu_memories)
            },
            "process_memory": {
                "current": process_memories[-1],
                "max": max(process_memories),
                "avg": sum(process_memories) / len(process_memories)
            },
            "total_snapshots": len(snapshots)
        }
        
    def optimize_memory_usage(self) -> List[str]:
        """内存使用优化建议"""
        suggestions = []
        summary = self.get_memory_usage_summary()
        
        if "error" in summary:
            return ["无法获取内存数据，请先开始监控"]
            
        # GPU内存优化建议
        if summary["gpu_memory"]["max"] > 8000:  # 8GB
            suggestions.append("GPU内存使用过高，建议减少批次大小或启用梯度检查点")
            
        if summary["gpu_memory"]["current"] > summary["gpu_memory"]["avg"] * 1.5:
            suggestions.append("当前GPU内存使用异常高，可能存在内存泄漏")
            
        # CPU内存优化建议
        if summary["process_memory"]["max"] > 4000:  # 4GB
            suggestions.append("进程内存使用较高，建议优化数据加载和缓存策略")
            
        if not suggestions:
            suggestions.append("内存使用正常，无需优化")
            
        return suggestions
        
    def export_performance_data(self, filepath: str):
        """导出性能数据到文件"""
        data = {
            "memory_history": [
                {
                    "timestamp": s.timestamp,
                    "cpu_memory_mb": s.cpu_memory_mb,
                    "gpu_memory_mb": s.gpu_memory_mb,
                    "gpu_memory_cached_mb": s.gpu_memory_cached_mb,
                    "process_memory_mb": s.process_memory_mb,
                    "n_uavs": s.n_uavs,
                    "n_targets": s.n_targets,
                    "stage": s.stage
                }
                for s in self.memory_history
            ],
            "summary": self.get_memory_usage_summary(),
            "optimization_suggestions": self.optimize_memory_usage()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"性能数据已导出到: {filepath}")
