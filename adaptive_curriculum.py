# -*- coding: utf-8 -*-
"""
自适应课程训练模块

本模块实现了基于智能体表现的自适应课程训练机制，包括：
- 性能监控器：跟踪和分析智能体在各个难度等级的表现
- 晋级判断器：基于性能数据做出晋级、继续或降级决策
- 自适应训练控制器：协调各组件，控制整个自适应训练流程
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
from enum import Enum
import logging

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    completion_rate: float
    episode_reward: float
    step_count: int
    training_time: float
    memory_usage: Optional[float] = None
    convergence_indicator: float = 0.0


@dataclass
class LevelPerformance:
    """单个等级的性能数据类"""
    level_index: int
    level_name: str
    completion_rates: Deque[float] = field(default_factory=deque)
    episode_count: int = 0
    total_episodes: int = 0
    start_time: float = field(default_factory=time.time)
    best_completion_rate: float = 0.0
    average_completion_rate: float = 0.0
    is_mastered: bool = False
    promotion_episode: Optional[int] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if not isinstance(self.completion_rates, deque):
            self.completion_rates = deque(self.completion_rates)


class Decision(Enum):
    """决策类型枚举"""
    CONTINUE = "continue"                   # 继续当前等级
    PROMOTE = "promote"                     # 晋级到下一等级
    REGRESS = "regress"                     # 降级到上一等级
    TIMEOUT_PROMOTE = "timeout_promote"     # 超时强制晋级
    TIMEOUT_CONTINUE = "timeout_continue"   # 超时继续训练


class PerformanceMonitor:
    """
    性能监控器类
    
    职责：跟踪和分析智能体在各个难度等级的表现
    """
    
    def __init__(self, config):
        """
        初始化性能监控器
        
        Args:
            config: 配置对象，包含自适应课程训练参数
        """
        self.config = config
        self.current_level_performance: Optional[LevelPerformance] = None
        self.level_history: List[LevelPerformance] = []
        self.performance_window_size = config.CURRICULUM_PERFORMANCE_WINDOW
        self.smoothing_factor = config.CURRICULUM_PERFORMANCE_SMOOTHING
        
        logger.info(f"性能监控器初始化完成，窗口大小: {self.performance_window_size}")
    
    def initialize_level(self, level_index: int, level_name: str) -> LevelPerformance:
        """
        初始化新的等级性能跟踪
        
        Args:
            level_index: 等级索引
            level_name: 等级名称
            
        Returns:
            LevelPerformance: 新创建的等级性能对象
        """
        # 如果有当前等级，保存到历史记录
        if self.current_level_performance is not None:
            self.level_history.append(self.current_level_performance)
        
        # 创建新的等级性能跟踪
        self.current_level_performance = LevelPerformance(
            level_index=level_index,
            level_name=level_name,
            completion_rates=deque(maxlen=self.performance_window_size),
            start_time=time.time()
        )
        
        logger.info(f"初始化等级 {level_index} ({level_name}) 的性能监控")
        return self.current_level_performance
    
    def record_episode_performance(self, metrics: PerformanceMetrics) -> None:
        """
        记录单轮训练表现
        
        Args:
            metrics: 性能指标数据
        """
        try:
            if self.current_level_performance is None:
                raise PerformanceMonitoringError(
                    "必须先初始化等级性能跟踪",
                    level_name="Unknown",
                    episode_count=0
                )
            
            # 验证输入数据
            completion_rate = metrics.completion_rate
            if not (0.0 <= completion_rate <= 1.0):
                logger.warning(f"完成率超出有效范围 [0,1]: {completion_rate}，将裁剪到有效范围")
                completion_rate = max(0.0, min(1.0, completion_rate))
            
            # 记录完成率到滑动窗口
            self.current_level_performance.completion_rates.append(completion_rate)
            self.current_level_performance.episode_count += 1
            self.current_level_performance.total_episodes += 1
            
            # 更新最佳完成率
            if completion_rate > self.current_level_performance.best_completion_rate:
                self.current_level_performance.best_completion_rate = completion_rate
            
            # 计算平均完成率（使用指数移动平均）
            if self.current_level_performance.average_completion_rate == 0.0:
                self.current_level_performance.average_completion_rate = completion_rate
            else:
                alpha = self.smoothing_factor
                self.current_level_performance.average_completion_rate = (
                    alpha * completion_rate + 
                    (1 - alpha) * self.current_level_performance.average_completion_rate
                )
            
            logger.debug(f"记录轮次 {self.current_level_performance.episode_count} 表现: "
                        f"完成率={completion_rate:.3f}, 平均完成率={self.current_level_performance.average_completion_rate:.3f}")
        
        except Exception as e:
            error_msg = f"记录性能数据失败: {e}"
            raise PerformanceMonitoringError(
                error_msg,
                level_name=self.current_level_performance.level_name if self.current_level_performance else "Unknown",
                episode_count=self.current_level_performance.episode_count if self.current_level_performance else 0
            )
    
    def calculate_mastery_score(self) -> float:
        """
        计算掌握度分数
        
        Returns:
            float: 掌握度分数 (0.0 - 1.0)
        """
        if self.current_level_performance is None or not self.current_level_performance.completion_rates:
            return 0.0
        
        # 使用滑动窗口内的平均完成率作为掌握度分数
        completion_rates = list(self.current_level_performance.completion_rates)
        mastery_score = np.mean(completion_rates)
        
        return float(mastery_score)
    
    def is_level_mastered(self) -> bool:
        """
        判断是否掌握当前等级
        
        Returns:
            bool: 是否掌握当前等级
        """
        if self.current_level_performance is None:
            return False
        
        # 检查是否有足够的数据
        if len(self.current_level_performance.completion_rates) < self.performance_window_size:
            return False
        
        # 计算掌握度分数
        mastery_score = self.calculate_mastery_score()
        
        # 判断是否达到掌握阈值
        is_mastered = mastery_score >= self.config.CURRICULUM_MASTERY_THRESHOLD
        
        if is_mastered and not self.current_level_performance.is_mastered:
            self.current_level_performance.is_mastered = True
            self.current_level_performance.promotion_episode = self.current_level_performance.episode_count
            logger.info(f"等级 {self.current_level_performance.level_name} 已掌握！"
                       f"掌握度分数: {mastery_score:.3f}, 训练轮次: {self.current_level_performance.episode_count}")
        
        return is_mastered
    
    def should_promote(self) -> bool:
        """
        判断是否应该晋级
        
        Returns:
            bool: 是否应该晋级
        """
        if self.current_level_performance is None:
            return False
        
        # 检查最小训练轮次要求
        if self.current_level_performance.episode_count < self.config.CURRICULUM_MIN_EPISODES_PER_LEVEL:
            return False
        
        # 检查是否掌握当前等级
        return self.is_level_mastered()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Returns:
            Dict: 性能摘要数据
        """
        if self.current_level_performance is None:
            return {}
        
        perf = self.current_level_performance
        completion_rates = list(perf.completion_rates)
        
        summary = {
            "level_index": perf.level_index,
            "level_name": perf.level_name,
            "episode_count": perf.episode_count,
            "mastery_score": self.calculate_mastery_score(),
            "is_mastered": perf.is_mastered,
            "best_completion_rate": perf.best_completion_rate,
            "average_completion_rate": perf.average_completion_rate,
            "recent_completion_rates": completion_rates[-5:] if completion_rates else [],
            "training_time": time.time() - perf.start_time,
            "window_full": len(completion_rates) >= self.performance_window_size,
            "promotion_episode": perf.promotion_episode
        }
        
        return summary
    
    def get_level_history_summary(self) -> List[Dict[str, Any]]:
        """
        获取所有等级的历史摘要
        
        Returns:
            List[Dict]: 历史等级摘要列表
        """
        history_summary = []
        
        for level_perf in self.level_history:
            completion_rates = list(level_perf.completion_rates)
            avg_completion = np.mean(completion_rates) if completion_rates else 0.0
            
            summary = {
                "level_index": level_perf.level_index,
                "level_name": level_perf.level_name,
                "episode_count": level_perf.episode_count,
                "average_completion_rate": avg_completion,
                "best_completion_rate": level_perf.best_completion_rate,
                "is_mastered": level_perf.is_mastered,
                "promotion_episode": level_perf.promotion_episode,
                "training_time": time.time() - level_perf.start_time
            }
            history_summary.append(summary)
        
        return history_summary
    
    def reset(self):
        """重置性能监控器"""
        self.current_level_performance = None
        self.level_history.clear()
        logger.info("性能监控器已重置")


class PromotionDecider:
    """
    晋级判断器类
    
    职责：基于性能数据做出晋级、继续或降级决策
    """
    
    def __init__(self, config):
        """
        初始化晋级判断器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.regression_count = 0
        self.last_promotion_episode = 0
        
        logger.info("晋级判断器初始化完成")
    
    def make_decision(self, performance: LevelPerformance, monitor: PerformanceMonitor) -> Decision:
        """
        基于性能数据做出决策
        
        Args:
            performance: 等级性能数据
            monitor: 性能监控器实例
            
        Returns:
            Decision: 决策结果
        """
        try:
            # 验证输入参数
            if performance is None:
                raise PromotionDecisionError("性能数据不能为空")
            
            if monitor is None:
                raise PromotionDecisionError("性能监控器不能为空")
            
            # 1. 检查最小训练轮次要求
            if performance.episode_count < self.config.CURRICULUM_MIN_EPISODES_PER_LEVEL:
                return Decision.CONTINUE
            
            # 2. 检查晋级冷却期
            if (performance.episode_count - self.last_promotion_episode) < self.config.CURRICULUM_PROMOTION_COOLDOWN:
                return Decision.CONTINUE
            
            # 3. 检查掌握度
            try:
                if monitor.is_level_mastered():
                    self.last_promotion_episode = performance.episode_count
                    return Decision.PROMOTE
            except Exception as e:
                logger.warning(f"掌握度检查失败: {e}，使用备用判断逻辑")
                # 使用简化的掌握度判断
                if performance.best_completion_rate >= self.config.CURRICULUM_MASTERY_THRESHOLD:
                    return Decision.PROMOTE
            
            # 4. 检查超时
            if performance.episode_count >= self.config.CURRICULUM_MAX_EPISODES_PER_LEVEL:
                try:
                    # 根据配置决定超时后的行为
                    mastery_score = monitor.calculate_mastery_score()
                    if mastery_score >= (self.config.CURRICULUM_MASTERY_THRESHOLD * 0.8):  # 80%的掌握度
                        return Decision.TIMEOUT_PROMOTE
                    else:
                        return Decision.TIMEOUT_CONTINUE
                except Exception as e:
                    logger.warning(f"超时决策计算失败: {e}，使用保守策略")
                    return Decision.TIMEOUT_CONTINUE
            
            # 5. 检查是否需要降级（可选）
            if self.config.CURRICULUM_ENABLE_REGRESSION:
                try:
                    if self.should_regress(performance, monitor):
                        return Decision.REGRESS
                except Exception as e:
                    logger.warning(f"降级判断失败: {e}，跳过降级检查")
            
            return Decision.CONTINUE
        
        except PromotionDecisionError:
            raise  # 重新抛出已知的决策错误
        except Exception as e:
            # 捕获未预期的错误
            decision_context = {
                'level_name': performance.level_name if performance else "Unknown",
                'episode_count': performance.episode_count if performance else 0,
                'error_type': type(e).__name__
            }
            raise PromotionDecisionError(f"决策制定过程中发生未预期错误: {e}", decision_context)
    
    def should_regress(self, performance: LevelPerformance, monitor: PerformanceMonitor) -> bool:
        """
        判断是否应该降级
        
        Args:
            performance: 等级性能数据
            monitor: 性能监控器实例
            
        Returns:
            bool: 是否应该降级
        """
        # 检查降级次数限制
        if self.regression_count >= self.config.CURRICULUM_MAX_REGRESSION_COUNT:
            logger.debug(f"降级次数已达上限: {self.regression_count}/{self.config.CURRICULUM_MAX_REGRESSION_COUNT}")
            return False
        
        # 检查是否在第一个等级（不能再降级）
        if performance.level_index <= 0:
            logger.debug("已在第一个等级，无法降级")
            return False
        
        # 检查是否有足够的数据
        if len(performance.completion_rates) < self.config.CURRICULUM_PERFORMANCE_WINDOW:
            logger.debug(f"数据不足，无法判断降级: {len(performance.completion_rates)}/{self.config.CURRICULUM_PERFORMANCE_WINDOW}")
            return False
        
        # 检查表现是否持续低于退步阈值
        recent_performance = monitor.calculate_mastery_score()
        if recent_performance < self.config.CURRICULUM_REGRESSION_THRESHOLD:
            # 额外检查：确保不是暂时的波动，需要持续较长时间的低表现
            min_episodes_for_regression = max(
                self.config.CURRICULUM_MIN_EPISODES_PER_LEVEL * 2,
                self.config.CURRICULUM_PERFORMANCE_WINDOW
            )
            
            if performance.episode_count >= min_episodes_for_regression:
                # 进一步检查：最近的表现是否确实在恶化
                completion_rates = list(performance.completion_rates)
                if len(completion_rates) >= 10:  # 至少需要10个数据点
                    # 计算前半部分和后半部分的平均值
                    mid_point = len(completion_rates) // 2
                    early_avg = np.mean(completion_rates[:mid_point])
                    recent_avg = np.mean(completion_rates[mid_point:])
                    
                    # 如果最近的表现明显低于早期表现，考虑降级
                    performance_decline = early_avg - recent_avg
                    decline_threshold = 0.1  # 10%的下降阈值
                    
                    if performance_decline > decline_threshold:
                        logger.info(f"检测到性能下降: 早期平均={early_avg:.3f}, 最近平均={recent_avg:.3f}, 下降={performance_decline:.3f}")
                        return True
                else:
                    # 数据点不足时，使用简单的阈值判断
                    return True
        
        return False
    
    def handle_regression(self, current_level_name: str = "", target_level_name: str = ""):
        """
        处理降级操作
        
        Args:
            current_level_name: 当前等级名称
            target_level_name: 目标等级名称
        """
        self.regression_count += 1
        
        # 记录详细的降级信息
        regression_info = {
            'regression_count': self.regression_count,
            'max_regressions': self.config.CURRICULUM_MAX_REGRESSION_COUNT,
            'current_level': current_level_name,
            'target_level': target_level_name,
            'timestamp': time.time()
        }
        
        logger.warning(f"执行降级操作 #{self.regression_count}: {current_level_name} -> {target_level_name}")
        logger.info(f"降级统计: {self.regression_count}/{self.config.CURRICULUM_MAX_REGRESSION_COUNT}")
        
        # 如果接近降级次数上限，给出警告
        if self.regression_count >= self.config.CURRICULUM_MAX_REGRESSION_COUNT - 1:
            logger.warning("⚠️ 接近最大降级次数限制！后续将不再允许降级")
        
        return regression_info
    
    def can_regress_again(self) -> bool:
        """
        检查是否还能再次降级
        
        Returns:
            bool: 是否还能降级
        """
        return self.regression_count < self.config.CURRICULUM_MAX_REGRESSION_COUNT
    
    def get_regression_statistics(self) -> Dict[str, Any]:
        """
        获取降级统计信息
        
        Returns:
            Dict: 降级统计数据
        """
        return {
            'total_regressions': self.regression_count,
            'max_allowed_regressions': self.config.CURRICULUM_MAX_REGRESSION_COUNT,
            'remaining_regressions': max(0, self.config.CURRICULUM_MAX_REGRESSION_COUNT - self.regression_count),
            'regression_enabled': self.config.CURRICULUM_ENABLE_REGRESSION,
            'can_regress_again': self.can_regress_again()
        }
    
    def get_decision_rationale(self, decision: Decision, performance: LevelPerformance, 
                             monitor: PerformanceMonitor) -> str:
        """
        获取决策理由说明
        
        Args:
            decision: 决策结果
            performance: 等级性能数据
            monitor: 性能监控器实例
            
        Returns:
            str: 决策理由说明
        """
        mastery_score = monitor.calculate_mastery_score()
        
        if decision == Decision.PROMOTE:
            return (f"晋级：掌握度分数 {mastery_score:.3f} 达到阈值 {self.config.CURRICULUM_MASTERY_THRESHOLD:.3f}，"
                   f"训练轮次 {performance.episode_count}")
        
        elif decision == Decision.TIMEOUT_PROMOTE:
            return (f"超时晋级：训练轮次达到上限 {self.config.CURRICULUM_MAX_EPISODES_PER_LEVEL}，"
                   f"掌握度分数 {mastery_score:.3f}")
        
        elif decision == Decision.TIMEOUT_CONTINUE:
            return (f"超时继续：训练轮次达到上限但掌握度不足，"
                   f"掌握度分数 {mastery_score:.3f} < {self.config.CURRICULUM_MASTERY_THRESHOLD:.3f}")
        
        elif decision == Decision.REGRESS:
            return (f"降级：表现持续低于退步阈值，"
                   f"掌握度分数 {mastery_score:.3f} < {self.config.CURRICULUM_REGRESSION_THRESHOLD:.3f}")
        
        else:  # Decision.CONTINUE
            return (f"继续：掌握度分数 {mastery_score:.3f}，"
                   f"训练轮次 {performance.episode_count}/{self.config.CURRICULUM_MAX_EPISODES_PER_LEVEL}")
    
    def reset(self):
        """重置晋级判断器"""
        self.regression_count = 0
        self.last_promotion_episode = 0
        logger.info("晋级判断器已重置")


# 异常类定义
class AdaptiveCurriculumError(Exception):
    """自适应课程训练基础异常"""
    
    def __init__(self, message: str, error_code: str = None, recovery_suggestion: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.recovery_suggestion = recovery_suggestion
        self.timestamp = time.time()
    
    def __str__(self):
        base_msg = super().__str__()
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.recovery_suggestion:
            base_msg += f"\n建议: {self.recovery_suggestion}"
        return base_msg


class InvalidConfigurationError(AdaptiveCurriculumError):
    """无效配置异常"""
    
    def __init__(self, message: str, invalid_params: List[str] = None):
        recovery_suggestion = "请检查配置参数并重置为默认值"
        if invalid_params:
            recovery_suggestion += f"，无效参数: {', '.join(invalid_params)}"
        
        super().__init__(message, "CONFIG_ERROR", recovery_suggestion)
        self.invalid_params = invalid_params or []


class PerformanceMonitoringError(AdaptiveCurriculumError):
    """性能监控异常"""
    
    def __init__(self, message: str, level_name: str = None, episode_count: int = None):
        recovery_suggestion = "尝试重置性能监控器或使用备用指标"
        super().__init__(message, "MONITOR_ERROR", recovery_suggestion)
        self.level_name = level_name
        self.episode_count = episode_count


class PromotionDecisionError(AdaptiveCurriculumError):
    """晋级决策异常"""
    
    def __init__(self, message: str, decision_context: Dict[str, Any] = None):
        recovery_suggestion = "将继续当前等级训练，请检查决策逻辑"
        super().__init__(message, "DECISION_ERROR", recovery_suggestion)
        self.decision_context = decision_context or {}


class TrainingInterruptionError(AdaptiveCurriculumError):
    """训练中断异常"""
    
    def __init__(self, message: str, current_level: int = None, episode_count: int = None):
        recovery_suggestion = "保存当前状态并尝试从中断点恢复训练"
        super().__init__(message, "TRAINING_ERROR", recovery_suggestion)
        self.current_level = current_level
        self.episode_count = episode_count


class RegressionLimitExceededError(AdaptiveCurriculumError):
    """降级次数超限异常"""
    
    def __init__(self, message: str, regression_count: int = None, max_regressions: int = None):
        recovery_suggestion = "已达到最大降级次数，将继续当前等级或强制晋级"
        super().__init__(message, "REGRESSION_LIMIT", recovery_suggestion)
        self.regression_count = regression_count
        self.max_regressions = max_regressions


# 错误处理工具类
class ErrorHandler:
    """自适应课程训练错误处理器"""
    
    def __init__(self, config):
        self.config = config
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
    
    def handle_configuration_error(self, error: InvalidConfigurationError) -> bool:
        """
        处理配置错误
        
        Args:
            error: 配置错误异常
            
        Returns:
            bool: 是否成功恢复
        """
        logger.error(f"配置错误: {error}")
        
        # 记录错误
        self._record_error("configuration", str(error), error.invalid_params)
        
        try:
            # 尝试重置为默认配置
            self.config.reset_adaptive_curriculum_to_defaults()
            logger.info("已重置为默认配置")
            return True
        except Exception as e:
            logger.error(f"配置恢复失败: {e}")
            return False
    
    def handle_monitoring_error(self, error: PerformanceMonitoringError, monitor: PerformanceMonitor) -> bool:
        """
        处理性能监控错误
        
        Args:
            error: 性能监控错误异常
            monitor: 性能监控器实例
            
        Returns:
            bool: 是否成功恢复
        """
        logger.error(f"性能监控错误: {error}")
        
        # 记录错误
        self._record_error("monitoring", str(error), {
            'level_name': error.level_name,
            'episode_count': error.episode_count
        })
        
        try:
            # 尝试使用备用指标或重置监控器
            if self.recovery_attempts < self.max_recovery_attempts:
                self.recovery_attempts += 1
                logger.info(f"尝试恢复性能监控器 (第{self.recovery_attempts}次)")
                
                # 可以选择重置或使用简化的监控逻辑
                return True
            else:
                logger.warning("性能监控恢复次数已达上限，使用简化监控")
                return False
        except Exception as e:
            logger.error(f"性能监控恢复失败: {e}")
            return False
    
    def handle_decision_error(self, error: PromotionDecisionError) -> Decision:
        """
        处理决策错误
        
        Args:
            error: 决策错误异常
            
        Returns:
            Decision: 默认决策
        """
        logger.error(f"决策错误: {error}")
        
        # 记录错误
        self._record_error("decision", str(error), error.decision_context)
        
        # 返回安全的默认决策
        logger.info("使用默认决策: 继续当前等级")
        return Decision.CONTINUE
    
    def handle_training_interruption(self, error: TrainingInterruptionError) -> Dict[str, Any]:
        """
        处理训练中断错误
        
        Args:
            error: 训练中断错误异常
            
        Returns:
            Dict: 恢复信息
        """
        logger.error(f"训练中断: {error}")
        
        # 记录错误
        self._record_error("training_interruption", str(error), {
            'current_level': error.current_level,
            'episode_count': error.episode_count
        })
        
        # 返回恢复信息
        recovery_info = {
            'should_save_state': True,
            'resume_from_level': error.current_level,
            'resume_from_episode': error.episode_count,
            'recovery_strategy': 'checkpoint_resume'
        }
        
        logger.info(f"准备从等级 {error.current_level} 轮次 {error.episode_count} 恢复")
        return recovery_info
    
    def _record_error(self, error_type: str, message: str, context: Any = None):
        """记录错误信息"""
        error_record = {
            'timestamp': time.time(),
            'type': error_type,
            'message': message,
            'context': context,
            'recovery_attempts': self.recovery_attempts
        }
        self.error_history.append(error_record)
        
        # 限制错误历史记录数量
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-50:]  # 保留最近50条
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_history:
            return {'total_errors': 0, 'error_types': {}}
        
        error_types = {}
        for error in self.error_history:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recent_errors': self.error_history[-5:],  # 最近5个错误
            'recovery_attempts': self.recovery_attempts
        }
    
    def reset_recovery_attempts(self):
        """重置恢复尝试次数"""
        self.recovery_attempts = 0
        logger.info("错误恢复尝试次数已重置")