# -*- coding: utf-8 -*-
# 文件名: ensemble_inference_manager.py
# 描述: 集成推理管理器，支持指定场景列表或随机场景的集成推理

import os
import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from config import Config
from scenarios import get_balanced_scenario, get_small_scenario, get_complex_scenario
from evaluator import ModelEvaluator
from trainer import GraphRLSolver
from environment import DirectedGraph, UAVTaskEnv
from entities import UAV, Target

class EnsembleInferenceManager:
    """集成推理管理器"""
    
    def __init__(self, config: Config):
        """
        初始化集成推理管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.evaluator = ModelEvaluator(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def run_ensemble_inference(self, 
                             model_paths: List[str], 
                             scenarios: List[str] = None,
                             num_plans: int = 5,
                             top_models: int = 5,
                             output_summary: bool = True) -> Dict[str, Any]:
        """
        运行集成推理
        
        Args:
            model_paths: 模型文件路径列表
            scenarios: 场景列表，如果包含'random'则生成随机场景
            num_plans: 每个场景生成的方案数量
            top_models: 使用的最优模型数量
            output_summary: 是否输出汇总信息
            
        Returns:
            Dict: 推理结果汇总
        """
        print("🚀 启动集成推理模式")
        print("=" * 60)
        
        # 验证模型文件
        valid_models = self._validate_models(model_paths)
        if not valid_models:
            print("❌ 没有有效的模型文件")
            return {}
        
        # 处理场景列表
        if scenarios is None:
            scenarios = ['small', 'balanced']
        
        scenario_list = self._process_scenarios(scenarios)
        
        print(f"📊 集成推理配置:")
        print(f"   模型数量: {len(valid_models)}")
        print(f"   使用模型: {min(top_models, len(valid_models))}")
        print(f"   场景数量: {len(scenario_list)}")
        print(f"   每场景方案数: {num_plans}")
        print("=" * 60)
        
        # 执行推理
        all_results = {}
        total_start_time = time.time()
        
        for i, scenario_info in enumerate(scenario_list, 1):
            scenario_name = scenario_info['name']
            scenario_data = scenario_info['data']
            
            print(f"\n🎯 场景 {i}/{len(scenario_list)}: {scenario_name}")
            print("-" * 40)
            
            # 执行单个场景的集成推理
            scenario_result = self._run_scenario_inference(
                valid_models[:top_models],
                scenario_data,
                scenario_name,
                num_plans
            )
            
            all_results[scenario_name] = scenario_result
            
            # 显示场景结果
            if scenario_result:
                best_plan = scenario_result.get('best_plan', {})
                print(f"✅ 最优方案: 完成率={best_plan.get('completion_rate', 0):.3f}, "
                      f"总奖励={best_plan.get('total_reward', 0):.1f}")
            else:
                print("❌ 场景推理失败")
        
        # 汇总结果
        total_time = time.time() - total_start_time
        summary = self._generate_summary(all_results, total_time)
        
        if output_summary:
            self._print_summary(summary)
        
        # 保存结果
        self._save_results(all_results, summary)
        
        return {
            'all_results': all_results,
            'summary': summary,
            'execution_time': total_time
        }
    
    def _validate_models(self, model_paths: List[str]) -> List[str]:
        """验证模型文件是否存在"""
        valid_models = []
        
        for path in model_paths:
            if os.path.exists(path):
                valid_models.append(path)
                print(f"✅ 模型文件: {os.path.basename(path)}")
            else:
                print(f"❌ 模型文件不存在: {path}")
        
        return valid_models
    
    def _process_scenarios(self, scenarios: List[str]) -> List[Dict[str, Any]]:
        """处理场景列表，支持random场景"""
        scenario_list = []
        
        for scenario in scenarios:
            if scenario == 'random':
                # 生成随机场景
                scenario_name = f"random_{datetime.now().strftime('%H%M%S')}"
                try:
                    uavs, targets, obstacles = generate_random_scenario()
                    scenario_list.append({
                        'name': scenario_name,
                        'data': (uavs, targets, obstacles)
                    })
                    print(f"🎲 生成随机场景: {scenario_name}")
                except Exception as e:
                    print(f"⚠️ 生成随机场景失败: {e}")
                    continue
            else:
                # 使用预定义场景
                try:
                    if scenario == 'balanced':
                        uavs, targets, obstacles = get_balanced_scenario(50.0)
                    elif scenario == 'small':
                        uavs, targets, obstacles = get_small_scenario(50.0)
                    elif scenario == 'complex':
                        uavs, targets, obstacles = get_complex_scenario(50.0)
                    else:
                        print(f"⚠️ 未知场景: {scenario}")
                        continue
                    
                    scenario_list.append({
                        'name': scenario,
                        'data': (uavs, targets, obstacles)
                    })
                    print(f"📋 加载预定义场景: {scenario}")
                except Exception as e:
                    print(f"⚠️ 加载场景失败 {scenario}: {e}")
                    continue
        
        return scenario_list
    
    def _run_scenario_inference(self, 
                              model_paths: List[str], 
                              scenario_data: tuple,
                              scenario_name: str,
                              num_plans: int) -> Dict[str, Any]:
        """执行单个场景的集成推理"""
        uavs, targets, obstacles = scenario_data
        
        print(f"   UAV数量: {len(uavs)}, 目标数量: {len(targets)}, 障碍物数量: {len(obstacles)}")
        
        try:
            # 使用evaluator的集成推理功能
            result = self.evaluator._ensemble_inference(model_paths, uavs, targets, obstacles)
            
            if result is None:
                return {}
            
            # 格式化结果
            formatted_result = {
                'scenario_info': {
                    'name': scenario_name,
                    'uavs': len(uavs),
                    'targets': len(targets),
                    'obstacles': len(obstacles)
                },
                'inference_result': result,
                'best_plan': {
                    'completion_rate': result.get('completion_rate', 0),
                    'total_reward': result.get('total_reward', 0),
                    'step_count': result.get('step_count', 0)
                },
                'model_count': len(model_paths)
            }
            
            return formatted_result
            
        except Exception as e:
            print(f"❌ 场景推理失败: {e}")
            return {}
    
    def _generate_summary(self, all_results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """生成汇总结果"""
        if not all_results:
            return {}
        
        # 统计数据
        total_scenarios = len(all_results)
        successful_scenarios = sum(1 for r in all_results.values() if r)
        
        completion_rates = []
        total_rewards = []
        step_counts = []
        
        for result in all_results.values():
            if result and 'best_plan' in result:
                best_plan = result['best_plan']
                completion_rates.append(best_plan.get('completion_rate', 0))
                total_rewards.append(best_plan.get('total_reward', 0))
                step_counts.append(best_plan.get('step_count', 0))
        
        # 计算统计指标
        summary = {
            'total_scenarios': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'success_rate': successful_scenarios / total_scenarios if total_scenarios > 0 else 0,
            'execution_time': total_time,
            'avg_completion_rate': np.mean(completion_rates) if completion_rates else 0,
            'avg_total_reward': np.mean(total_rewards) if total_rewards else 0,
            'avg_step_count': np.mean(step_counts) if step_counts else 0,
            'max_completion_rate': max(completion_rates) if completion_rates else 0,
            'max_total_reward': max(total_rewards) if total_rewards else 0,
            'min_completion_rate': min(completion_rates) if completion_rates else 0,
            'min_total_reward': min(total_rewards) if total_rewards else 0
        }
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """打印汇总信息"""
        print("\n" + "=" * 60)
        print("🏆 集成推理汇总结果")
        print("=" * 60)
        
        print(f"执行统计:")
        print(f"  总场景数: {summary['total_scenarios']}")
        print(f"  成功场景数: {summary['successful_scenarios']}")
        print(f"  成功率: {summary['success_rate']:.1%}")
        print(f"  总耗时: {summary['execution_time']:.2f}秒")
        
        print(f"\n性能指标:")
        print(f"  平均完成率: {summary['avg_completion_rate']:.3f}")
        print(f"  平均总奖励: {summary['avg_total_reward']:.1f}")
        print(f"  平均步数: {summary['avg_step_count']:.1f}")
        
        print(f"\n最优指标:")
        print(f"  最高完成率: {summary['max_completion_rate']:.3f}")
        print(f"  最高总奖励: {summary['max_total_reward']:.1f}")
        
        print(f"\n最差指标:")
        print(f"  最低完成率: {summary['min_completion_rate']:.3f}")
        print(f"  最低总奖励: {summary['min_total_reward']:.1f}")
        
        print("=" * 60)
    
    def _save_results(self, all_results: Dict[str, Any], summary: Dict[str, Any]):
        """保存推理结果到文件"""
        try:
            output_dir = os.path.join("output", "ensemble_inference")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存详细结果
            results_file = os.path.join(output_dir, f"ensemble_inference_results_{timestamp}.txt")
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("集成推理详细结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for scenario_name, result in all_results.items():
                    f.write(f"场景: {scenario_name}\n")
                    f.write("-" * 30 + "\n")
                    
                    if result:
                        scenario_info = result.get('scenario_info', {})
                        best_plan = result.get('best_plan', {})
                        
                        f.write(f"  UAV数量: {scenario_info.get('uavs', 0)}\n")
                        f.write(f"  目标数量: {scenario_info.get('targets', 0)}\n")
                        f.write(f"  障碍物数量: {scenario_info.get('obstacles', 0)}\n")
                        f.write(f"  完成率: {best_plan.get('completion_rate', 0):.3f}\n")
                        f.write(f"  总奖励: {best_plan.get('total_reward', 0):.1f}\n")
                        f.write(f"  步数: {best_plan.get('step_count', 0)}\n")
                        f.write(f"  使用模型数: {result.get('model_count', 0)}\n")
                    else:
                        f.write("  推理失败\n")
                    
                    f.write("\n")
                
                # 写入汇总信息
                f.write("汇总统计\n")
                f.write("=" * 30 + "\n")
                for key, value in summary.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            print(f"📁 结果已保存到: {results_file}")
            
        except Exception as e:
            print(f"⚠️ 保存结果失败: {e}")

def start_ensemble_inference(config: Config, 
                           model_paths: List[str], 
                           scenarios: List[str] = None,
                           num_plans: int = 5,
                           top_models: int = 5) -> Dict[str, Any]:
    """
    启动集成推理的主函数
    
    Args:
        config: 配置对象
        model_paths: 模型文件路径列表
        scenarios: 场景列表
        num_plans: 每个场景生成的方案数量
        top_models: 使用的最优模型数量
        
    Returns:
        Dict: 推理结果
    """
    manager = EnsembleInferenceManager(config)
    return manager.run_ensemble_inference(
        model_paths=model_paths,
        scenarios=scenarios,
        num_plans=num_plans,
        top_models=top_models
    )