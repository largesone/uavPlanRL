# -*- coding: utf-8 -*-
# 文件名: model_manager.py
# 描述: 模型管理器，用于保存和管理训练好的模型

import os
import torch
import glob
from typing import List, Optional


class ModelManager:
    """模型管理器 - 管理模型的保存、加载和版本控制"""
    
    def __init__(self, save_dir: str, max_models: int = 5):
        """
        初始化模型管理器
        
        Args:
            save_dir (str): 模型保存目录
            max_models (int): 最大保存模型数量
        """
        self.save_dir = save_dir
        self.max_models = max_models
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"模型管理器初始化完成，保存目录: {save_dir}")
    
    def save_model(self, model, episode: int, reward: float, suffix: str = ""):
        """
        保存模型
        
        Args:
            model: 要保存的模型
            episode (int): 训练轮次
            reward (float): 模型性能（奖励）
            suffix (str): 文件名后缀
        """
        # 生成文件名
        if suffix:
            filename = f"model_ep{episode}_{suffix}_reward{reward:.2f}.pth"
        else:
            filename = f"model_ep{episode}_reward{reward:.2f}.pth"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # 保存模型
        try:
            torch.save(model.state_dict(), filepath)
            print(f"模型已保存: {filepath}")
            
            # 清理旧模型
            self._cleanup_old_models()
            
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def load_model(self, model, model_path: str):
        """
        加载模型
        
        Args:
            model: 要加载到的模型对象
            model_path (str): 模型文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            if os.path.exists(model_path):
                try:
                    # 首先尝试使用weights_only=False加载
                    model_state = torch.load(model_path, weights_only=False)
                except TypeError as type_error:
                    # 处理PyTorch 2.6版本中weights_only参数的变化
                    if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                        print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                        model_state = torch.load(model_path)
                    else:
                        raise
                except Exception as load_error:
                    # 如果失败，尝试使用weights_only=True加载
                    print(f"使用weights_only=False加载失败，尝试使用weights_only=True: {load_error}")
                    try:
                        # 添加安全全局变量以支持numpy.core.multiarray.scalar
                        import torch.serialization
                        import numpy.core.multiarray
                        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                        model_state = torch.load(model_path, weights_only=True)
                    except TypeError as type_error:
                        # 处理PyTorch 2.6版本中weights_only参数的变化
                        if "got an unexpected keyword argument 'weights_only'" in str(type_error):
                            print(f"当前PyTorch版本不支持weights_only参数，使用默认参数加载")
                            model_state = torch.load(model_path)
                        else:
                            raise
                    except Exception as safe_load_error:
                        print(f"使用weights_only=True加载也失败，尝试使用map_location: {safe_load_error}")
                        # 最后尝试使用map_location参数
                        model_state = torch.load(model_path, map_location='cpu')
                
                # 处理不同的模型保存格式
                if isinstance(model_state, dict):
                    if 'model_state_dict' in model_state:
                        # 如果是完整的模型信息字典
                        model.load_state_dict(model_state['model_state_dict'])
                    elif 'state_dict' in model_state:
                        # 如果是包含state_dict的字典
                        model.load_state_dict(model_state['state_dict'])
                    else:
                        # 直接是state_dict
                        model.load_state_dict(model_state)
                else:
                    # 如果不是字典，可能是直接的state_dict
                    model.load_state_dict(model_state)
                
                print(f"模型加载成功: {model_path}")
                return True
            else:
                print(f"模型文件不存在: {model_path}")
                return False
        except Exception as e:
            print(f"加载模型失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            print(f"尝试检查模型格式...")
            
            # 尝试检查模型文件的内容
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                print(f"模型文件结构: {type(checkpoint)}")
                if isinstance(checkpoint, dict):
                    print(f"模型文件键: {list(checkpoint.keys())}")
            except Exception as check_error:
                print(f"无法检查模型文件: {check_error}")
            
            return False
    
    def get_latest_model(self) -> Optional[str]:
        """
        获取最新的模型文件路径
        
        Returns:
            Optional[str]: 最新模型路径，如果没有则返回None
        """
        model_files = glob.glob(os.path.join(self.save_dir, "*.pth"))
        
        if not model_files:
            return None
        
        # 按修改时间排序，返回最新的
        latest_model = max(model_files, key=os.path.getmtime)
        return latest_model
    
    def get_best_model(self) -> Optional[str]:
        """
        获取性能最好的模型文件路径（基于文件名中的奖励值）
        
        Returns:
            Optional[str]: 最佳模型路径，如果没有则返回None
        """
        model_files = glob.glob(os.path.join(self.save_dir, "*.pth"))
        
        if not model_files:
            return None
        
        best_model = None
        best_reward = float('-inf')
        
        for model_file in model_files:
            try:
                # 从文件名中提取奖励值
                filename = os.path.basename(model_file)
                if "reward" in filename:
                    reward_str = filename.split("reward")[1].split(".pth")[0]
                    reward = float(reward_str)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_model = model_file
            except:
                continue
        
        return best_model
    
    def list_models(self) -> List[str]:
        """
        列出所有保存的模型
        
        Returns:
            List[str]: 模型文件路径列表
        """
        model_files = glob.glob(os.path.join(self.save_dir, "*.pth"))
        return sorted(model_files, key=os.path.getmtime, reverse=True)
    
    def _cleanup_old_models(self):
        """清理旧模型，保持最大数量限制"""
        model_files = self.list_models()
        
        if len(model_files) > self.max_models:
            # 删除最旧的模型
            models_to_delete = model_files[self.max_models:]
            
            for model_file in models_to_delete:
                try:
                    os.remove(model_file)
                    print(f"删除旧模型: {os.path.basename(model_file)}")
                except Exception as e:
                    print(f"删除模型失败 {model_file}: {e}")
    
    def get_model_info(self, model_path: str) -> dict:
        """
        获取模型信息
        
        Args:
            model_path (str): 模型文件路径
            
        Returns:
            dict: 模型信息字典
        """
        if not os.path.exists(model_path):
            return {}
        
        filename = os.path.basename(model_path)
        info = {
            'path': model_path,
            'filename': filename,
            'size': os.path.getsize(model_path),
            'modified_time': os.path.getmtime(model_path)
        }
        
        # 尝试从文件名中提取信息
        try:
            if "ep" in filename:
                episode_str = filename.split("ep")[1].split("_")[0]
                info['episode'] = int(episode_str)
            
            if "reward" in filename:
                reward_str = filename.split("reward")[1].split(".pth")[0]
                info['reward'] = float(reward_str)
        except:
            pass
        
        return info