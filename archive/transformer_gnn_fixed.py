# -*- coding: utf-8 -*-
# 文件名: transformer_gnn_fixed.py
# 描述: 修复版TransformerGNN网络架构，解决特征提取、注意力机制和空间信息处理的关键风险点

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

# 导入修复版的组件
from networks_fixed import RobustFeatureExtractor, RelativePositionEncoder, GraphAttentionLayer

# 导入NoisyLinear相关功能
import sys
import os
from local_attention import LocalAttention, MultiScaleLocalAttention

# 添加temp_tests目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
temp_tests_dir = os.path.join(current_dir, 'temp_tests')
if os.path.exists(temp_tests_dir) and temp_tests_dir not in sys.path:
    sys.path.insert(0, temp_tests_dir)

try:
    from noisy_linear import NoisyLinear, replace_linear_with_noisy, reset_noise_in_module
except ImportError as e:
    print(f"警告：无法导入NoisyLinear: {e}")
    # 创建占位符类以避免错误
    class NoisyLinear(nn.Linear):
        pass
    def replace_linear_with_noisy(module, std_init=0.5):
        return module
    def reset_noise_in_module(module):
        pass


class FixedTransformerGNN(TorchModelV2, nn.Module):
    """
    修复版TransformerGNN网络架构 - 解决三个关键风险点
    
    核心修复：
    1. 鲁棒的特征提取：基于语义的特征分割，而非简单对半切分
    2. 真正的注意力机制：实现完整的图注意力计算
    3. 完整的空间信息处理：相对位置编码和结构感知
    
    设计理念：
    - 保持与RLlib的完全兼容性
    - 向后兼容现有的扁平观测格式
    - 支持图模式和扁平模式双输入
    - 真正的零样本迁移能力
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        初始化修复版TransformerGNN网络
        
        Args:
            obs_space: 观测空间
            action_space: 动作空间
            num_outputs: 输出维度
            model_config: 模型配置字典
            name: 模型名称
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # 从配置中获取参数
        self.embed_dim = model_config.get("embed_dim", 128)
        self.num_heads = model_config.get("num_heads", 8)
        self.num_layers = model_config.get("num_layers", 3)
        self.dropout = model_config.get("dropout", 0.1)
        self.use_position_encoding = model_config.get("use_position_encoding", True)
        self.use_noisy_linear = model_config.get("use_noisy_linear", True)
        self.noisy_std_init = model_config.get("noisy_std_init", 0.5)
        
        # 特征提取配置
        self.extraction_strategy = model_config.get("extraction_strategy", "semantic")
        
        # 局部注意力机制配置
        self.use_local_attention = model_config.get("use_local_attention", True)
        self.k_adaptive = model_config.get("k_adaptive", True)
        self.k_fixed = model_config.get("k_fixed", None)
        self.k_min = model_config.get("k_min", 4)
        self.k_max = model_config.get("k_max", 16)
        self.use_flash_attention = model_config.get("use_flash_attention", True)
        
        print(f"[FixedTransformerGNN] 初始化开始 - 特征提取策略: {self.extraction_strategy}")
        
        # 观测空间处理
        if hasattr(obs_space, 'original_space'):
            # 处理Dict观测空间
            self.obs_space_dict = obs_space.original_space
            self.is_dict_obs = True
            self.input_dim = None
        elif isinstance(obs_space, Dict) or hasattr(obs_space, 'spaces'):
            # 直接是Dict观测空间
            self.obs_space_dict = obs_space
            self.is_dict_obs = True
            self.input_dim = None
        elif hasattr(obs_space, 'shape'):
            # 处理Box观测空间
            self.obs_space_dict = None
            self.is_dict_obs = False
            self.input_dim = obs_space.shape[0]
        else:
            # 默认处理
            self.obs_space_dict = None
            self.is_dict_obs = False
            self.input_dim = 128  # 默认值
        
        # 构建特征提取配置
        self.feature_config = self._build_feature_config(model_config)
        
        # === 修复点1: 鲁棒的特征提取器 ===
        self.feature_extractor = RobustFeatureExtractor(self.feature_config)
        
        # 根据观测模式确定特征维度
        if self.is_dict_obs:
            # 图模式：从观测空间字典中获取精确的特征维度
            uav_features_dim = self.obs_space_dict['uav_features'].shape[-1]
            target_features_dim = self.obs_space_dict['target_features'].shape[-1]
            print(f"[FixedTransformerGNN] 图模式初始化 - UAV特征维度: {uav_features_dim}, 目标特征维度: {target_features_dim}")
        else:
            # 扁平模式：使用特征提取器计算维度
            uav_features_dim = self.feature_extractor._calculate_uav_feature_dim(self.feature_config)
            target_features_dim = self.feature_extractor._calculate_target_feature_dim(self.feature_config)
            print(f"[FixedTransformerGNN] 扁平模式初始化 - UAV特征维度: {uav_features_dim}, 目标特征维度: {target_features_dim}")
        
        # === 实体编码器架构设计 ===
        self.uav_encoder = self._build_robust_entity_encoder(uav_features_dim, self.embed_dim, "UAV")
        self.target_encoder = self._build_robust_entity_encoder(target_features_dim, self.embed_dim, "Target")
        
        # === 修复点3: 相对位置编码器 ===
        if self.use_position_encoding:
            self.position_encoder = RelativePositionEncoder(
                position_dim=2,  # x, y坐标
                embed_dim=self.embed_dim,
                max_distance=model_config.get('max_distance', 1000.0),
                num_distance_bins=model_config.get('num_distance_bins', 32),
                num_angle_bins=model_config.get('num_angle_bins', 16)
            )
        
        # === 修复点2: 真正的图注意力机制 ===
        self.graph_attention_layers = nn.ModuleList([
            GraphAttentionLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_position_encoding=self.use_position_encoding
            )
            for _ in range(self.num_layers)
        ])
        
        # 局部注意力机制（可选）
        if self.use_local_attention:
            self.local_attention = LocalAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                k_adaptive=self.k_adaptive,
                k_fixed=self.k_fixed,
                k_min=self.k_min,
                k_max=self.k_max,
                use_flash_attention=self.use_flash_attention
            )
            print(f"[FixedTransformerGNN] 局部注意力机制已启用")
        else:
            self.local_attention = None
            print(f"[FixedTransformerGNN] 局部注意力机制已禁用")
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, num_outputs)
        )
        
        # 值函数头
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
        # 将所有Linear层替换为NoisyLinear层（如果启用）
        if self.use_noisy_linear:
            self._replace_with_noisy_linear()
        
        # 存储最后的值函数输出
        self._last_value = None
        
        print(f"[FixedTransformerGNN] 初始化完成 - 嵌入维度: {self.embed_dim}, 注意力层数: {self.num_layers}")
    
    def _build_feature_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建特征提取配置
        
        Args:
            model_config: 模型配置
            
        Returns:
            特征提取配置字典
        """
        feature_config = {
            'extraction_strategy': self.extraction_strategy,
            'total_input_dim': self.input_dim,
            'embedding_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'max_distance': model_config.get('max_distance', 1000.0),
            
            # 环境相关配置
            'n_uavs': model_config.get('n_uavs', 1),
            'n_targets': model_config.get('n_targets', 1),
            'uav_features_per_entity': model_config.get('uav_features_per_entity', 8),
            'target_features_per_entity': model_config.get('target_features_per_entity', 7),
            
            # 比例策略配置
            'target_feature_ratio': model_config.get('target_feature_ratio', 0.6),
            
            # 固定维度策略配置
            'uav_feature_dim': model_config.get('uav_feature_dim', None),
            'target_feature_dim': model_config.get('target_feature_dim', None),
        }
        
        return feature_config
    
    def _build_robust_entity_encoder(self, input_dim: int, output_dim: int, entity_type: str) -> nn.Module:
        """
        构建鲁棒的实体编码器
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            entity_type: 实体类型（用于调试）
            
        Returns:
            实体编码器模块
        """
        encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # 添加额外的编码层以增强表达能力
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        print(f"[FixedTransformerGNN] {entity_type}编码器构建完成 - 输入维度: {input_dim}, 输出维度: {output_dim}")
        
        return encoder
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.zeros_(module.bias)
    
    def _replace_with_noisy_linear(self):
        """将所有Linear层替换为NoisyLinear层"""
        # 替换实体编码器中的Linear层
        self.uav_encoder = replace_linear_with_noisy(self.uav_encoder, self.noisy_std_init)
        self.target_encoder = replace_linear_with_noisy(self.target_encoder, self.noisy_std_init)
        
        # 替换位置编码器中的Linear层
        if self.use_position_encoding:
            self.position_encoder = replace_linear_with_noisy(
                self.position_encoder, self.noisy_std_init
            )
        
        # 替换输出层中的Linear层
        self.output_layer = replace_linear_with_noisy(
            self.output_layer, self.noisy_std_init
        )
        
        # 替换值函数头中的Linear层
        self.value_head = replace_linear_with_noisy(
            self.value_head, self.noisy_std_init
        )
        
        # 替换局部注意力中的Linear层
        if self.use_local_attention and self.local_attention is not None:
            self.local_attention = replace_linear_with_noisy(
                self.local_attention, self.noisy_std_init
            )
        
        # 替换图注意力层中的Linear层
        for i, attention_layer in enumerate(self.graph_attention_layers):
            self.graph_attention_layers[i] = replace_linear_with_noisy(
                attention_layer, self.noisy_std_init
            )
    
    def reset_noise(self):
        """重置所有NoisyLinear层的噪声"""
        if self.use_noisy_linear:
            reset_noise_in_module(self)
    
    def _extract_features_from_dict_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        从字典观测中提取特征 - 图模式输入解析和预处理
        
        Args:
            obs_dict: 字典形式的观测
            
        Returns:
            uav_features, target_features, relative_positions, additional_info
        """
        # 提取基本特征
        uav_features = obs_dict['uav_features']
        target_features = obs_dict['target_features']
        
        # 提取相对位置信息
        relative_positions = None
        if 'relative_positions' in obs_dict and self.use_position_encoding:
            relative_positions = obs_dict['relative_positions']
            print(f"[FixedTransformerGNN] 提取相对位置信息，形状: {relative_positions.shape}")
        
        # 提取额外信息（距离矩阵和掩码）
        additional_info = {}
        
        # 距离矩阵 - 用于局部注意力机制
        if 'distances' in obs_dict:
            additional_info['distances'] = obs_dict['distances']
            print(f"[FixedTransformerGNN] 提取距离矩阵，形状: {obs_dict['distances'].shape}")
        
        # 掩码信息 - 用于鲁棒性处理
        if 'masks' in obs_dict:
            additional_info['masks'] = obs_dict['masks']
        
        print(f"[FixedTransformerGNN] 图模式特征提取完成 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        
        return uav_features, target_features, relative_positions, additional_info
    
    def _extract_features_from_flat_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        从扁平观测中提取特征 - 使用修复版特征提取器
        
        Args:
            obs: 扁平观测张量 [batch_size, input_dim]
            
        Returns:
            uav_features, target_features, relative_positions, additional_info
        """
        batch_size = obs.shape[0]
        
        # === 修复点1: 使用鲁棒的特征提取器 ===
        uav_features, target_features, additional_features = self.feature_extractor.extract_features(obs)
        
        # 为图注意力添加序列维度
        uav_features = uav_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        target_features = target_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        print(f"[FixedTransformerGNN] 扁平模式特征提取完成 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        
        # 尝试从特征中推断相对位置
        relative_positions = None
        if self.use_position_encoding:
            relative_positions = self._compute_relative_positions_from_features(
                uav_features, target_features
            )
            if relative_positions is not None:
                print(f"[FixedTransformerGNN] 从扁平特征推断相对位置，形状: {relative_positions.shape}")
        
        # 生成默认的额外信息
        additional_info = {
            'masks': {
                'uav_mask': torch.ones(batch_size, 1, dtype=torch.bool, device=obs.device),
                'target_mask': torch.ones(batch_size, 1, dtype=torch.bool, device=obs.device)
            },
            'additional_features': additional_features
        }
        
        return uav_features, target_features, relative_positions, additional_info
    
    def _compute_relative_positions_from_features(
        self,
        uav_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        从特征中计算相对位置（改进版）
        
        Args:
            uav_features: UAV特征张量 [batch_size, num_uavs, feature_dim]
            target_features: 目标特征张量 [batch_size, num_targets, feature_dim]
            
        Returns:
            相对位置张量 [batch_size, num_pairs, 2] 或 None
        """
        # 根据特征提取策略确定位置信息的位置
        if self.extraction_strategy == 'semantic':
            # 语义策略：UAV位置在特征的前2个维度，目标位置也在前2个维度
            if uav_features.shape[-1] >= 2 and target_features.shape[-1] >= 2:
                uav_positions = uav_features[..., :2]  # [batch_size, num_uavs, 2]
                target_positions = target_features[..., :2]  # [batch_size, num_targets, 2]
                
                # 计算相对位置
                batch_size, num_uavs, _ = uav_positions.shape
                _, num_targets, _ = target_positions.shape
                
                # 扩展维度进行广播
                uav_pos_expanded = uav_positions.unsqueeze(2)  # [batch_size, num_uavs, 1, 2]
                target_pos_expanded = target_positions.unsqueeze(1)  # [batch_size, 1, num_targets, 2]
                
                # 计算相对位置 (target_pos - uav_pos)
                relative_positions = target_pos_expanded - uav_pos_expanded  # [batch_size, num_uavs, num_targets, 2]
                
                # 重塑为 [batch_size, num_pairs, 2]
                relative_positions = relative_positions.view(batch_size, num_uavs * num_targets, 2)
                
                return relative_positions
        
        # 如果无法提取位置信息，返回None
        return None
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        修复版前向传播 - 实现真正的图注意力计算
        
        Args:
            input_dict: 输入字典，包含观测
            state: RNN状态（未使用）
            seq_lens: 序列长度（未使用）
            
        Returns:
            logits: 动作logits
            state: 更新后的状态
        """
        obs = input_dict["obs"]
        
        # 在训练模式下重置噪声（每次前向传播时）
        if self.training and self.use_noisy_linear:
            self.reset_noise()
        
        # === 修复点1: 鲁棒的特征提取 ===
        if self.is_dict_obs:
            uav_features, target_features, relative_positions, additional_info = self._extract_features_from_dict_obs(obs)
        else:
            uav_features, target_features, relative_positions, additional_info = self._extract_features_from_flat_obs(obs)
        
        # 获取批次大小和实体数量信息
        batch_size = uav_features.shape[0]
        num_uavs = uav_features.shape[1]
        num_targets = target_features.shape[1]
        
        print(f"[FixedTransformerGNN] 前向传播开始 - 批次: {batch_size}, UAV数量: {num_uavs}, 目标数量: {num_targets}")
        
        # === 实体特征编码 ===
        uav_embeddings = self.uav_encoder(uav_features)  # [batch_size, num_uavs, embed_dim]
        target_embeddings = self.target_encoder(target_features)  # [batch_size, num_targets, embed_dim]
        
        print(f"[FixedTransformerGNN] 实体编码完成 - UAV嵌入: {uav_embeddings.shape}, 目标嵌入: {target_embeddings.shape}")
        
        # 应用掩码处理（鲁棒性机制）
        if 'masks' in additional_info:
            masks = additional_info['masks']
            
            # 应用UAV掩码
            if 'uav_mask' in masks:
                uav_mask = masks['uav_mask'].unsqueeze(-1)  # [batch_size, num_uavs, 1]
                uav_embeddings = uav_embeddings * uav_mask.float()
                print(f"[FixedTransformerGNN] 应用UAV掩码")
            
            # 应用目标掩码
            if 'target_mask' in masks:
                target_mask = masks['target_mask'].unsqueeze(-1)  # [batch_size, num_targets, 1]
                target_embeddings = target_embeddings * target_mask.float()
                print(f"[FixedTransformerGNN] 应用目标掩码")
        
        # === 修复点3: 相对位置编码处理 ===
        position_embeddings = None
        if self.use_position_encoding and relative_positions is not None:
            position_embeddings = self.position_encoder(relative_positions)
            print(f"[FixedTransformerGNN] 位置编码生成完成，形状: {position_embeddings.shape}")
        
        # === 局部注意力机制处理（可选） ===
        if self.use_local_attention and self.local_attention is not None and 'distances' in additional_info:
            distances = additional_info['distances']
            print(f"[FixedTransformerGNN] 应用局部注意力机制")
            
            # 应用局部注意力到UAV嵌入
            uav_attention_output = self.local_attention(
                uav_embeddings, target_embeddings, distances, additional_info.get('masks')
            )
            
            # 将注意力输出与原始嵌入结合（残差连接）
            uav_embeddings = uav_embeddings + uav_attention_output
            print(f"[FixedTransformerGNN] 局部注意力应用完成")
        
        # 合并实体嵌入
        entity_embeddings = torch.cat([uav_embeddings, target_embeddings], dim=1)  # [batch_size, num_entities, embed_dim]
        print(f"[FixedTransformerGNN] 实体嵌入合并完成，总实体数: {entity_embeddings.shape[1]}")
        
        # === 修复点2: 真正的图注意力计算 ===
        node_embeddings = entity_embeddings
        for i, attention_layer in enumerate(self.graph_attention_layers):
            node_embeddings = attention_layer(node_embeddings, position_embeddings)
            print(f"[FixedTransformerGNN] 图注意力层 {i+1} 完成")
        
        print(f"[FixedTransformerGNN] 所有图注意力层完成，输出形状: {node_embeddings.shape}")
        
        # === 图级别聚合 - 使用注意力池化 ===
        graph_embedding = self._attention_pooling(node_embeddings)  # [batch_size, embed_dim]
        print(f"[FixedTransformerGNN] 注意力池化完成，形状: {graph_embedding.shape}")
        
        # 生成动作logits
        logits = self.output_layer(graph_embedding)
        print(f"[FixedTransformerGNN] 动作logits生成，形状: {logits.shape}")
        
        # 计算值函数
        self._last_value = self.value_head(graph_embedding).squeeze(-1)
        print(f"[FixedTransformerGNN] 值函数计算完成，形状: {self._last_value.shape}")
        
        return logits, state
    
    def _attention_pooling(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        注意力池化 - 替代简单的平均池化
        
        Args:
            node_embeddings: 节点嵌入 [batch_size, num_nodes, embed_dim]
            
        Returns:
            图级别嵌入 [batch_size, embed_dim]
        """
        batch_size, num_nodes, embed_dim = node_embeddings.shape
        
        # 计算注意力权重
        attention_scores = torch.sum(node_embeddings, dim=-1)  # [batch_size, num_nodes]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_nodes]
        
        # 加权聚合
        graph_embedding = torch.sum(
            node_embeddings * attention_weights.unsqueeze(-1), dim=1
        )  # [batch_size, embed_dim]
        
        return graph_embedding
    
    @override(TorchModelV2)
    def value_function(self):
        """返回值函数输出"""
        return self._last_value


def create_fixed_transformer_gnn_model(obs_space, action_space, num_outputs, model_config, name="FixedTransformerGNN"):
    """
    创建修复版TransformerGNN模型的工厂函数
    
    Args:
        obs_space: 观测空间
        action_space: 动作空间
        num_outputs: 输出维度
        model_config: 模型配置
        name: 模型名称
        
    Returns:
        修复版TransformerGNN模型实例
    """
    return FixedTransformerGNN(obs_space, action_space, num_outputs, model_config, name)
