# -*- coding: utf-8 -*-
# 文件名: networks_fixed.py
# 描述: 修复版神经网络模块，解决特征提取、注意力机制和空间信息处理的关键风险点

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Dict, Any, Tuple

class RobustFeatureExtractor(nn.Module):
    """
    鲁棒的特征提取器 - 解决风险点1：简化的特征提取
    
    核心改进：
    1. 基于语义的特征分割，而非简单的对半切分
    2. 自适应特征维度检测
    3. 配置驱动的特征映射
    4. 向后兼容性保证
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(RobustFeatureExtractor, self).__init__()
        
        self.config = config
        
        # 从配置中获取特征维度信息
        self.uav_feature_dim = config.get('uav_feature_dim', None)
        self.target_feature_dim = config.get('target_feature_dim', None)
        self.total_input_dim = config.get('total_input_dim', None)
        
        # 特征分割策略
        self.extraction_strategy = config.get('extraction_strategy', 'semantic')  # 'semantic', 'ratio', 'fixed'
        
        # 语义特征映射（基于环境状态结构）
        self.feature_mapping = self._build_feature_mapping(config)
        
        print(f"[RobustFeatureExtractor] 初始化完成 - 策略: {self.extraction_strategy}")
        print(f"[RobustFeatureExtractor] UAV特征维度: {self.uav_feature_dim}, 目标特征维度: {self.target_feature_dim}")
    
    def _build_feature_mapping(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建基于环境状态结构的特征映射
        
        Args:
            config: 配置字典，包含环境状态结构信息
            
        Returns:
            特征映射字典
        """
        # 默认特征映射（基于environment.py中的_get_flat_state结构）
        default_mapping = {
            'target_features': {
                'start_idx': 0,
                'features_per_entity': 7,  # position(2) + resources(2) + value(1) + remaining_resources(2)
                'count': config.get('n_targets', 1)
            },
            'uav_features': {
                'start_idx': None,  # 动态计算
                'features_per_entity': 8,  # position(2) + heading(1) + resources(2) + max_distance(1) + velocity_range(2)
                'count': config.get('n_uavs', 1)
            },
            'collaboration_features': {
                'start_idx': None,  # 动态计算
                'total_dim': config.get('n_targets', 1) * config.get('n_uavs', 1)
            },
            'global_features': {
                'start_idx': None,  # 动态计算
                'total_dim': 10
            }
        }
        
        # 计算动态起始索引
        target_total_dim = default_mapping['target_features']['features_per_entity'] * default_mapping['target_features']['count']
        uav_total_dim = default_mapping['uav_features']['features_per_entity'] * default_mapping['uav_features']['count']
        
        default_mapping['uav_features']['start_idx'] = target_total_dim
        default_mapping['collaboration_features']['start_idx'] = target_total_dim + uav_total_dim
        default_mapping['global_features']['start_idx'] = target_total_dim + uav_total_dim + default_mapping['collaboration_features']['total_dim']
        
        # 使用配置中的映射（如果提供）或默认映射
        return config.get('feature_mapping', default_mapping)
    
    def extract_features(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        鲁棒的特征提取方法
        
        Args:
            state: 输入状态张量 [batch_size, state_dim]
            
        Returns:
            uav_features: UAV特征张量 [batch_size, uav_feature_dim]
            target_features: 目标特征张量 [batch_size, target_feature_dim]
            additional_features: 额外特征字典（协同信息、全局信息等）
        """
        batch_size, state_dim = state.shape
        
        if self.extraction_strategy == 'semantic':
            return self._extract_semantic_features(state)
        elif self.extraction_strategy == 'ratio':
            return self._extract_ratio_features(state)
        elif self.extraction_strategy == 'fixed':
            return self._extract_fixed_features(state)
        else:
            # 回退到语义提取
            return self._extract_semantic_features(state)
    
    def _extract_semantic_features(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        基于语义的特征提取 - 根据环境状态结构精确分割
        
        Args:
            state: 输入状态张量 [batch_size, state_dim]
            
        Returns:
            UAV特征、目标特征和额外特征
        """
        batch_size, state_dim = state.shape
        
        # 提取目标特征
        target_mapping = self.feature_mapping['target_features']
        target_start = target_mapping['start_idx']
        target_end = target_start + target_mapping['features_per_entity'] * target_mapping['count']
        target_features = state[:, target_start:target_end]
        
        # 提取UAV特征
        uav_mapping = self.feature_mapping['uav_features']
        uav_start = uav_mapping['start_idx']
        uav_end = uav_start + uav_mapping['features_per_entity'] * uav_mapping['count']
        uav_features = state[:, uav_start:uav_end]
        
        # 提取额外特征
        additional_features = {}
        
        # 协同特征
        if 'collaboration_features' in self.feature_mapping:
            collab_mapping = self.feature_mapping['collaboration_features']
            collab_start = collab_mapping['start_idx']
            collab_end = collab_start + collab_mapping['total_dim']
            additional_features['collaboration'] = state[:, collab_start:collab_end]
        
        # 全局特征
        if 'global_features' in self.feature_mapping:
            global_mapping = self.feature_mapping['global_features']
            global_start = global_mapping['start_idx']
            global_end = global_start + global_mapping['total_dim']
            additional_features['global'] = state[:, global_start:global_end]
        
        print(f"[RobustFeatureExtractor] 语义提取完成 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        
        return uav_features, target_features, additional_features
    
    def _extract_ratio_features(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        基于比例的特征提取 - 改进的对半切分，支持不等比例
        
        Args:
            state: 输入状态张量 [batch_size, state_dim]
            
        Returns:
            UAV特征、目标特征和额外特征
        """
        batch_size, state_dim = state.shape
        
        # 使用配置的比例或默认6:4比例（目标特征通常更复杂）
        target_ratio = self.config.get('target_feature_ratio', 0.6)
        
        target_dim = int(state_dim * target_ratio)
        uav_dim = state_dim - target_dim
        
        target_features = state[:, :target_dim]
        uav_features = state[:, target_dim:]
        
        additional_features = {}
        
        print(f"[RobustFeatureExtractor] 比例提取完成 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        
        return uav_features, target_features, additional_features
    
    def _extract_fixed_features(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        基于固定维度的特征提取 - 使用预定义的特征维度
        
        Args:
            state: 输入状态张量 [batch_size, state_dim]
            
        Returns:
            UAV特征、目标特征和额外特征
        """
        batch_size, state_dim = state.shape
        
        if self.uav_feature_dim is None or self.target_feature_dim is None:
            raise ValueError("固定维度提取需要预定义uav_feature_dim和target_feature_dim")
        
        # 确保维度不超过状态总维度
        total_required = self.uav_feature_dim + self.target_feature_dim
        if total_required > state_dim:
            # 按比例缩放
            scale_factor = state_dim / total_required
            uav_dim = int(self.uav_feature_dim * scale_factor)
            target_dim = state_dim - uav_dim
        else:
            uav_dim = self.uav_feature_dim
            target_dim = self.target_feature_dim
        
        target_features = state[:, :target_dim]
        uav_features = state[:, target_dim:target_dim + uav_dim]
        
        # 剩余部分作为额外特征
        additional_features = {}
        if target_dim + uav_dim < state_dim:
            additional_features['remaining'] = state[:, target_dim + uav_dim:]
        
        print(f"[RobustFeatureExtractor] 固定维度提取完成 - UAV: {uav_features.shape}, 目标: {target_features.shape}")
        
        return uav_features, target_features, additional_features
    
    def _calculate_uav_feature_dim(self, config: Dict[str, Any]) -> int:
        """计算UAV特征维度"""
        if config.get('extraction_strategy') == 'semantic':
            return config.get('uav_features_per_entity', 8) * config.get('n_uavs', 1)
        elif config.get('extraction_strategy') == 'fixed':
            return config.get('uav_feature_dim', 64)
        else:
            # 比例策略
            total_dim = config.get('total_input_dim', 128)
            target_ratio = config.get('target_feature_ratio', 0.6)
            return int(total_dim * (1 - target_ratio))
    
    def _calculate_target_feature_dim(self, config: Dict[str, Any]) -> int:
        """计算目标特征维度"""
        if config.get('extraction_strategy') == 'semantic':
            return config.get('target_features_per_entity', 7) * config.get('n_targets', 1)
        elif config.get('extraction_strategy') == 'fixed':
            return config.get('target_feature_dim', 64)
        else:
            # 比例策略
            total_dim = config.get('total_input_dim', 128)
            target_ratio = config.get('target_feature_ratio', 0.6)
            return int(total_dim * target_ratio)


class TrueGraphAttentionNetwork(nn.Module):
    """
    真正的图注意力网络 - 解决风险点2：伪注意力机制
    
    核心改进：
    1. 实现真正的多头图注意力机制
    2. 支持UAV-目标和目标-UAV双向注意力
    3. 集成相对位置编码
    4. 可配置的注意力策略
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        config: Dict[str, Any]
    ):
        super(TrueGraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.config = config
        
        # 嵌入维度
        self.embedding_dim = config.get('embedding_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # 特征提取器
        self.feature_extractor = RobustFeatureExtractor(config)
        
        # 计算实际的UAV和目标特征维度
        self.uav_feature_dim = self._calculate_uav_feature_dim(config)
        self.target_feature_dim = self._calculate_target_feature_dim(config)
        
        # 实体编码器
        self.uav_encoder = self._build_entity_encoder(self.uav_feature_dim, self.embedding_dim)
        self.target_encoder = self._build_entity_encoder(self.target_feature_dim, self.embedding_dim)
        
        # 相对位置编码器 - 解决风险点3
        self.position_encoder = RelativePositionEncoder(
            position_dim=2,
            embed_dim=self.embedding_dim,
            max_distance=config.get('max_distance', 1000.0)
        )
        
        # 真正的图注意力层
        self.graph_attention_layers = nn.ModuleList([
            GraphAttentionLayer(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_position_encoding=True
            )
            for _ in range(config.get('num_attention_layers', 2))
        ])
        
        # 输出层
        self.output_layer = self._build_output_layer(self.embedding_dim, output_dim)
        
        self._init_weights()
        
        print(f"[TrueGraphAttentionNetwork] 初始化完成 - 嵌入维度: {self.embedding_dim}, 注意力头数: {self.num_heads}")
    
    def _calculate_uav_feature_dim(self, config: Dict[str, Any]) -> int:
        """计算UAV特征维度"""
        if config.get('extraction_strategy') == 'semantic':
            return config.get('uav_features_per_entity', 8) * config.get('n_uavs', 1)
        elif config.get('extraction_strategy') == 'fixed':
            return config.get('uav_feature_dim', 64)
        else:
            # 比例策略
            total_dim = config.get('total_input_dim', 128)
            target_ratio = config.get('target_feature_ratio', 0.6)
            return int(total_dim * (1 - target_ratio))
    
    def _calculate_target_feature_dim(self, config: Dict[str, Any]) -> int:
        """计算目标特征维度"""
        if config.get('extraction_strategy') == 'semantic':
            return config.get('target_features_per_entity', 7) * config.get('n_targets', 1)
        elif config.get('extraction_strategy') == 'fixed':
            return config.get('target_feature_dim', 64)
        else:
            # 比例策略
            total_dim = config.get('total_input_dim', 128)
            target_ratio = config.get('target_feature_ratio', 0.6)
            return int(total_dim * target_ratio)
    
    def _build_entity_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建实体编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def _build_output_layer(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建输出层"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 真正的图注意力计算
        
        Args:
            x: 输入状态张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # 1. 鲁棒的特征提取
        uav_features, target_features, additional_features = self.feature_extractor.extract_features(x)
        
        # 2. 实体编码
        uav_embeddings = self.uav_encoder(uav_features)  # [batch_size, embedding_dim]
        target_embeddings = self.target_encoder(target_features)  # [batch_size, embedding_dim]
        
        # 3. 构建图结构 - 将实体嵌入组织为图节点
        # 假设单UAV单目标场景，扩展维度以支持图注意力
        uav_embeddings = uav_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        target_embeddings = target_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # 合并所有节点
        all_nodes = torch.cat([uav_embeddings, target_embeddings], dim=1)  # [batch_size, 2, embedding_dim]
        
        # 4. 计算相对位置编码
        position_embeddings = self._compute_position_embeddings(uav_features, target_features, batch_size)
        
        # 5. 应用图注意力层
        node_embeddings = all_nodes
        for attention_layer in self.graph_attention_layers:
            node_embeddings = attention_layer(node_embeddings, position_embeddings)
        
        # 6. 图级别聚合 - 使用注意力池化而非简单平均
        graph_embedding = self._attention_pooling(node_embeddings)  # [batch_size, embedding_dim]
        
        # 7. 输出层
        output = self.output_layer(graph_embedding)
        
        return output
    
    def _compute_position_embeddings(
        self,
        uav_features: torch.Tensor,
        target_features: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        计算相对位置编码
        
        Args:
            uav_features: UAV特征
            target_features: 目标特征
            batch_size: 批次大小
            
        Returns:
            位置编码张量 [batch_size, num_pairs, embedding_dim]
        """
        # 从特征中提取位置信息（假设前两个维度是位置）
        if uav_features.shape[1] >= 2 and target_features.shape[1] >= 2:
            uav_pos = uav_features[:, :2]  # [batch_size, 2]
            target_pos = target_features[:, :2]  # [batch_size, 2]
            
            # 计算相对位置
            relative_pos = target_pos - uav_pos  # [batch_size, 2]
            
            # 生成位置编码
            position_embeddings = self.position_encoder(relative_pos.unsqueeze(1))  # [batch_size, 1, embedding_dim]
            
            return position_embeddings
        else:
            # 如果无法提取位置信息，返回零编码
            return torch.zeros(batch_size, 1, self.embedding_dim, device=uav_features.device)
    
    def _attention_pooling(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        注意力池化 - 替代简单的平均池化
        
        Args:
            node_embeddings: 节点嵌入 [batch_size, num_nodes, embedding_dim]
            
        Returns:
            图级别嵌入 [batch_size, embedding_dim]
        """
        batch_size, num_nodes, embedding_dim = node_embeddings.shape
        
        # 计算注意力权重
        attention_weights = torch.softmax(
            torch.sum(node_embeddings, dim=-1), dim=-1
        )  # [batch_size, num_nodes]
        
        # 加权聚合
        graph_embedding = torch.sum(
            node_embeddings * attention_weights.unsqueeze(-1), dim=1
        )  # [batch_size, embedding_dim]
        
        return graph_embedding


class RelativePositionEncoder(nn.Module):
    """
    相对位置编码器 - 解决风险点3：缺乏空间/结构信息
    
    核心特性：
    1. 基于相对距离和角度的位置编码
    2. 可学习的位置嵌入
    3. 支持不同尺度的空间关系
    """
    
    def __init__(
        self,
        position_dim: int = 2,
        embed_dim: int = 64,
        max_distance: float = 1000.0,
        num_distance_bins: int = 32,
        num_angle_bins: int = 16
    ):
        super(RelativePositionEncoder, self).__init__()
        
        self.position_dim = position_dim
        self.embed_dim = embed_dim
        self.max_distance = max_distance
        self.num_distance_bins = num_distance_bins
        self.num_angle_bins = num_angle_bins
        
        # 距离编码
        self.distance_embedding = nn.Embedding(num_distance_bins, embed_dim // 2)
        
        # 角度编码
        self.angle_embedding = nn.Embedding(num_angle_bins, embed_dim // 2)
        
        # 位置MLP
        self.position_mlp = nn.Sequential(
            nn.Linear(position_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2)
        )
        
        # 融合层
        self.fusion_layer = nn.Linear(embed_dim + embed_dim // 2, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, relative_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            relative_positions: 相对位置张量 [batch_size, num_pairs, position_dim] 或 [batch_size, num_uavs, num_targets, position_dim]
            
        Returns:
            位置编码张量 [batch_size, num_pairs, embed_dim]
        """
        # 处理不同维度的输入
        if len(relative_positions.shape) == 4:
            # 4D输入: [batch_size, num_uavs, num_targets, position_dim]
            batch_size, num_uavs, num_targets, position_dim = relative_positions.shape
            # 重塑为3D: [batch_size, num_pairs, position_dim]
            relative_positions = relative_positions.view(batch_size, num_uavs * num_targets, position_dim)
            num_pairs = num_uavs * num_targets
        elif len(relative_positions.shape) == 3:
            # 3D输入: [batch_size, num_pairs, position_dim]
            batch_size, num_pairs, position_dim = relative_positions.shape
        else:
            raise ValueError(f"不支持的相对位置张量维度: {relative_positions.shape}")
        
        # 确保position_dim正确
        if position_dim != self.position_dim:
            raise ValueError(f"位置维度不匹配: 期望{self.position_dim}, 实际{position_dim}")
        
        # 计算距离
        distances = torch.norm(relative_positions, dim=-1)  # [batch_size, num_pairs]
        
        # 计算角度
        angles = torch.atan2(relative_positions[..., 1], relative_positions[..., 0])  # [batch_size, num_pairs]
        
        # 距离分箱
        distance_bins = torch.clamp(
            (distances / self.max_distance * self.num_distance_bins).long(),
            0, self.num_distance_bins - 1
        )
        
        # 角度分箱
        angle_bins = torch.clamp(
            ((angles + math.pi) / (2 * math.pi) * self.num_angle_bins).long(),
            0, self.num_angle_bins - 1
        )
        
        # 获取嵌入
        distance_emb = self.distance_embedding(distance_bins)  # [batch_size, num_pairs, embed_dim//2]
        angle_emb = self.angle_embedding(angle_bins)  # [batch_size, num_pairs, embed_dim//2]
        position_emb = self.position_mlp(relative_positions)  # [batch_size, num_pairs, embed_dim//2]
        
        # 融合所有编码
        combined_emb = torch.cat([distance_emb, angle_emb, position_emb], dim=-1)
        final_emb = self.fusion_layer(combined_emb)
        
        return final_emb


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 - 实现真正的图注意力机制
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_position_encoding: bool = True
    ):
        super(GraphAttentionLayer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_position_encoding = use_position_encoding
        
        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_embeddings: 节点嵌入 [batch_size, num_nodes, embed_dim]
            position_embeddings: 位置编码 [batch_size, num_pairs, embed_dim]
            
        Returns:
            更新后的节点嵌入 [batch_size, num_nodes, embed_dim]
        """
        # 添加位置编码（如果提供）
        if self.use_position_encoding and position_embeddings is not None:
            # 简化处理：将位置编码加到节点嵌入上
            if position_embeddings.shape[1] == 1:
                # 广播位置编码到所有节点
                node_embeddings = node_embeddings + position_embeddings.expand(-1, node_embeddings.shape[1], -1)
        
        # 多头自注意力
        residual = node_embeddings
        node_embeddings = self.norm1(node_embeddings)
        
        attn_output, _ = self.multihead_attention(
            query=node_embeddings,
            key=node_embeddings,
            value=node_embeddings
        )
        
        node_embeddings = residual + self.dropout_layer(attn_output)
        
        # 前馈网络
        residual = node_embeddings
        node_embeddings = self.norm2(node_embeddings)
        ff_output = self.feed_forward(node_embeddings)
        node_embeddings = residual + self.dropout_layer(ff_output)
        
        return node_embeddings


def create_fixed_network(
    network_type: str,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    创建修复版网络的工厂函数
    
    Args:
        network_type: 网络类型
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        config: 配置字典
        
    Returns:
        修复版网络模型实例
    """
    if config is None:
        config = {}
    
    # 设置默认配置
    default_config = {
        'total_input_dim': input_dim,
        'extraction_strategy': 'semantic',
        'embedding_dim': 128,
        'num_heads': 8,
        'dropout': 0.1,
        'num_attention_layers': 2,
        'n_uavs': 1,
        'n_targets': 1,
        'uav_features_per_entity': 8,
        'target_features_per_entity': 7,
        'max_distance': 1000.0
    }
    
    # 合并配置
    final_config = {**default_config, **config}
    
    if network_type == "TrueGraphAttentionNetwork":
        return TrueGraphAttentionNetwork(input_dim, hidden_dims, output_dim, final_config)
    else:
        raise ValueError(f"不支持的网络类型: {network_type}")
