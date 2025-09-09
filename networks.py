# -*- coding: utf-8 -*-
# 文件名: networks.py
# 描述: 统一的神经网络模块，包含所有网络结构定义

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any
import math

class LowRankSelfAttention(nn.Module):
    """
    简化的低秩近似自注意力机制
    
    核心思想：使用线性投影减少注意力计算复杂度
    """
    
    def __init__(self, d_model: int, low_rank_dim: int, nhead: int, dropout: float = 0.1):
        super(LowRankSelfAttention, self).__init__()
        
        self.d_model = d_model
        self.low_rank_dim = low_rank_dim
        self.nhead = nhead
        
        # 简化的线性投影
        self.query_proj = nn.Linear(d_model, low_rank_dim, bias=False)
        self.key_proj = nn.Linear(d_model, low_rank_dim, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(low_rank_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        
        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # 确保输入在正确的设备上
        device = x.device
        x = x.to(device)
        
        # 1. 计算Q, K, V
        q = self.query_proj(x)  # [batch_size, seq_len, low_rank_dim]
        k = self.key_proj(x)    # [batch_size, seq_len, low_rank_dim]
        v = self.value_proj(x)  # [batch_size, seq_len, d_model]
        
        # 2. 计算注意力分数（在低维空间）
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, seq_len, seq_len]
        
        # 3. 应用掩码
        if mask is not None:
            # 确保掩码在正确的设备上
            mask = mask.to(device)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, seq_len, d_model]
        
        # 6. 输出投影
        output = self.out_proj(attn_output)
        
        return output


class LowRankCrossAttention(nn.Module):
    """
    简化的低秩近似交叉注意力机制
    
    用于UAV-目标间的交互，通过低秩投影减少计算复杂度
    """
    
    def __init__(self, d_model: int, low_rank_dim: int, nhead: int, dropout: float = 0.1):
        super(LowRankCrossAttention, self).__init__()
        
        self.d_model = d_model
        self.low_rank_dim = low_rank_dim
        self.nhead = nhead
        
        # 简化的线性投影
        self.query_proj = nn.Linear(d_model, low_rank_dim, bias=False)
        self.key_proj = nn.Linear(d_model, low_rank_dim, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(low_rank_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        前向传播
        
        Args:
            tgt: 目标序列 [batch_size, tgt_len, d_model]
            memory: 记忆序列 [batch_size, memory_len, d_model]
            tgt_mask: 目标掩码
            memory_mask: 记忆掩码
        
        Returns:
            torch.Tensor: 输出张量 [batch_size, tgt_len, d_model]
        """
        batch_size, tgt_len, d_model = tgt.size()
        memory_len = memory.size(1)
        
        # 确保输入在正确的设备上
        device = tgt.device
        tgt = tgt.to(device)
        memory = memory.to(device)
        
        # 1. 计算Q, K, V
        q = self.query_proj(tgt)      # [batch_size, tgt_len, low_rank_dim]
        k = self.key_proj(memory)     # [batch_size, memory_len, low_rank_dim]
        v = self.value_proj(memory)   # [batch_size, memory_len, d_model]
        
        # 2. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, tgt_len, memory_len]
        
        # 3. 应用掩码
        if memory_mask is not None:
            # 确保掩码在正确的设备上
            memory_mask = memory_mask.to(device)
            scores = scores.masked_fill(memory_mask == 0, -1e9)
        
        # 4. 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, tgt_len, d_model]
        
        # 6. 输出投影
        output = self.out_proj(attn_output)
        
        return output


class SimpleNetwork(nn.Module):
    """简化的网络结构 - 基础版本"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(SimpleNetwork, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.extend([
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

class DeepFCN(nn.Module):
    """深度全连接网络"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(DeepFCN, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        layers.extend([
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

# GAT网络已移除 - 使用ZeroShotGNN替代

class DeepFCNResidual(nn.Module):
    """带残差连接的深度全连接网络 - 优化版本"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.2):
        super(DeepFCNResidual, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims else [256, 128, 64]  # 默认层次结构
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 输入层 - 添加BatchNorm
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # 输入层使用较小的dropout
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            block = ResidualBlock(self.hidden_dims[i], self.hidden_dims[i+1], dropout)
            self.residual_blocks.append(block)
        
        # 注意力机制 - 简化版本
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 4, self.hidden_dims[-1]),
            nn.Sigmoid()
        )
        
        # 输出层 - 优化结构
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用He初始化，适合ReLU激活函数
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """前向传播 - 添加注意力机制"""
        # 输入层
        x = self.input_layer(x)
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 输出层
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    """优化的残差块 - 改进的结构和正则化"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        
        # 主路径 - 使用预激活结构
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        
        # 跳跃连接
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 添加SE注意力模块
        self.se_attention = SEBlock(out_dim, reduction=4)
    
    def forward(self, x):
        """前向传播 - 预激活残差连接"""
        residual = self.shortcut(x)
        out = self.layers(x)
        
        # 应用SE注意力
        out = self.se_attention(out)
        
        # 残差连接
        return out + residual

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播"""
        # 全局平均池化
        b, c = x.size()
        y = x.mean(dim=0, keepdim=True)  # 简化的全局池化
        
        # 激励操作
        y = self.excitation(y)
        
        # 重新加权
        return x * y.expand_as(x)

def create_network(network_type: str, input_dim: int, hidden_dims: List[int], output_dim: int, config=None) -> nn.Module:
    """
    创建指定类型的网络
    
    Args:
        network_type: 网络类型 ("SimpleNetwork", "DeepFCN", "GAT", "DeepFCNResidual", "ZeroShotGNN")
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        
    Returns:
        网络模型实例
    """
    if network_type == "SimpleNetwork":
        return SimpleNetwork(input_dim, hidden_dims, output_dim)
    elif network_type == "DeepFCN":
        return DeepFCN(input_dim, hidden_dims, output_dim)
    # GAT网络已移除，请使用ZeroShotGNN
    elif network_type == "DeepFCNResidual":
        return DeepFCNResidual(input_dim, hidden_dims, output_dim)
    elif network_type == "ZeroShotGNN":
        return ZeroShotGNN(input_dim, hidden_dims, output_dim, config=config)
    else:
        raise ValueError(f"不支持的网络类型: {network_type}")

class ZeroShotGNN(nn.Module):
    """
    真正的零样本图神经网络 - 基于Transformer的架构
    
    核心特性：
    1. 参数共享的实体编码器，支持可变数量的UAV和目标
    2. 自注意力机制学习同类实体间的内部关系
    3. 交叉注意力机制学习UAV-目标间的交互关系
    4. 支持掩码机制，忽略填充的无效数据
    5. 零样本迁移能力，适应不同规模的场景
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1, config=None):
        super(ZeroShotGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims else [256, 128]
        self.output_dim = output_dim
        self.dropout = dropout
        self.config = config  # 保存config引用
        
        # 嵌入维度
        self.embedding_dim = 128
        
        # === 1. 参数共享的实体编码器 ===
        # UAV特征维度：position(2) + heading(1) + resources_ratio(2) + max_distance_norm(1) + 
        #              velocity_norm(2) + is_alive(1) + is_idle(1) = 10
        # 新增：is_idle特征，明确标识UAV是否处于空闲状态]
        # 新增：scarcity_metric_res1(1) + scarcity_metric_res2(1) = 2
        self.uav_feature_dim = 12
        
        # 目标特征维度：position(2) + resources_ratio(2) + value_norm(1) + 
        #              remaining_ratio(2) + is_visible(1) = 8
        self.target_feature_dim = 8
        
        # UAV编码器
        self.uav_encoder = nn.Sequential(
            nn.Linear(self.uav_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        # 目标编码器
        self.target_encoder = nn.Sequential(
            nn.Linear(self.target_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        # === 2. 注意力层（支持新旧模型兼容） ===
        # 检查是否使用低秩近似（新版本）或标准Transformer（旧版本）
        self.use_low_rank_attention = getattr(config, 'USE_LOW_RANK_ATTENTION', True) if config else True
        
        # 强制启用低秩近似注意力以测试性能
        if config and hasattr(config, 'FORCE_LOW_RANK_ATTENTION') and config.FORCE_LOW_RANK_ATTENTION:
            self.use_low_rank_attention = True
        
        if self.use_low_rank_attention:
            # 从config获取注意力头数
            nhead = getattr(config, 'num_heads', 8) if config else 8
            
            # UAV内部自注意力 - 使用低秩近似
            self.uav_self_attention = LowRankSelfAttention(
                d_model=self.embedding_dim,
                low_rank_dim=32,  # 低秩维度，远小于embedding_dim
                nhead=nhead,
                dropout=dropout
            )
            
            # 目标内部自注意力 - 使用低秩近似
            self.target_self_attention = LowRankSelfAttention(
                d_model=self.embedding_dim,
                low_rank_dim=32,
                nhead=nhead,
                dropout=dropout
            )
            
            # UAV-目标交叉注意力 - 使用低秩近似
            self.cross_attention = LowRankCrossAttention(
                d_model=self.embedding_dim,
                low_rank_dim=32,
                nhead=nhead,
                dropout=dropout
            )
        else:
            # 从config获取注意力头数
            nhead = getattr(config, 'num_heads', 8) if config else 8
            
            # 标准Transformer层（兼容旧模型）
            self.uav_self_attention = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=nhead,
                dim_feedforward=256,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            
            self.target_self_attention = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=nhead,
                dim_feedforward=256,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            
            self.cross_attention = nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=nhead,
                dim_feedforward=256,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
        
        # === 4. 位置编码 ===
        self.position_encoder = PositionalEncoding(self.embedding_dim, dropout)
        
        # === 5. Q值解码器 ===
        # 为每个UAV-目标对输出所有可能接近角度的Q值
        # 输出维度为config.GRAPH_N_PHI（角度数量P）
        n_phi = getattr(config, 'GRAPH_N_PHI', 6) if hasattr(config, 'GRAPH_N_PHI') else 6
        self.n_phi = n_phi
        
        self.q_decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_phi)  # 每个UAV-目标对输出n_phi个Q值
        )
        
        # === 6. 空间编码器 ===
        # 预先定义空间编码器，避免动态创建导致的state_dict不匹配
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 32),  # 相对位置(2) + 距离(1)
            nn.ReLU(),
            nn.Linear(32, self.embedding_dim // 4)
        )
        
        # === 7. 全局聚合层 ===
        # 将所有UAV的表示聚合为最终的动作Q值
        self.global_aggregator = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
        self._register_gradient_hooks()
    
    def load_state_dict(self, state_dict, strict=True):
        """
        重写load_state_dict方法，支持新旧模型兼容
        """
        # 检测模型版本
        has_old_attention = any('self_attn.in_proj_weight' in key for key in state_dict.keys())
        
        if has_old_attention:
            # 旧模型，根据config决定是否使用低秩近似
            if getattr(self.config, 'USE_LOW_RANK_ATTENTION', False):
                print("🔍 检测到旧模型格式，但根据配置使用低秩近似注意力机制")
                self.use_low_rank_attention = True
                # 重新初始化低秩注意力层
                self._init_low_rank_attention_layers()
            else:
                print("🔍 检测到旧模型格式，使用标准Transformer注意力机制")
                self.use_low_rank_attention = False
                # 重新初始化标准注意力层
                self._init_old_attention_layers()
        else:
            # 新模型，使用低秩近似
            print("🔍 检测到新模型格式，使用低秩近似注意力机制")
            self.use_low_rank_attention = True
        
        # 调用父类的load_state_dict，忽略不匹配的权重
        try:
            return super().load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"⚠️ 权重加载部分失败，但继续使用新架构: {e}")
            # 即使权重加载失败，也继续使用新的低秩注意力架构
            return None
    
    def _init_old_attention_layers(self):
        """初始化旧版本的注意力层"""
        # 从config获取注意力头数
        nhead = getattr(self.config, 'num_heads', 8) if hasattr(self, 'config') and self.config else 8
        
        self.uav_self_attention = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        
        self.target_self_attention = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        
        self.cross_attention = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
    
    def _init_low_rank_attention_layers(self):
        """初始化低秩近似注意力层"""
        # 从config获取注意力头数
        nhead = getattr(self.config, 'num_heads', 8) if hasattr(self, 'config') and self.config else 8
        
        self.uav_self_attention = LowRankSelfAttention(
            d_model=self.embedding_dim,
            low_rank_dim=32,
            nhead=nhead,
            dropout=self.dropout
        )
        
        self.target_self_attention = LowRankSelfAttention(
            d_model=self.embedding_dim,
            low_rank_dim=32,
            nhead=nhead,
            dropout=self.dropout
        )
        
        self.cross_attention = LowRankCrossAttention(
            d_model=self.embedding_dim,
            low_rank_dim=32,
            nhead=nhead,
            dropout=self.dropout
        )
    
    def _init_weights(self):
        """初始化网络权重 - 数值稳定版本"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 使用更保守的权重初始化
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # 进一步减小gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # 特别处理注意力层的权重
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        nn.init.xavier_uniform_(param, gain=0.05)  # 非常小的gain
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
    
    def _register_gradient_hooks(self):
        """注册梯度裁剪hook"""
        def gradient_hook(grad):
            # 裁剪梯度，防止梯度爆炸
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                return torch.zeros_like(grad)
            return torch.clamp(grad, -1.0, 1.0)
        
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(gradient_hook)
    
    def _create_attention_mask(self, padding_mask):
        """
        创建注意力掩码
        
        Args:
            padding_mask: [batch_size, seq_len] 布尔掩码，True表示需要忽略的位置
        
        Returns:
            torch.Tensor: [batch_size, seq_len, seq_len] 注意力掩码
        """
        batch_size, seq_len = padding_mask.shape
        # 确保掩码在正确的设备上
        device = padding_mask.device
        # 创建2D掩码，True表示需要忽略的位置
        mask = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)  # [batch_size, seq_len, seq_len]
        return mask.to(device)
    
    def forward(self, graph_obs):
        """
        前向传播 - 处理图结构观测
        
        Args:
            graph_obs (dict): 图结构观测字典，包含：
                - uav_features: [batch_size, N_uav, uav_feature_dim]
                - target_features: [batch_size, N_target, target_feature_dim]
                - relative_positions: [batch_size, N_uav, N_target, 2]
                - distances: [batch_size, N_uav, N_target]
                - masks: 掩码字典
        
        Returns:
            torch.Tensor: Q值 [batch_size, N_actions]
        """
        # 提取输入并确保设备一致性
        uav_features = graph_obs["uav_features"]  # [batch_size, N_uav, uav_feat_dim]
        target_features = graph_obs["target_features"]  # [batch_size, N_target, target_feat_dim]
        relative_positions = graph_obs["relative_positions"]  # [batch_size, N_uav, N_target, 2]
        distances = graph_obs["distances"]  # [batch_size, N_uav, N_target]
        uav_mask = graph_obs["masks"]["uav_mask"]  # [batch_size, N_uav]
        target_mask = graph_obs["masks"]["target_mask"]  # [batch_size, N_target]
        
        # 确保所有输入张量都在正确的设备上
        device = next(self.parameters()).device
        uav_features = uav_features.to(device)
        target_features = target_features.to(device)
        relative_positions = relative_positions.to(device)
        distances = distances.to(device)
        uav_mask = uav_mask.to(device)
        target_mask = target_mask.to(device)
        
        # 形状检查和修复 - 处理各种维度情况
        try:
            # 处理uav_features的各种形状
            if len(uav_features.shape) == 4:
                # 如果是4维 [1, 1, N, features]，压缩为3维 [1, N, features]
                uav_features = uav_features.squeeze(1)
            elif len(uav_features.shape) == 2:
                # 如果是2维，添加batch维度
                uav_features = uav_features.unsqueeze(0)
            elif len(uav_features.shape) == 1:
                # 如果是1维，需要重塑为[1, 1, features]
                uav_features = uav_features.unsqueeze(0).unsqueeze(0)
                
            # 处理target_features的各种形状
            if len(target_features.shape) == 4:
                # 如果是4维 [1, 1, N, features]，压缩为3维 [1, N, features]
                target_features = target_features.squeeze(1)
            elif len(target_features.shape) == 2:
                # 如果是2维，添加batch维度
                target_features = target_features.unsqueeze(0)
            elif len(target_features.shape) == 1:
                # 如果是1维，需要重塑为[1, 1, features]
                target_features = target_features.unsqueeze(0).unsqueeze(0)
            
            # 处理其他张量的维度
            if len(relative_positions.shape) == 5:
                # 如果是5维，压缩为4维
                relative_positions = relative_positions.squeeze(1)
            elif len(relative_positions.shape) == 3:
                relative_positions = relative_positions.unsqueeze(0)
            elif len(relative_positions.shape) == 2:
                relative_positions = relative_positions.unsqueeze(0).unsqueeze(0)
                
            if len(distances.shape) == 4:
                # 如果是4维，压缩为3维
                distances = distances.squeeze(1)
            elif len(distances.shape) == 2:
                distances = distances.unsqueeze(0)
            elif len(distances.shape) == 1:
                distances = distances.unsqueeze(0).unsqueeze(0)
                
            if len(uav_mask.shape) == 3:
                # 如果是3维，压缩为2维
                uav_mask = uav_mask.squeeze(1)
            elif len(uav_mask.shape) == 1:
                uav_mask = uav_mask.unsqueeze(0)
                
            if len(target_mask.shape) == 3:
                # 如果是3维，压缩为2维
                target_mask = target_mask.squeeze(1)
            elif len(target_mask.shape) == 1:
                target_mask = target_mask.unsqueeze(0)
                
            # 确保形状正确后再解包
            if len(uav_features.shape) != 3:
                raise ValueError(f"uav_features形状异常: {uav_features.shape}, 期望3维")
            if len(target_features.shape) != 3:
                raise ValueError(f"target_features形状异常: {target_features.shape}, 期望3维")
                
            batch_size, n_uavs, _ = uav_features.shape
            _, n_targets, _ = target_features.shape
            
        except Exception as e:
            print(f"[ERROR] ZeroShotGNN前向传播形状错误: {e}")
            print(f"[ERROR] uav_features形状: {uav_features.shape}")
            print(f"[ERROR] target_features形状: {target_features.shape}")
            raise e
        
        # === 1. 增强的实体编码（融合空间信息）===
        # 数值稳定性检查 - 输入验证
        uav_features = torch.clamp(uav_features, -1e6, 1e6)
        target_features = torch.clamp(target_features, -1e6, 1e6)
        
        # 编码UAV特征
        uav_embeddings = self.uav_encoder(uav_features)  # [batch_size, N_uav, embedding_dim]
        uav_embeddings = torch.clamp(uav_embeddings, -1e6, 1e6)  # 防止编码器输出异常值
        
        # 编码目标特征
        target_embeddings = self.target_encoder(target_features)  # [batch_size, N_target, embedding_dim]
        target_embeddings = torch.clamp(target_embeddings, -1e6, 1e6)  # 防止编码器输出异常值
        
        # 融合空间关系信息到UAV嵌入中
        uav_embeddings_enhanced = self._enhance_uav_embeddings_with_spatial_info(
            uav_embeddings, target_embeddings, relative_positions, distances
        )
        
        # 添加位置编码
        uav_embeddings_enhanced = self.position_encoder(uav_embeddings_enhanced)
        target_embeddings = self.position_encoder(target_embeddings)
        
        # === 2. 自注意力 ===
        # 安全的掩码处理，避免维度不匹配
        try:
            # UAV内部自注意力 - 学习UAV间的协作关系
            uav_mask_bool = (uav_mask == 0)  # 转换为布尔掩码，True表示需要忽略的位置
            
            # 确保掩码维度正确 [batch_size, N_uav]
            if uav_mask_bool.dim() == 1:
                uav_mask_bool = uav_mask_bool.unsqueeze(0)
            
            if self.use_low_rank_attention:
                # 为低秩注意力创建掩码
                uav_attention_mask = self._create_attention_mask(uav_mask_bool)
                
                uav_contextualized = self.uav_self_attention(
                    uav_embeddings_enhanced,
                    mask=uav_attention_mask
                )  # [batch_size, N_uav, embedding_dim]
                
                # 目标内部自注意力 - 学习目标间的依赖关系
                target_mask_bool = (target_mask == 0)
                
                # 确保掩码维度正确 [batch_size, N_target]
                if target_mask_bool.dim() == 1:
                    target_mask_bool = target_mask_bool.unsqueeze(0)
                
                # 为低秩注意力创建掩码
                target_attention_mask = self._create_attention_mask(target_mask_bool)
                
                target_contextualized = self.target_self_attention(
                    target_embeddings,
                    mask=target_attention_mask
                )  # [batch_size, N_target, embedding_dim]
            else:
                # 标准Transformer注意力
                uav_contextualized = self.uav_self_attention(
                    uav_embeddings_enhanced,
                    src_key_padding_mask=uav_mask_bool
                )  # [batch_size, N_uav, embedding_dim]
                
                # 目标内部自注意力 - 学习目标间的依赖关系
                target_mask_bool = (target_mask == 0)
                
                # 确保掩码维度正确 [batch_size, N_target]
                if target_mask_bool.dim() == 1:
                    target_mask_bool = target_mask_bool.unsqueeze(0)
                
                target_contextualized = self.target_self_attention(
                    target_embeddings,
                    src_key_padding_mask=target_mask_bool
                )  # [batch_size, N_target, embedding_dim]
            
            # === 3. 优化的逐无人机交叉注意力 ===
            # 使用逐无人机的方式计算交叉注意力，避免创建巨大张量
            uav_target_aware = torch.zeros_like(uav_contextualized)
            
            # 逐个处理每架无人机，避免显存瓶颈
            for uav_idx in range(n_uavs):
                # 检查当前UAV是否有效
                if uav_idx < uav_mask_bool.shape[1] and uav_mask_bool[0, uav_idx]:
                    continue  # 跳过无效的UAV
                
                # 提取单个UAV的表示 [batch_size, 1, embedding_dim]
                single_uav = uav_contextualized[:, uav_idx:uav_idx+1, :]
                
                # 为单个UAV计算对所有目标的交叉注意力
                try:
                    if self.use_low_rank_attention:
                        # 为交叉注意力创建掩码
                        cross_attention_mask = self._create_attention_mask(target_mask_bool)
                        
                        single_uav_aware = self.cross_attention(
                            tgt=single_uav,  # query: 单个UAV表示
                            memory=target_contextualized,  # key & value: 所有目标表示
                            tgt_mask=None,  # 单个UAV不需要掩码
                            memory_mask=cross_attention_mask
                        )  # [batch_size, 1, embedding_dim]
                    else:
                        # 标准Transformer交叉注意力
                        single_uav_aware = self.cross_attention(
                            tgt=single_uav,  # query: 单个UAV表示
                            memory=target_contextualized,  # key & value: 所有目标表示
                            tgt_key_padding_mask=None,  # 单个UAV不需要掩码
                            memory_key_padding_mask=target_mask_bool
                        )  # [batch_size, 1, embedding_dim]
                    
                    # 将结果存储回原位置
                    uav_target_aware[:, uav_idx, :] = single_uav_aware.squeeze(1)
                    
                except Exception as e:
                    # 如果单个UAV的交叉注意力失败，使用原始表示
                    uav_target_aware[:, uav_idx, :] = uav_contextualized[:, uav_idx, :]
            
        except Exception as e:
            # 如果注意力机制失败，使用简化的处理方式
            print(f"注意力机制失败，使用简化处理: {str(e)[:100]}...")
            uav_target_aware = uav_embeddings_enhanced
            target_contextualized = target_embeddings
        
        # === 4. 完整的Q值解码 - 输出四维张量 ===
        batch_size, n_uavs, embed_dim = uav_target_aware.shape
        _, n_targets, _ = target_contextualized.shape
        
        # 初始化四维Q值张量 [batch_size, n_uavs, n_targets, n_phi]
        # 使用实际的UAV和目标数量，而不是固定的最大值
        q_values_4d = torch.full((batch_size, n_uavs, n_targets, self.n_phi), 
                                float('-inf'), device=uav_target_aware.device)
        
        # 为每个有效的UAV-目标对计算Q值
        for i in range(n_uavs):
            for j in range(n_targets):
                # 计算UAV-目标交互特征
                uav_emb = uav_target_aware[:, i, :]  # [batch_size, embed_dim]
                target_emb = target_contextualized[:, j, :]  # [batch_size, embed_dim]
                
                # 数值稳定性检查
                uav_emb = torch.clamp(uav_emb, -1e6, 1e6)
                target_emb = torch.clamp(target_emb, -1e6, 1e6)
                
                # 简单的加法融合（可以改为更复杂的融合方式）
                interaction_emb = uav_emb + target_emb  # [batch_size, embed_dim]
                interaction_emb = torch.clamp(interaction_emb, -1e6, 1e6)
                
                # 通过Q值解码器得到所有角度的Q值
                q_values_phi = self.q_decoder(interaction_emb)  # [batch_size, n_phi]
                
                # 数值稳定性检查Q值输出
                if torch.isnan(q_values_phi).any() or torch.isinf(q_values_phi).any():
                    q_values_phi = torch.full_like(q_values_phi, -1e-3)
                else:
                    q_values_phi = torch.clamp(q_values_phi, -1e6, 1e6)
                
                # 存储到四维张量中
                q_values_4d[:, i, j, :] = q_values_phi
        
        # === 5. 应用掩码操作 ===
        # 对无效的UAV和目标设置极小负数
        for i in range(n_uavs):
            for j in range(n_targets):
                # 检查UAV是否有效
                if i < uav_mask.shape[1] and uav_mask[0, i] == 0:
                    q_values_4d[:, i, j, :] = -1e9
                
                # 检查目标是否有效
                if j < target_mask.shape[1] and target_mask[0, j] == 0:
                    q_values_4d[:, i, j, :] = -1e9
        
        # === 6. 展平为一维向量 ===
        # 将四维张量展平为 [batch_size, n_uavs * n_targets * n_phi]
        q_values_final = q_values_4d.view(batch_size, -1)
        
        # 如果输出维度与期望的output_dim不匹配，进行调整
        if q_values_final.shape[1] != self.output_dim:
            if q_values_final.shape[1] > self.output_dim:
                # 如果实际输出大于期望，截断
                q_values_final = q_values_final[:, :self.output_dim]
            else:
                # 如果实际输出小于期望，填充
                padding = torch.full((batch_size, self.output_dim - q_values_final.shape[1]), 
                                   -1e9, device=q_values_final.device)
                q_values_final = torch.cat([q_values_final, padding], dim=1)
        
        # 【NaN修复】输出验证和修复
        if torch.isnan(q_values_final).any() or torch.isinf(q_values_final).any():
            print("⚠️ 警告: ZeroShotGNN网络输出包含NaN/Inf，使用安全默认值")
            # 使用小的负值而不是零，保持Q-learning的探索性
            q_values_final = torch.full_like(q_values_final, -1e-3)
        
        return q_values_final
    
    def _enhance_uav_embeddings_with_spatial_info(self, uav_embeddings, target_embeddings, 
                                                  relative_positions, distances):
        """
        使用空间信息增强UAV嵌入
        
        Args:
            uav_embeddings: UAV嵌入 [batch_size, N_uav, embedding_dim]
            target_embeddings: 目标嵌入 [batch_size, N_target, embedding_dim]
            relative_positions: 相对位置 [batch_size, N_uav, N_target, 2]
            distances: 距离矩阵 [batch_size, N_uav, N_target]
        
        Returns:
            torch.Tensor: 增强的UAV嵌入
        """
        batch_size, n_uavs, embedding_dim = uav_embeddings.shape
        _, n_targets, _ = target_embeddings.shape
        
        # 使用预定义的空间编码器
        
        # 为每个UAV计算空间上下文
        enhanced_embeddings = []
        
        for uav_idx in range(n_uavs):
            # 获取该UAV到所有目标的空间信息
            # 检查relative_positions的维度
            if relative_positions.dim() == 4:
                uav_rel_pos = relative_positions[:, uav_idx, :, :]  # [batch_size, N_target, 2]
            elif relative_positions.dim() == 3:
                # 如果是3维，假设是[batch_size, N_uav*N_target, 2]，需要重新整形
                n_targets = relative_positions.shape[1] // n_uavs
                uav_rel_pos = relative_positions[:, uav_idx*n_targets:(uav_idx+1)*n_targets, :]  # [batch_size, N_target, 2]
            else:
                # 降级处理：使用零张量
                uav_rel_pos = torch.zeros(batch_size, n_targets, 2, device=relative_positions.device)
            
            # 检查distances的维度
            if distances.dim() == 3:
                uav_distances = distances[:, uav_idx, :].unsqueeze(-1)  # [batch_size, N_target, 1]
            elif distances.dim() == 2:
                # 如果是2维，假设是[batch_size, N_uav*N_target]，需要重新整形
                n_targets = distances.shape[1] // n_uavs
                uav_distances = distances[:, uav_idx*n_targets:(uav_idx+1)*n_targets].unsqueeze(-1)  # [batch_size, N_target, 1]
            else:
                # 降级处理：使用零张量
                uav_distances = torch.zeros(batch_size, n_targets, 1, device=distances.device)
            
            # 组合空间特征
            # 确保两个张量的前两个维度匹配
            if uav_rel_pos.shape[:2] != uav_distances.shape[:2]:
                # 如果维度不匹配，使用较小的维度
                min_batch = min(uav_rel_pos.shape[0], uav_distances.shape[0])
                min_targets = min(uav_rel_pos.shape[1], uav_distances.shape[1])
                uav_rel_pos = uav_rel_pos[:min_batch, :min_targets, :]
                uav_distances = uav_distances[:min_batch, :min_targets, :]
            
            spatial_features = torch.cat([uav_rel_pos, uav_distances], dim=-1)  # [batch_size, N_target, 3]
            
            # 编码空间特征
            spatial_encoded = self.spatial_encoder(spatial_features)  # [batch_size, N_target, embedding_dim//4]
            
            # 聚合空间上下文（使用注意力权重）
            spatial_context = spatial_encoded.mean(dim=1)  # [batch_size, embedding_dim//4]
            
            # 将空间上下文融合到UAV嵌入中
            uav_emb = uav_embeddings[:, uav_idx, :]  # [batch_size, embedding_dim]
            
            # 简单的拼接融合（可以改为更复杂的融合方式）
            if spatial_context.shape[-1] + uav_emb.shape[-1] <= embedding_dim:
                # 如果维度允许，直接拼接
                padding_size = embedding_dim - spatial_context.shape[-1] - uav_emb.shape[-1]
                if padding_size > 0:
                    padding = torch.zeros(batch_size, padding_size, device=uav_emb.device)
                    enhanced_emb = torch.cat([uav_emb[:, :embedding_dim-spatial_context.shape[-1]], 
                                            spatial_context, padding], dim=-1)
                else:
                    enhanced_emb = torch.cat([uav_emb[:, :embedding_dim-spatial_context.shape[-1]], 
                                            spatial_context], dim=-1)
            else:
                # 使用加权融合
                spatial_weight = 0.2
                enhanced_emb = (1 - spatial_weight) * uav_emb + spatial_weight * torch.cat([
                    spatial_context, torch.zeros(batch_size, embedding_dim - spatial_context.shape[-1], 
                                                device=spatial_context.device)
                ], dim=-1)
            
            enhanced_embeddings.append(enhanced_emb.unsqueeze(1))
        
        return torch.cat(enhanced_embeddings, dim=1)  # [batch_size, N_uav, embedding_dim]
    
    def _compute_q_values_vectorized(self, uav_target_aware, target_contextualized, uav_mask, target_mask):
        """
        向量化计算Q值，提高效率
        
        Args:
            uav_target_aware: UAV目标感知表示 [batch_size, N_uav, embedding_dim]
            target_contextualized: 目标上下文表示 [batch_size, N_target, embedding_dim]
            uav_mask: UAV掩码 [batch_size, N_uav]
            target_mask: 目标掩码 [batch_size, N_target]
        
        Returns:
            torch.Tensor: Q值矩阵 [batch_size, N_uav * N_target]
        """
        batch_size, n_uavs, embedding_dim = uav_target_aware.shape
        _, n_targets, _ = target_contextualized.shape
        
        # 扩展维度以进行广播
        uav_expanded = uav_target_aware.unsqueeze(2)  # [batch_size, N_uav, 1, embedding_dim]
        target_expanded = target_contextualized.unsqueeze(1)  # [batch_size, 1, N_target, embedding_dim]
        
        # 计算UAV-目标交互特征
        interaction_features = uav_expanded + target_expanded  # [batch_size, N_uav, N_target, embedding_dim]
        
        # 重塑为批次处理
        interaction_flat = interaction_features.view(batch_size * n_uavs * n_targets, embedding_dim)
        
        # 通过Q值解码器
        q_values_flat = self.q_decoder(interaction_flat)  # [batch_size * N_uav * N_target, 1]
        
        # 重塑回原始形状
        q_values_matrix = q_values_flat.view(batch_size, n_uavs * n_targets)
        
        return q_values_matrix
    

    
    def _create_action_mask(self, uav_mask, target_mask, n_phi):
        """
        创建动作掩码，屏蔽无效的UAV-目标-phi组合
        
        Args:
            uav_mask: UAV掩码 [batch_size, N_uav]
            target_mask: 目标掩码 [batch_size, N_target]
            n_phi: phi维度数量
        
        Returns:
            torch.Tensor: 动作掩码 [batch_size, N_actions]
        """
        batch_size, n_uavs = uav_mask.shape
        _, n_targets = target_mask.shape
        
        # 创建UAV-目标对掩码
        uav_mask_expanded = uav_mask.unsqueeze(2)  # [batch_size, N_uav, 1]
        target_mask_expanded = target_mask.unsqueeze(1)  # [batch_size, 1, N_target]
        
        # 无效的UAV-目标对：任一实体无效
        pair_mask = (uav_mask_expanded == 0) | (target_mask_expanded == 0)  # [batch_size, N_uav, N_target]
        
        # 扩展到包含phi维度
        pair_mask_expanded = pair_mask.unsqueeze(-1).repeat(1, 1, 1, n_phi)  # [batch_size, N_uav, N_target, n_phi]
        action_mask = pair_mask_expanded.view(batch_size, -1)  # [batch_size, N_actions]
        
        return action_mask

class PositionalEncoding(nn.Module):
    """
    位置编码模块 - 为序列添加位置信息
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def get_network_info(network_type: str) -> dict:
    """
    获取网络信息
    
    Args:
        network_type: 网络类型
        
    Returns:
        网络信息字典
    """
    network_info = {
        "SimpleNetwork": {
            "description": "基础全连接网络",
            "features": ["BatchNorm", "Dropout", "Xavier初始化"],
            "complexity": "低"
        },
        "DeepFCN": {
            "description": "深度全连接网络",
            "features": ["多层结构", "BatchNorm", "Dropout"],
            "complexity": "中"
        },
        # GAT网络已移除
        "DeepFCNResidual": {
            "description": "带残差连接的深度网络",
            "features": ["残差连接", "BatchNorm", "Dropout"],
            "complexity": "中"
        },
        "ZeroShotGNN": {
            "description": "零样本图神经网络",
            "features": ["Transformer架构", "自注意力", "交叉注意力", "参数共享", "零样本迁移"],
            "complexity": "高"
        }
    }
    
    return network_info.get(network_type, {"description": "未知网络", "features": [], "complexity": "未知"}) 