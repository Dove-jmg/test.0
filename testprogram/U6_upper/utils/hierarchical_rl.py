import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Union, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class U6HighLevelPolicy(nn.Module):
    """U6_upper的高层策略网络，专为6无人机系统设计"""
    
    def __init__(
        self, 
        task_feature_dim: int = 64,
        uav_feature_dim: int = 32,
        region_feature_dim: int = 32,
        time_feature_dim: int = 16,
        hidden_dim: int = 256,
        num_uavs: int = 6,  # 固定为6个无人机
        num_regions: int = 6,
        num_priority_levels: int = 3
    ):
        super(U6HighLevelPolicy, self).__init__()
        
        # 特征维度
        self.task_feature_dim = task_feature_dim
        self.uav_feature_dim = uav_feature_dim
        self.region_feature_dim = region_feature_dim
        self.time_feature_dim = time_feature_dim
        
        # 编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(5, task_feature_dim),  # 任务特征：[类型, 位置x, 位置y, 数据量, 时间窗口]
            nn.ReLU(),
            nn.Linear(task_feature_dim, task_feature_dim),
            nn.ReLU()
        )
        
        self.uav_encoder = nn.Sequential(
            nn.Linear(4, uav_feature_dim),  # UAV特征：[位置x, 位置y, 存储容量, 最大飞行距离]
            nn.ReLU(),
            nn.Linear(uav_feature_dim, uav_feature_dim),
            nn.ReLU()
        )
        
        self.region_encoder = nn.Sequential(
            nn.Linear(3, region_feature_dim),  # 区域特征：[中心x, 中心y, 面积]
            nn.ReLU(),
            nn.Linear(region_feature_dim, region_feature_dim),
            nn.ReLU()
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(2, time_feature_dim),  # 时间特征：[当前时间, 任务总时长]
            nn.ReLU(),
            nn.Linear(time_feature_dim, time_feature_dim),
            nn.ReLU()
        )
        
        # 高层策略网络 - U6专用版本
        self.policy_network = nn.Sequential(
            nn.Linear(task_feature_dim + uav_feature_dim + region_feature_dim + time_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 用于产生不同动作的头部网络
        self.assign_task_head = nn.Linear(hidden_dim, num_uavs)  # 分配任务给6个无人机
        self.divide_region_head = nn.Linear(hidden_dim, num_regions)  # 划分区域
        self.set_priority_head = nn.Linear(hidden_dim, num_priority_levels)  # 设置优先级
        self.collaboration_head = nn.Linear(hidden_dim, num_uavs)  # 协作组分配
        
    def forward(
        self, 
        task_features: torch.Tensor,
        uav_features: torch.Tensor,
        region_features: torch.Tensor,
        time_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # 编码任务、无人机、区域和时间特征
        task_encoding = self.task_encoder(task_features)
        uav_encoding = self.uav_encoder(uav_features)
        region_encoding = self.region_encoder(region_features)
        time_encoding = self.time_encoder(time_features)
        
        # 连接所有特征
        combined_features = torch.cat([
            task_encoding, uav_encoding, region_encoding, time_encoding
        ], dim=1)
        
        # 通过策略网络
        policy_features = self.policy_network(combined_features)
        
        # 计算各种动作的概率分布
        assign_probs = F.softmax(self.assign_task_head(policy_features), dim=1)
        divide_probs = F.softmax(self.divide_region_head(policy_features), dim=1)
        priority_probs = F.softmax(self.set_priority_head(policy_features), dim=1)
        collaboration_probs = F.sigmoid(self.collaboration_head(policy_features))  # 使用sigmoid以便多选
        
        return {
            "assign_probs": assign_probs,
            "divide_probs": divide_probs,
            "priority_probs": priority_probs,
            "collaboration_probs": collaboration_probs
        }

class U6LowLevelPolicy(nn.Module):
    """U6_upper的低层策略网络，为6无人机系统设计"""
    
    def __init__(
        self,
        state_dim: int = 12,  # [位置x, 位置y, 剩余存储, 任务特征...]
        action_dim: int = 6,  # [执行任务, 调整顺序, 暂停卸载, 协作, 待命, 不操作]
        hidden_dim: int = 128,
        task_embed_dim: int = 32
    ):
        super(U6LowLevelPolicy, self).__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.task_encoder = nn.Sequential(
            nn.Linear(5, task_embed_dim),  # 任务特征：[类型, 位置x, 位置y, 数据量, 时间窗口]
            nn.ReLU()
        )
        
        # 协作特征编码器（用于编码协作组信息）
        self.collaboration_encoder = nn.Sequential(
            nn.Linear(6, 16),  # 6个无人机的协作状态
            nn.ReLU()
        )
        
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim + task_embed_dim + 16, hidden_dim),  # 加入协作特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略头
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # 价值头（用于计算优势函数）
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        state: torch.Tensor,
        task_features: torch.Tensor,
        collaboration_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码状态和任务
        state_encoding = self.state_encoder(state)
        task_encoding = self.task_encoder(task_features)
        collab_encoding = self.collaboration_encoder(collaboration_features)
        
        # 连接所有特征
        combined_features = torch.cat([state_encoding, task_encoding, collab_encoding], dim=1)
        
        # 通过策略网络
        policy_features = self.policy_network(combined_features)
        
        # 计算动作概率和状态价值
        action_probs = F.softmax(self.action_head(policy_features), dim=1)
        state_value = self.value_head(policy_features)
        
        return action_probs, state_value 