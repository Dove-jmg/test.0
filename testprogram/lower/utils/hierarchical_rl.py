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

class HighLevelPolicy(nn.Module):
    """高层策略网络，负责任务分配和区域划分"""
    
    def __init__(
        self, 
        task_feature_dim: int = 64,
        uav_feature_dim: int = 32,
        region_feature_dim: int = 32,
        time_feature_dim: int = 16,
        hidden_dim: int = 256,
        num_uavs: int = 6,
        num_regions: int = 4,
        num_priority_levels: int = 3
    ):
        super(HighLevelPolicy, self).__init__()
        
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
        
        # 高层策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(task_feature_dim + uav_feature_dim + region_feature_dim + time_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 用于产生不同动作的头部网络
        self.assign_task_head = nn.Linear(hidden_dim, num_uavs)  # 分配任务给无人机
        self.divide_region_head = nn.Linear(hidden_dim, num_regions)  # 划分区域
        self.set_priority_head = nn.Linear(hidden_dim, num_priority_levels)  # 设置优先级
        
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
        
        return {
            "assign_probs": assign_probs,
            "divide_probs": divide_probs,
            "priority_probs": priority_probs
        }

class LowLevelPolicy(nn.Module):
    """低层策略网络，负责具体任务执行和顺序调整"""
    
    def __init__(
        self,
        state_dim: int = 12,  # [位置x, 位置y, 剩余存储, 任务特征...]
        action_dim: int = 4,  # [执行任务, 调整顺序, 暂停卸载, 不操作]
        hidden_dim: int = 128,
        task_embed_dim: int = 32
    ):
        super(LowLevelPolicy, self).__init__()
        
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
        
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim + task_embed_dim, hidden_dim),
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
        task_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码状态和任务
        state_encoding = self.state_encoder(state)
        task_encoding = self.task_encoder(task_features)
        
        # 连接状态和任务特征
        combined_features = torch.cat([state_encoding, task_encoding], dim=1)
        
        # 通过策略网络
        policy_features = self.policy_network(combined_features)
        
        # 计算动作概率和状态价值
        action_probs = F.softmax(self.action_head(policy_features), dim=1)
        state_value = self.value_head(policy_features)
        
        return action_probs, state_value

class HierarchicalRL:
    """分层强化学习管理器"""
    
    def __init__(
        self,
        high_level_policy: HighLevelPolicy,
        low_level_policy: LowLevelPolicy,
        high_level_lr: float = 1e-4,
        low_level_lr: float = 1e-4,
        gamma: float = 0.99,
        lambd: float = 0.95,
        high_level_weights: List[float] = [0.5, 0.3, 0.2],
        low_level_weights: List[float] = [0.6, 0.2, 0.2],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # 初始化高层和低层策略
        self.high_level_policy = high_level_policy.to(device)
        self.low_level_policy = low_level_policy.to(device)
        
        # 优化器
        self.high_level_optimizer = torch.optim.Adam(
            self.high_level_policy.parameters(), lr=high_level_lr
        )
        self.low_level_optimizer = torch.optim.Adam(
            self.low_level_policy.parameters(), lr=low_level_lr
        )
        
        # 强化学习参数
        self.gamma = gamma  # 折扣因子
        self.lambd = lambd  # GAE参数
        
        # 高层和低层奖励权重
        self.high_level_weights = high_level_weights
        self.low_level_weights = low_level_weights
        
        # 高层和低层记忆缓冲区
        self.high_level_buffer = []
        self.low_level_buffer = []
        
    def high_level_reward(
        self,
        assigned_value: float,
        total_value: float,
        num_uavs: int,
        max_uavs: int,
        high_delay: float,
        total_time: float
    ) -> float:
        """计算高层奖励函数"""
        # 任务分配奖励
        assign_reward = self.high_level_weights[0] * (assigned_value / total_value)
        
        # 无人机使用效率奖励
        uav_reward = self.high_level_weights[1] * (1 - (num_uavs / max_uavs))
        
        # 延迟惩罚
        delay_penalty = self.high_level_weights[2] * (high_delay / total_time)
        
        return assign_reward + uav_reward - delay_penalty
    
    def low_level_reward(
        self,
        completed_value: float,
        assigned_value: float,
        storage_left: float,
        max_storage: float,
        low_delay: float,
        total_time: float
    ) -> float:
        """计算低层奖励函数"""
        # 完成任务奖励
        completion_reward = self.low_level_weights[0] * (completed_value / assigned_value)
        
        # 存储管理奖励
        storage_reward = self.low_level_weights[1] * (storage_left / max_storage)
        
        # 延迟惩罚
        delay_penalty = self.low_level_weights[2] * (low_delay / total_time)
        
        return completion_reward + storage_reward - delay_penalty
    
    def high_level_step(
        self,
        task_features: torch.Tensor,
        uav_features: torch.Tensor,
        region_features: torch.Tensor,
        time_features: torch.Tensor
    ) -> Dict[str, Union[int, List[float]]]:
        """执行高层决策步骤"""
        with torch.no_grad():
            task_features_t = torch.FloatTensor(task_features).to(self.device)
            uav_features_t = torch.FloatTensor(uav_features).to(self.device)
            region_features_t = torch.FloatTensor(region_features).to(self.device)
            time_features_t = torch.FloatTensor(time_features).to(self.device)
            
            # 获取高层策略输出
            policy_output = self.high_level_policy(
                task_features_t.unsqueeze(0),
                uav_features_t.unsqueeze(0),
                region_features_t.unsqueeze(0),
                time_features_t.unsqueeze(0)
            )
            
            # 从概率分布中采样动作
            assign_probs = policy_output["assign_probs"].squeeze(0).cpu().numpy()
            divide_probs = policy_output["divide_probs"].squeeze(0).cpu().numpy()
            priority_probs = policy_output["priority_probs"].squeeze(0).cpu().numpy()
            
            # 采样具体动作
            assigned_uav = np.random.choice(len(assign_probs), p=assign_probs)
            divided_region = np.random.choice(len(divide_probs), p=divide_probs)
            priority_level = np.random.choice(len(priority_probs), p=priority_probs)
            
            action = {
                "assigned_uav": int(assigned_uav),
                "divided_region": int(divided_region),
                "priority_level": int(priority_level),
                "assign_probs": assign_probs.tolist(),
                "divide_probs": divide_probs.tolist(),
                "priority_probs": priority_probs.tolist()
            }
            
            return action
    
    def low_level_step(
        self,
        state: torch.Tensor,
        task_features: torch.Tensor
    ) -> Dict[str, Union[int, float]]:
        """执行低层决策步骤"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            task_features_t = torch.FloatTensor(task_features).to(self.device)
            
            # 获取低层策略输出
            action_probs, state_value = self.low_level_policy(
                state_t.unsqueeze(0),
                task_features_t.unsqueeze(0)
            )
            
            # 从概率分布中采样动作
            action_probs = action_probs.squeeze(0).cpu().numpy()
            state_value = state_value.squeeze(0).item()
            
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            
            action = {
                "action_idx": int(action_idx),
                "action_probs": action_probs.tolist(),
                "state_value": state_value
            }
            
            return action
    
    def store_high_level_transition(
        self,
        task_features: torch.Tensor,
        uav_features: torch.Tensor,
        region_features: torch.Tensor,
        time_features: torch.Tensor,
        action: Dict[str, Any],
        reward: float,
        next_task_features: torch.Tensor,
        next_uav_features: torch.Tensor,
        next_region_features: torch.Tensor,
        next_time_features: torch.Tensor,
        done: bool
    ):
        """存储高层策略的转换记忆"""
        self.high_level_buffer.append({
            "task_features": task_features,
            "uav_features": uav_features,
            "region_features": region_features,
            "time_features": time_features,
            "action": action,
            "reward": reward,
            "next_task_features": next_task_features,
            "next_uav_features": next_uav_features,
            "next_region_features": next_region_features,
            "next_time_features": next_time_features,
            "done": done
        })
    
    def store_low_level_transition(
        self,
        state: torch.Tensor,
        task_features: torch.Tensor,
        action: Dict[str, Any],
        reward: float,
        next_state: torch.Tensor,
        next_task_features: torch.Tensor,
        done: bool
    ):
        """存储低层策略的转换记忆"""
        self.low_level_buffer.append({
            "state": state,
            "task_features": task_features,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "next_task_features": next_task_features,
            "done": done
        })
    
    def update_high_level_policy(self, batch_size: int = 64):
        """更新高层策略"""
        if len(self.high_level_buffer) < batch_size:
            return
        
        indices = np.random.choice(len(self.high_level_buffer), batch_size, replace=False)
        batch = [self.high_level_buffer[i] for i in indices]
        
        task_features = torch.FloatTensor(np.array([b["task_features"] for b in batch])).to(self.device)
        uav_features = torch.FloatTensor(np.array([b["uav_features"] for b in batch])).to(self.device)
        region_features = torch.FloatTensor(np.array([b["region_features"] for b in batch])).to(self.device)
        time_features = torch.FloatTensor(np.array([b["time_features"] for b in batch])).to(self.device)
        
        rewards = torch.FloatTensor(np.array([b["reward"] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b["done"] for b in batch])).to(self.device)
        
        next_task_features = torch.FloatTensor(np.array([b["next_task_features"] for b in batch])).to(self.device)
        next_uav_features = torch.FloatTensor(np.array([b["next_uav_features"] for b in batch])).to(self.device)
        next_region_features = torch.FloatTensor(np.array([b["next_region_features"] for b in batch])).to(self.device)
        next_time_features = torch.FloatTensor(np.array([b["next_time_features"] for b in batch])).to(self.device)
        
        # 获取当前策略输出
        current_output = self.high_level_policy(task_features, uav_features, region_features, time_features)
        
        # 创建目标动作的one-hot向量
        assigned_uavs = torch.LongTensor(np.array([b["action"]["assigned_uav"] for b in batch])).to(self.device)
        divided_regions = torch.LongTensor(np.array([b["action"]["divided_region"] for b in batch])).to(self.device)
        priority_levels = torch.LongTensor(np.array([b["action"]["priority_level"] for b in batch])).to(self.device)
        
        # 计算下一状态的价值估计
        with torch.no_grad():
            next_output = self.high_level_policy(next_task_features, next_uav_features, next_region_features, next_time_features)
            # 这里简化为使用最高概率的动作的简单估计
            next_values = (
                torch.max(next_output["assign_probs"], dim=1)[0] +
                torch.max(next_output["divide_probs"], dim=1)[0] +
                torch.max(next_output["priority_probs"], dim=1)[0]
            ) / 3.0
            
            # 使用贝尔曼方程计算目标价值
            targets = rewards + self.gamma * next_values * (1 - dones)
        
        # 计算当前估计
        current_values = (
            torch.gather(current_output["assign_probs"], 1, assigned_uavs.unsqueeze(1)).squeeze(1) +
            torch.gather(current_output["divide_probs"], 1, divided_regions.unsqueeze(1)).squeeze(1) +
            torch.gather(current_output["priority_probs"], 1, priority_levels.unsqueeze(1)).squeeze(1)
        ) / 3.0
        
        # 计算损失（均方误差）
        loss = F.mse_loss(current_values, targets)
        
        # 反向传播和优化
        self.high_level_optimizer.zero_grad()
        loss.backward()
        self.high_level_optimizer.step()
        
        return loss.item()
    
    def update_low_level_policy(self, batch_size: int = 64):
        """更新低层策略"""
        if len(self.low_level_buffer) < batch_size:
            return
        
        indices = np.random.choice(len(self.low_level_buffer), batch_size, replace=False)
        batch = [self.low_level_buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([b["state"] for b in batch])).to(self.device)
        task_features = torch.FloatTensor(np.array([b["task_features"] for b in batch])).to(self.device)
        
        actions_idx = torch.LongTensor(np.array([b["action"]["action_idx"] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b["reward"] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b["done"] for b in batch])).to(self.device)
        
        next_states = torch.FloatTensor(np.array([b["next_state"] for b in batch])).to(self.device)
        next_task_features = torch.FloatTensor(np.array([b["next_task_features"] for b in batch])).to(self.device)
        
        # 获取当前动作概率和状态价值
        action_probs, state_values = self.low_level_policy(states, task_features)
        state_values = state_values.squeeze(1)
        
        # 计算下一状态的价值估计
        with torch.no_grad():
            _, next_state_values = self.low_level_policy(next_states, next_task_features)
            next_state_values = next_state_values.squeeze(1)
            
            # 使用贝尔曼方程计算目标价值
            targets = rewards + self.gamma * next_state_values * (1 - dones)
        
        # 计算策略损失和价值损失
        selected_action_probs = torch.gather(action_probs, 1, actions_idx.unsqueeze(1)).squeeze(1)
        advantages = targets - state_values.detach()
        
        # 策略梯度损失（最大化期望奖励）
        policy_loss = -torch.log(selected_action_probs) * advantages
        
        # 价值损失（均方误差）
        value_loss = F.mse_loss(state_values, targets)
        
        # 总损失 = 策略损失 + 价值损失
        loss = policy_loss.mean() + value_loss
        
        # 反向传播和优化
        self.low_level_optimizer.zero_grad()
        loss.backward()
        self.low_level_optimizer.step()
        
        return loss.item()
    
    def save_models(self, high_level_path: str, low_level_path: str):
        """保存模型"""
        torch.save(self.high_level_policy.state_dict(), high_level_path)
        torch.save(self.low_level_policy.state_dict(), low_level_path)
        logger.info(f"模型已保存到 {high_level_path} 和 {low_level_path}")
    
    def load_models(self, high_level_path: str, low_level_path: str):
        """加载模型"""
        self.high_level_policy.load_state_dict(torch.load(high_level_path, map_location=self.device))
        self.low_level_policy.load_state_dict(torch.load(low_level_path, map_location=self.device))
        logger.info(f"模型已从 {high_level_path} 和 {low_level_path} 加载") 