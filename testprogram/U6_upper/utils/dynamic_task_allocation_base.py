import torch
import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Union, Any, Optional
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class U6DroneState:
    """无人机状态类 - U6版本，专为6无人机系统设计"""
    
    def __init__(
        self,
        drone_id: int,
        position: Tuple[float, float],
        max_storage: float = 100.0,
        current_storage: float = 0.0,
        battery: float = 100.0,
        max_speed: float = 5.0,
        sensor_types: List[str] = None
    ):
        """
        初始化无人机状态
        
        Args:
            drone_id: 无人机ID (0-5)
            position: 无人机当前位置 (x, y)
            max_storage: 最大存储容量
            current_storage: 当前已使用存储
            battery: 电池电量百分比
            max_speed: 最大飞行速度
            sensor_types: 搭载的传感器类型列表
        """
        self.drone_id = drone_id
        self.position = position
        self.max_storage = max_storage
        self.current_storage = current_storage
        self.battery = battery
        self.max_speed = max_speed
        self.sensor_types = sensor_types or ["温度", "湿度", "空气质量"]
        
        # 任务相关状态
        self.current_task = None
        self.task_queue = deque()
        self.completed_tasks = []
        self.is_busy = False
        
        # 协作相关状态
        self.collaboration_group = []  # 当前参与协作的无人机ID列表
        self.is_collaborating = False  # 是否正在协作
        self.collaboration_role = None  # 协作角色（主导/辅助）
        
        # 运行时间统计
        self.total_distance = 0.0
        self.total_flight_time = 0.0
        self.idle_time = 0.0
        self.last_update_time = time.time()
    
    def update_position(self, new_position: Tuple[float, float], timestamp: Optional[float] = None):
        """
        更新无人机位置
        
        Args:
            new_position: 新位置坐标 (x, y)
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 计算移动距离
        distance = ((new_position[0] - self.position[0]) ** 2 + 
                   (new_position[1] - self.position[1]) ** 2) ** 0.5
        
        # 更新总飞行距离
        self.total_distance += distance
        
        # 按距离消耗电量 (简化模型)
        self.battery -= distance * 0.1
        self.battery = max(0.0, self.battery)
        
        # 计算飞行时间
        flight_time = distance / self.max_speed
        self.total_flight_time += flight_time
        
        # 更新位置
        self.position = new_position
        self.last_update_time = timestamp
    
    def add_task(self, task):
        """添加任务到队列"""
        self.task_queue.append(task)
    
    def start_task(self, task):
        """开始执行任务"""
        self.current_task = task
        self.is_busy = True
        task.mark_in_progress()
    
    def complete_task(self):
        """完成当前任务"""
        if self.current_task:
            self.current_task.mark_completed()
            self.completed_tasks.append(self.current_task)
            self.current_task = None
            self.is_busy = False
            
            # 如果在协作中，结束协作
            if self.is_collaborating:
                self.end_collaboration()
    
    def start_collaboration(self, collaborators: List[int], role: str = "primary"):
        """
        开始协作
        
        Args:
            collaborators: 协作者ID列表
            role: 协作角色 ("primary" 或 "assistant")
        """
        self.collaboration_group = collaborators
        self.is_collaborating = True
        self.collaboration_role = role
    
    def end_collaboration(self):
        """结束协作"""
        self.collaboration_group = []
        self.is_collaborating = False
        self.collaboration_role = None
    
    def update_storage(self, data_size: float):
        """更新存储空间使用"""
        self.current_storage += data_size
        self.current_storage = min(self.current_storage, self.max_storage)
    
    def get_remaining_storage(self) -> float:
        """获取剩余存储空间"""
        return self.max_storage - self.current_storage
    
    def get_next_task(self):
        """获取队列中的下一个任务"""
        if self.task_queue:
            return self.task_queue.popleft()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """将状态转换为字典表示"""
        return {
            "drone_id": self.drone_id,
            "position": self.position,
            "max_storage": self.max_storage,
            "current_storage": self.current_storage,
            "battery": self.battery,
            "max_speed": self.max_speed,
            "sensor_types": self.sensor_types,
            "is_busy": self.is_busy,
            "is_collaborating": self.is_collaborating,
            "collaboration_role": self.collaboration_role,
            "collaboration_group": self.collaboration_group,
            "total_distance": self.total_distance,
            "total_flight_time": self.total_flight_time,
            "idle_time": self.idle_time,
            "completed_tasks_count": len(self.completed_tasks)
        }
    
    def get_feature_vector(self) -> np.ndarray:
        """获取特征向量用于RL模型"""
        return np.array([
            self.position[0], 
            self.position[1],
            self.current_storage / self.max_storage,
            self.battery / 100.0,
            float(self.is_busy),
            float(self.is_collaborating)
        ])

class U6TaskInfo:
    """任务信息类 - U6版本"""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        position: Tuple[float, float],
        start_time: float,
        end_time: float,
        priority: int = 3,
        data_size: float = 10.0,
        region_id: int = 0,
        requires_sensors: List[str] = None,
        requires_collaboration: bool = False,  # 是否需要协作
        min_collaborators: int = 0,  # 最少协作者数量
        collaboration_benefit: float = 1.0  # 协作带来的效益倍数
    ):
        """
        初始化任务信息
        
        Args:
            task_id: 任务ID
            task_type: 任务类型（数据收集、监测等）
            position: 任务位置坐标
            start_time: 开始时间
            end_time: 结束时间
            priority: 优先级（1-5，5最高）
            data_size: 预计产生的数据量
            region_id: 区域ID
            requires_sensors: 所需传感器类型
            requires_collaboration: 是否需要协作
            min_collaborators: 最少协作者数量
            collaboration_benefit: 协作带来的效益倍数
        """
        self.task_id = task_id
        self.task_type = task_type
        self.position = position
        self.start_time = start_time
        self.end_time = end_time
        self.priority = priority
        self.data_size = data_size
        self.region_id = region_id
        self.requires_sensors = requires_sensors or []
        self.requires_collaboration = requires_collaboration
        self.min_collaborators = min_collaborators
        self.collaboration_benefit = collaboration_benefit
        
        # 任务状态
        self.assigned_drone = None
        self.collaborator_drones = []  # 协作无人机ID列表
        self.is_completed = False
        self.is_in_progress = False
        self.completion_time = None
        self.start_execution_time = None
        self.collaboration_success = None  # 协作是否成功
    
    def assign_to_drone(self, drone_id: int):
        """分配任务给指定无人机"""
        self.assigned_drone = drone_id
    
    def add_collaborator(self, drone_id: int):
        """添加协作无人机"""
        if drone_id not in self.collaborator_drones:
            self.collaborator_drones.append(drone_id)
    
    def mark_in_progress(self):
        """标记任务为进行中"""
        self.is_in_progress = True
        self.start_execution_time = time.time()
    
    def mark_completed(self, completion_time: Optional[float] = None, collaboration_success: Optional[bool] = None):
        """
        标记任务为已完成
        
        Args:
            completion_time: 完成时间
            collaboration_success: 协作是否成功
        """
        self.is_completed = True
        self.is_in_progress = False
        self.completion_time = completion_time or time.time()
        if collaboration_success is not None:
            self.collaboration_success = collaboration_success
    
    def get_time_window_length(self) -> float:
        """获取时间窗口长度（秒）"""
        return self.end_time - self.start_time
    
    def is_expired(self, current_time: float) -> bool:
        """检查任务是否已过期"""
        return current_time > self.end_time and not self.is_completed
    
    def get_urgency(self, current_time: float) -> float:
        """计算任务紧急度（0-1之间，1表示最紧急）"""
        if current_time >= self.end_time:
            return 1.0
        
        time_window = self.get_time_window_length()
        if time_window <= 0:
            return 1.0
        
        time_left = self.end_time - current_time
        return 1.0 - (time_left / time_window)
    
    def has_sufficient_collaborators(self) -> bool:
        """检查是否有足够的协作者"""
        if not self.requires_collaboration:
            return True
        return len(self.collaborator_drones) >= self.min_collaborators
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务信息转换为字典表示"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "position": self.position,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "priority": self.priority,
            "data_size": self.data_size,
            "region_id": self.region_id,
            "requires_sensors": self.requires_sensors,
            "requires_collaboration": self.requires_collaboration,
            "min_collaborators": self.min_collaborators,
            "collaboration_benefit": self.collaboration_benefit,
            "assigned_drone": self.assigned_drone,
            "collaborator_drones": self.collaborator_drones,
            "is_completed": self.is_completed,
            "is_in_progress": self.is_in_progress,
            "completion_time": self.completion_time,
            "start_execution_time": self.start_execution_time,
            "collaboration_success": self.collaboration_success
        }
    
    def get_feature_vector(self) -> np.ndarray:
        """获取特征向量用于RL模型"""
        # 任务类型编码为数字
        type_encoding = {
            "数据收集": 0,
            "监测": 1,
            "巡逻": 2,
            "数据卸载": 3
        }.get(self.task_type, 4)  # 默认为4
        
        return np.array([
            type_encoding / 4.0,  # 归一化任务类型
            self.position[0],
            self.position[1],
            self.data_size / 100.0,  # 归一化数据大小
            self.priority / 5.0,  # 归一化优先级
            self.region_id / 5.0,  # 归一化区域ID
            float(self.requires_collaboration),
            self.min_collaborators / 5.0,  # 归一化最少协作者数量
            self.collaboration_benefit / 2.0  # 归一化协作效益
        ]) 